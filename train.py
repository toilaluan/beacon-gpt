import argparse
import datetime
import os
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.distributed as dist
import wandb
from loguru import logger
from transformers import AutoTokenizer

from src.data.beacon_injecting import inject_beacon_to_docs
from src.data.dist_dataloader import distributed_data_generator
from src.modeling.transformer import TransformerModel, TransformerConfig, make_block_mask
from src.optimizer.muon import DistAdam, Muon
from src.utils import visualize_attention_scores

DEBUG_MODE = os.getenv("TRAIN_MODE") == "overfit"

ARCH_ARGS = {
    "gemma-270m": {
        "head_dim": 256,
        "hidden_size": 640,
        "intermediate_size": 2048,
        "max_position_embeddings": 32768,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "num_hidden_layers": 18,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "initializer_range": 0.02,
        "query_pre_attn_scalar": 256,
    }
}


@dataclass
class TrainingConfig:
    train_data_pattern: str = Path("tokenized_data")
    arch_name: str = "gemma-270m"
    pretrained_tokenizer_name: str = "google/gemma-3-270m"
    batch_size: int = 1 * 1024
    target_tokens: int = 100_000_000
    sample_every_n_steps: int = 100
    use_beacon: bool = True
    beacon_stride: int = 16
    sample_text: str = "In the US, "
    project_name: str = "beacon-gpt"
    seed: int = 42

    adam_lr: float = 3e-4
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.1

    muon_lr: float = 0.02
    muon_weight_decay: float = 0.1
    muon_momentum: float = 0.95

    lr_decay_ratio: float = 0.45
    min_lr_ratio: float = 0.1
    max_steps: int = 1000000  # will be updated later


@dataclass
class DistributedConfig:
    rank: int = field(default_factory=lambda: int(os.environ.get("RANK", 0)))
    world_size: int = field(
        default_factory=lambda: int(os.environ.get("WORLD_SIZE", 1))
    )
    device: torch.device = field(init=False)
    is_master: bool = field(init=False)

    def __post_init__(self):
        self.device = torch.device("cuda", self.rank)
        self.is_master = self.rank == 0

def init_distributed():
    dist_cfg = DistributedConfig()
    torch.cuda.set_device(dist_cfg.device)
    dist.init_process_group(backend="nccl", device_id=dist_cfg.device)
    dist.barrier()
    return dist_cfg


def log_master(message, is_master):
    if is_master:
        logger.info(message)


def get_const_then_linear_decay_lr(step, max_steps, decaying_ratio=0.45, min_lr=0.1):
    x = step / max_steps
    if x < 1 - decaying_ratio:
        return 1.0
    else:
        w = (1 - x) / decaying_ratio
        return 1.0 * w + (1 - w) * min_lr


def setup_data(cfg: TrainingConfig, dist_cfg: DistributedConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_tokenizer_name)
    tokenizer._special_tokens = {}
    n = tokenizer.add_special_tokens({"cls_token": "<|beacon|>"})
    train_loader = distributed_data_generator(
        cfg.train_data_pattern,
        batch_size=cfg.batch_size * dist_cfg.world_size,
        prefix_tokens=[tokenizer.bos_token_id],
        local_rank=dist_cfg.rank,
        world_size=dist_cfg.world_size,
        doc_multiple_of_n=cfg.beacon_stride,
    )

    sample_text_ids = tokenizer.encode(cfg.sample_text)
    sample_text_ids = torch.tensor(
        [tokenizer.bos_token_id] + sample_text_ids, dtype=torch.int32
    )

    if cfg.use_beacon:
        sample_text_ids = inject_beacon_to_docs(
            sample_text_ids,
            bos_id=tokenizer.bos_token_id,
            beacon_id=tokenizer.cls_token_id,
            stride=cfg.beacon_stride,
        )

    return tokenizer, train_loader, sample_text_ids


def init_model(
    cfg: TrainingConfig, dist_cfg: DistributedConfig, tokenizer: AutoTokenizer, model_cfg: TransformerConfig
):
    model = TransformerModel(
        config=model_cfg,
        beacon_token_id=tokenizer.cls_token_id,
        beacon_stride=cfg.beacon_stride,
    ).to(dist_cfg.device)

    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()

    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    model = torch.compile(model, dynamic=False)

    return model


def setup_optimizers(model: TransformerModel, cfg: TrainingConfig, tokenizer: AutoTokenizer):
    hidden_params = [p for p in model.layers.parameters()]
    direct_params = [p for p in model.embed_tokens.parameters()] + [
        p for p in model.lm_head.parameters()
    ]

    adam_optimizer = DistAdam(
        direct_params,
        lr=cfg.adam_lr,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_weight_decay,
    )
    muon_optimizer = Muon(
        hidden_params,
        lr=cfg.muon_lr,
        weight_decay=cfg.muon_weight_decay,
        momentum=cfg.muon_momentum,
    )

    optimizers = [adam_optimizer, muon_optimizer]

    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return optimizers


def visualize_initial_sample(
    train_loader,
    cfg: TrainingConfig,
    dist_cfg: DistributedConfig,
    tokenizer: AutoTokenizer,
):
    if not dist_cfg.is_master:
        return

    sample = next(train_loader)[:80]
    log_master(f"Sample IDS: sample.tolist()", dist_cfg.is_master)
    log_master(f"Sample text: {tokenizer.decode(sample.tolist())}", dist_cfg.is_master)
    log_master(
        inject_beacon_to_docs(
            sample,
            bos_id=tokenizer.bos_token_id,
            beacon_id=tokenizer.cls_token_id,
            stride=cfg.beacon_stride,
        ).tolist(),
        dist_cfg.is_master,
    )

    fake_q = torch.rand(1, 1, len(sample), 16, device="cpu")
    fake_k = torch.rand(1, 1, len(sample), 16, device="cpu")
    mask_mod = make_block_mask(
        inject_beacon_to_docs(
            sample.cpu(),
            bos_id=tokenizer.bos_token_id,
            beacon_id=tokenizer.cls_token_id,
            stride=cfg.beacon_stride,
        ),
        bos_token_id=tokenizer.bos_token_id,
        beacon_token_id=tokenizer.cls_token_id,
        mask_type="beacon_causal_document",
        return_block_mask=False,
    )
    visualize_attention_scores(fake_q, fake_k, mask_mod=mask_mod, device="cpu")


def train_step(
    model,
    ids,
    cfg: TrainingConfig,
    dist_cfg: DistributedConfig,
    tokenizer: AutoTokenizer,
):
    if cfg.use_beacon:
        ids = inject_beacon_to_docs(
            ids,
            bos_id=tokenizer.bos_token_id,
            beacon_id=tokenizer.cls_token_id,
            stride=cfg.beacon_stride,
        )

    ids = ids[: cfg.batch_size + 1]
    inputs = ids[:-1].to(dist_cfg.device, dtype=torch.int32)
    targets = ids[1:].to(dist_cfg.device, dtype=torch.int64)
    targets[targets == tokenizer.cls_token_id] = -100
    mask = make_block_mask(
        input_ids=inputs,
        bos_token_id=tokenizer.bos_token_id,
        beacon_token_id=tokenizer.cls_token_id,
        mask_type="beacon_causal_document" if cfg.use_beacon else "causal_document",
    )

    _, loss = model(inputs, targets, mask)
    return loss


def update_lr(optimizers, step, cfg: TrainingConfig):
    lr_scale = get_const_then_linear_decay_lr(
        step, cfg.max_steps, cfg.lr_decay_ratio, cfg.min_lr_ratio
    )
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale


def main():
    cfg = TrainingConfig()
    dist_cfg = init_distributed()

    torch.manual_seed(cfg.seed)

    if dist_cfg.is_master:
        wandb.init(
            project=cfg.project_name,
            name=f"beacon:{cfg.use_beacon}-stride:{cfg.beacon_stride}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(cfg),
        )

    tokenizer, train_loader, sample_text_ids = setup_data(cfg, dist_cfg)
    model_cfg = TransformerConfig(
        **ARCH_ARGS["gemma-270m"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=tokenizer.vocab_size
    )

    model = init_model(cfg, dist_cfg, tokenizer, model_cfg)
    log_master(model, dist_cfg.is_master)
    optimizers = setup_optimizers(model, cfg, tokenizer)

    visualize_initial_sample(train_loader, cfg, dist_cfg, tokenizer)

    total_step_time_ms = 0.0

    sample_ids = next(train_loader)

    max_steps = cfg.target_tokens // (cfg.batch_size * dist_cfg.world_size)
    cfg.max_steps = max_steps

    for step in range(cfg.max_steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if DEBUG_MODE:
            ids = sample_ids
        else:
            ids = next(train_loader)
        assert ids.ndim == 1, "ids must be a 1D tensor"

        loss = train_step(model, ids, cfg, dist_cfg, tokenizer)
        loss.backward()

        update_lr(optimizers, step, cfg)

        for opt in optimizers:
            opt.step()

        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000
        total_step_time_ms += step_time_ms
        avg_step_time = total_step_time_ms / (step + 1)

        log_master(
            f"Step {step}, avg_step_time {avg_step_time:.2f} ms, train_loss {loss.item() / cfg.batch_size:.4f}",
            dist_cfg.is_master,
        )

        if dist_cfg.is_master:
            wandb.log(
                {
                    "train_loss": loss.item() / cfg.batch_size,
                    "step_time": avg_step_time,
                    "step": step,
                }
            )

        if step % cfg.sample_every_n_steps == 0 and dist_cfg.is_master:
            sample_output = model.generate(
                sample_ids[:6].to(dist_cfg.device),
                max_new_tokens=32,
                use_beacon=cfg.use_beacon,
            )
            sample_text = tokenizer.decode(sample_output.cpu().tolist())
            log_master(f"Sample text: {sample_text}", dist_cfg.is_master)
            wandb.log({"sample_text": wandb.Html(f"<pre>{sample_text}</pre>")})


if __name__ == "__main__":
    main()
