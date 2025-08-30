import argparse
import datetime
import os
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
from src.modeling.transformer import Transformer, create_block_mask
from src.optimizer.muon import DistAdam, Muon
from src.utils import visualize_attention_scores


@dataclass
class TrainingConfig:
    train_data_pattern: str = "data/dclm/shard_*.bin"
    val_data_pattern: str = "data/dclm/shard_*.bin"
    pretrained_model_name: str = "gpt2"
    batch_size: int = 24 * 1024
    max_steps: int = 5000
    sample_every_n_steps: int = 100
    use_beacon: bool = True
    beacon_stride: int = 16
    sample_text: str = "In the US, "
    project_name: str = "beacon-gpt"
    seed: int = 42

    n_head: int = 6
    n_layer: int = 12
    hidden_size: int = 768

    adam_lr: float = 3e-4
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.1

    muon_lr: float = 0.02
    muon_weight_decay: float = 0.1
    muon_momentum: float = 0.95

    lr_decay_ratio: float = 0.45
    min_lr_ratio: float = 0.1


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


def parse_args():
    parser = argparse.ArgumentParser(description="Beacon GPT Training Script")
    parser.add_argument(
        "--train-data-pattern",
        type=str,
        default="data/dclm/shard_*.bin",
    )
    parser.add_argument(
        "--val-data-pattern",
        type=str,
        default="data/dclm/shard_*.bin",
    )
    parser.add_argument("--pretrained-model-name", type=str, default="gpt2")
    parser.add_argument("--batch-size", type=int, default=24 * 1024)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--sample-every-n-steps", type=int, default=100)
    parser.add_argument("--use-beacon", action="store_true", default=True)
    parser.add_argument("--no-beacon", dest="use_beacon", action="store_false")
    parser.add_argument("--beacon-stride", type=int, default=16)
    parser.add_argument("--sample-text", type=str, default="In the US, ")
    parser.add_argument("--project-name", type=str, default="beacon-gpt")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=768)

    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adam-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-momentum", type=float, default=0.95)

    parser.add_argument("--lr-decay-ratio", type=float, default=0.45)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)

    args = parser.parse_args()
    return args


def args_to_config(args):
    return TrainingConfig(
        train_data_pattern=args.train_data_pattern,
        val_data_pattern=args.val_data_pattern,
        pretrained_model_name=args.pretrained_model_name,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        sample_every_n_steps=args.sample_every_n_steps,
        use_beacon=args.use_beacon,
        beacon_stride=args.beacon_stride,
        sample_text=args.sample_text,
        project_name=args.project_name,
        seed=args.seed,
        n_head=args.n_head,
        n_layer=args.n_layer,
        hidden_size=args.hidden_size,
        adam_lr=args.adam_lr,
        adam_weight_decay=args.adam_weight_decay,
        muon_lr=args.muon_lr,
        muon_weight_decay=args.muon_weight_decay,
        muon_momentum=args.muon_momentum,
        lr_decay_ratio=args.lr_decay_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )


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
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
    tokenizer._special_tokens = {}
    tokenizer.add_special_tokens({"cls_token": "<|beacon|>"})

    train_loader = distributed_data_generator(
        cfg.train_data_pattern,
        batch_size=cfg.batch_size * dist_cfg.world_size,
        prefix_tokens=[tokenizer.bos_token_id],
        local_rank=dist_cfg.rank,
        world_size=dist_cfg.world_size,
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
    cfg: TrainingConfig, dist_cfg: DistributedConfig, tokenizer: AutoTokenizer
):
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        hidden_size=cfg.hidden_size,
        max_seq_len=cfg.batch_size,
        beacon_id=tokenizer.cls_token_id,
        bos_id=tokenizer.bos_token_id,
        beacon_stride=cfg.beacon_stride,
    ).to(dist_cfg.device)

    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()

    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    if dist_cfg.device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    return model


def setup_optimizers(model, cfg: TrainingConfig, tokenizer: AutoTokenizer):
    hidden_params = [p for p in model.blocks.parameters()]
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


def log_config(
    cfg: TrainingConfig, dist_cfg: DistributedConfig, tokenizer: AutoTokenizer
):
    log_master(f"Sample text: {cfg.sample_text}", dist_cfg.is_master)
    log_master(f"Vocab size: {tokenizer.vocab_size}", dist_cfg.is_master)
    log_master(f"Beacon id: {tokenizer.cls_token_id}", dist_cfg.is_master)
    log_master(f"BOS id: {tokenizer.bos_token_id}", dist_cfg.is_master)
    log_master(f"Beacon stride: {cfg.beacon_stride}", dist_cfg.is_master)
    log_master(f"Use beacon: {cfg.use_beacon}", dist_cfg.is_master)
    log_master(f"Batch size: {cfg.batch_size}", dist_cfg.is_master)
    log_master(f"World size: {dist_cfg.world_size}", dist_cfg.is_master)
    log_master(f"Max steps: {cfg.max_steps}", dist_cfg.is_master)


def visualize_initial_sample(
    train_loader,
    cfg: TrainingConfig,
    dist_cfg: DistributedConfig,
    tokenizer: AutoTokenizer,
):
    if not dist_cfg.is_master:
        return

    sample = next(train_loader)[:80]
    print(sample.tolist())
    print(
        inject_beacon_to_docs(
            sample,
            bos_id=tokenizer.bos_token_id,
            beacon_id=tokenizer.cls_token_id,
            stride=cfg.beacon_stride,
        ).tolist()
    )

    fake_q = torch.rand(1, 1, len(sample), 16, device="cpu")
    fake_k = torch.rand(1, 1, len(sample), 16, device="cpu")
    mask_mod = create_block_mask(
        inject_beacon_to_docs(
            sample.cpu(),
            bos_id=tokenizer.bos_token_id,
            beacon_id=tokenizer.cls_token_id,
            stride=cfg.beacon_stride,
        ),
        bos_id=tokenizer.bos_token_id,
        beacon_id=tokenizer.cls_token_id,
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

    mask = create_block_mask(
        input_ids=inputs,
        bos_id=tokenizer.bos_token_id,
        beacon_id=tokenizer.cls_token_id,
        mask_type="beacon_causal_document" if cfg.use_beacon else "causal_document",
    )

    logits, loss = model(inputs, targets, mask)
    return loss


def update_lr(optimizers, step, cfg: TrainingConfig):
    lr_scale = get_const_then_linear_decay_lr(
        step, cfg.max_steps, cfg.lr_decay_ratio, cfg.min_lr_ratio
    )
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale


def main():
    args = parse_args()
    cfg = args_to_config(args)
    dist_cfg = init_distributed()

    torch.manual_seed(cfg.seed)

    if dist_cfg.is_master:
        wandb.init(
            project=cfg.project_name,
            name=f"beacon:{cfg.use_beacon}-stride:{cfg.beacon_stride}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(cfg),
        )

    tokenizer, train_loader, sample_text_ids = setup_data(cfg, dist_cfg)
    log_config(cfg, dist_cfg, tokenizer)

    model = init_model(cfg, dist_cfg, tokenizer)
    optimizers = setup_optimizers(model, cfg, tokenizer)

    visualize_initial_sample(train_loader, cfg, dist_cfg, tokenizer)

    total_step_time_ms = 0.0

    for step in range(cfg.max_steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

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
                sample_text_ids.to(dist_cfg.device),
                max_new_tokens=64,
                use_beacon=cfg.use_beacon,
            )
            sample_text = tokenizer.decode(sample_output.cpu().tolist())
            log_master(f"Sample text: {sample_text}", dist_cfg.is_master)
            wandb.log({"sample_text": wandb.Html(f"<pre>{sample_text}</pre>")})


if __name__ == "__main__":
    main()
