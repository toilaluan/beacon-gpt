import tiktoken
from src.modeling.transformer import Transformer, next_multiple_of_n, create_block_mask
from src.data.dist_dataloader import distributed_data_generator
from src.data.beacon_injecting import inject_beacon_to_docs
from src.utils import visualize_attention_scores
from src.optimizer.muon import DistAdam, Muon
import torch
import torch.distributed as dist
import os
from loguru import logger
import time
import wandb
import datetime

# Distributed Init

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DEVICE = torch.device("cuda", RANK)
torch.cuda.set_device(DEVICE)
dist.init_process_group(
    backend="nccl", device_id=DEVICE,
)
dist.barrier()
IS_MASTER = RANK == 0


def log_master(message):
    if IS_MASTER:
        logger.info(message)


def get_const_then_linear_decay_lr(
    step: int, max_steps: int, decaying_ratio: float = 0.45, min_lr: float = 0.1
):
    x = step / max_steps
    if x < 1 - decaying_ratio:
        return 1.0
    else:
        w = (1 - x) / decaying_ratio
        return 1.0 * w + (1 - w) * min_lr


TRAIN_DATA_PATTERN = "scripts/data/fineweb10B/fineweb_train_*.bin"
VAL_DATA_PATTERN = "scripts/data/fineweb10B/fineweb_val_*.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 24 * 1024
TOKENIZER = tiktoken.get_encoding("gpt2")
TOKENIZER._special_tokens = {}
TRAIN_LOADER = distributed_data_generator(
    TRAIN_DATA_PATTERN, batch_size=BATCH_SIZE * WORLD_SIZE, align_to_bos=True
)
SAMPLE_TEXT = "In the US, "
SAMPLE_TEXT_IDS = TOKENIZER.encode(SAMPLE_TEXT)
USE_BEACON = True
BEACON_STRIDE = 16
BOS_ID = 50256
BEACON_ID = 50257
SAMPLE_TEXT_IDS = torch.tensor([BOS_ID]+SAMPLE_TEXT_IDS, dtype=torch.int32)
VOCAB_SIZE = next_multiple_of_n(TOKENIZER.n_vocab, n=16)
MAX_STEPS = 5000
DECAYING_RATIO = 0.45
MIN_LR = 0.1
SAMPLE_EVERY_N_STEPS = 100

if USE_BEACON:
    SAMPLE_TEXT_IDS = inject_beacon_to_docs(
        SAMPLE_TEXT_IDS, bos_id=BOS_ID, beacon_id=BEACON_ID, stride=BEACON_STRIDE
    )
if IS_MASTER:
    wandb.init(project="beacon-gpt", name=f"beacon:{USE_BEACON}-beacon_stride:{BEACON_STRIDE}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
log_master(f"Sample text: {SAMPLE_TEXT}")
log_master(f"Sample text ids: {SAMPLE_TEXT_IDS}")
log_master(f"Vocab size: {VOCAB_SIZE}")
log_master(f"Beacon id: {BEACON_ID}")
log_master(f"BOS id: {BOS_ID}")
log_master(f"Beacon stride: {BEACON_STRIDE}")
log_master(f"Use beacon: {USE_BEACON}")
log_master(f"Batch size: {BATCH_SIZE}")
log_master(f"World size: {WORLD_SIZE}")


MODEL = Transformer(
    vocab_size=VOCAB_SIZE,
    n_head=6,
    n_layer=12,
    hidden_size=768,
    max_seq_len=BATCH_SIZE,
    beacon_id=BEACON_ID,
    bos_id=BOS_ID,
    beacon_stride=BEACON_STRIDE,
).to(DEVICE)

for m in MODEL.modules():
    if isinstance(m, torch.nn.Embedding):
        m.bfloat16()
for param in MODEL.parameters():
    dist.broadcast(param.detach(), 0)

hidden_params = [p for p in MODEL.blocks.parameters()]
direct_params = [p for p in MODEL.embed_tokens.parameters()] + [
    p for p in MODEL.lm_head.parameters()
]

adam_optimizer = DistAdam(
    direct_params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
)
muon_optimizer = Muon(hidden_params, lr=0.02, weight_decay=0.1, momentum=0.95)
optimizers = [adam_optimizer, muon_optimizer]

for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

if DEVICE == "cuda":
    model = torch.compile(MODEL, dynamic=False)


sample = next(TRAIN_LOADER)[:80]
if IS_MASTER:
    print(sample.tolist())
    print(inject_beacon_to_docs(sample, bos_id=BOS_ID, beacon_id=BEACON_ID, stride=BEACON_STRIDE).tolist())
    fake_q = torch.rand(1, 1, len(sample), 16, device="cpu")
    fake_k = torch.rand(1, 1, len(sample), 16, device="cpu")
    mask_mod = create_block_mask(
        inject_beacon_to_docs(sample.cpu(), bos_id=BOS_ID, beacon_id=BEACON_ID, stride=BEACON_STRIDE),
        bos_id=BOS_ID,
        beacon_id=BEACON_ID,
        mask_type="beacon_causal_document",
        return_block_mask=False,
    )
    visualize_attention_scores(fake_q, fake_k, mask_mod=mask_mod, device="cpu")
    # print(TOKENIZER.decode(inject_beacon_to_docs(sample.cpu(), bos_id=BOS_ID, beacon_id=BEACON_ID, stride=BEACON_STRIDE).cpu().tolist()))


## TRAINING LOOP

total_step_time_ms = 0.0

for step in range(MAX_STEPS + 1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids = next(TRAIN_LOADER)
    assert ids.ndim == 1, "ids must be a 1D tensor"
    if USE_BEACON:
        ids = inject_beacon_to_docs(
            ids, bos_id=BOS_ID, beacon_id=BEACON_ID, stride=BEACON_STRIDE
        )
    ids = ids[: BATCH_SIZE + 1]
    inputs = ids[:-1].to(DEVICE, dtype=torch.int32)
    targets = ids[1:].to(DEVICE, dtype=torch.int64)
    targets[targets == BEACON_ID] = -100
    mask = create_block_mask(
        input_ids=inputs,
        bos_id=BOS_ID,
        beacon_id=BEACON_ID,
        mask_type="beacon_causal_document" if USE_BEACON else "causal_document",
    )
    logits, loss = model(inputs, targets, mask)
    loss.backward()

    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_const_then_linear_decay_lr(step, MAX_STEPS)
    for opt in optimizers:
        opt.step()
    MODEL.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    total_step_time_ms += (time.perf_counter() - t0) * 1000
    log_master(
        f"Step {step}, avg_step_time {total_step_time_ms / (step+1)} ms, train_loss {loss.item() / BATCH_SIZE}"
    )
    if IS_MASTER:
        wandb.log({
            "train_loss": loss.item() / BATCH_SIZE,
            "step_time": total_step_time_ms / (step+1),
            "step": step,
        })

    if step % SAMPLE_EVERY_N_STEPS == 0 and IS_MASTER:
        sample_text = TOKENIZER.decode(
            MODEL.generate(SAMPLE_TEXT_IDS.to(DEVICE), max_new_tokens=64, use_beacon=USE_BEACON).cpu().tolist()
        )
        log_master(f"Sample text: {sample_text}")
