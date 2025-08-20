# Reference fully to https://github.com/KellerJordan/modded-nanogpt except modeling and masking rule for beacon

import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask
from modeling_beacon_gpt import BeaconGPT
import torch.distributed as dist
import glob
from pathlib import Path
from loguru import logger
import uuid
from muon import MuonWithAuxAdam
from torch.nn.parallel import DistributedDataParallel as DDP

log_uuid = str(uuid.uuid4())

logger.add(f"logs/train_{log_uuid}.log")

## Init distributed

def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:  # single-GPU
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank

RANK, WORLD_SIZE, LOCAL_RANK = setup_distributed()

print(f"RANK: {RANK}, WORLD_SIZE: {WORLD_SIZE}, LOCAL_RANK: {LOCAL_RANK}")


def log_main(message: str):
    if LOCAL_RANK == 0:
        logger.info(message)


num_train_steps = 1750
beacon_offset = 16
use_beacon = True
beacon_token_id = 50258
end_of_document_token_id = 50256
if use_beacon:
    vocab_size = 50259
else:
    vocab_size = 50257


## Data loader, https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py#L528


def _load_data_shard(file: Path):
    header = torch.from_file(
        str(file), False, 256, dtype=torch.int32
    )  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=True
        )  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


# find world_size starting indicies, such that each begins with token 50256 and local_batches don't overlap
def find_batch_starts(
    tokens: torch.Tensor, pos: int, local_batch_size: int, max_batch_span: int
):
    boundary_mask = tokens[pos : pos + max_batch_span] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item()
        if end - start >= local_batch_size:
            starts.append(start)  # append start once end pos is confirmed
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False  # increase max_batch_span if necessary


def distributed_data_generator(
    filename_pattern: str, batch_size: int, align_to_bos: bool, use_beacon: bool = False
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(
        files
    )  # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    max_batch_span = (
        2 * batch_size if align_to_bos else batch_size
    )  # provide buffer to handle samples up to length local_batch_size
    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(
                tokens, pos, local_batch_size, max_batch_span
            )
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size
        buf = tokens[start_idx:][: local_batch_size + 1]
        if use_beacon:
            # Inject beacon tokens every beacon_offset^th from end of document
            eod_mask = buf == end_of_document_token_id
            eod_idxs = torch.where(eod_mask)[0]
            boundary_points = sorted(list(set([0] + eod_idxs.tolist() + [len(buf)])))
            if len(boundary_points) < 2:
                # No change to buf, logic proceeds with the original buffer.
                pass
            else:
                new_buf_parts = []
                for start_idx, end_idx in zip(
                    boundary_points[:-1], boundary_points[1:]
                ):
                    # Get the current document segment
                    segment = buf[start_idx:end_idx]

                    # Skip empty segments which can occur if there are consecutive EOD tokens
                    if len(segment) == 0:
                        continue

                    # 4. Inject beacons into the segment.
                    # This inner loop is the same as your original, but applied to a correctly defined segment.
                    for j in range(0, len(segment), beacon_offset):
                        chunk = segment[j : j + beacon_offset]
                        # To avoid adding a beacon to an empty chunk at the very end
                        if len(chunk) > 0:
                            new_buf_parts.append(chunk)
                            # Don't add a beacon if the last token is already an EOD token.
                            # This is an optional but good practice to avoid ...<EOD><BEACON>... sequences.
                            if not (
                                len(chunk) == beacon_offset
                                and chunk[-1] == end_of_document_token_id
                            ):
                                new_buf_parts.append(
                                    torch.tensor(
                                        [beacon_token_id],
                                        device=buf.device,
                                        dtype=buf.dtype,
                                    )
                                )

            # 5. Reconstruct the buffer if any new parts were created.
            if new_buf_parts:
                buf = torch.cat(new_buf_parts)

        inputs = buf[:-1].to(
            device="cuda", dtype=torch.int32, non_blocking=True
        )  # no sync on host side;
        targets = buf[1:].to(
            device="cuda", dtype=torch.int64, non_blocking=True
        )  # H2D in another stream isn't helpful.
        targets[targets == beacon_token_id] = -100
        pos += batch_span
        yield inputs, targets


local_batch_size = 48 * 1024

train_loader = distributed_data_generator(
    "data/fineweb10B/*train*.bin", local_batch_size * WORLD_SIZE, True, False
)

sample = next(train_loader)


log_main(f"Train loader sample: inputs {sample[0].shape}, targets {sample[1].shape}")
log_main(
    f"Running with pytorch version {torch.__version__} compiled for CUDA {torch.version.cuda}"
)


def nvidia_smi():
    import subprocess  # avoid top level import

    return subprocess.run(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ).stdout


log_main(nvidia_smi())

# Init GPT-2 model config
model: nn.Module = BeaconGPT(
    vocab_size=vocab_size,
    hidden_size=768,
    n_layer=12,
    n_head=6,
    max_seq_len=local_batch_size,
    beacon_offset=beacon_offset,
    use_beacon=use_beacon,
)


log_main(f"model {model}")

model.to("cuda")
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

hidden_weights = [p for p in model.blocks.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.blocks.parameters() if p.ndim < 2]
nonhidden_params = [*model.lm_head.parameters(), *model.wte.parameters()]

log_main(f"hidden_weights {len(hidden_weights)}")
log_main(f"hidden_gains_biases {len(hidden_gains_biases)}")
log_main(f"nonhidden_params {len(nonhidden_params)}")

param_groups = [
    dict(params=hidden_weights, lr=0.05, use_muon=True, weight_decay=0.01, momentum=0.95),
    dict(params=hidden_gains_biases+nonhidden_params, lr=3e-4, use_muon=False, weight_decay=0.01, betas=(0.8, 0.95), eps=1e-10),
]

optimizer = MuonWithAuxAdam(param_groups)

cooldown_frac = 0.45
cooldown_mark = False


def get_lr(step: int):
    global cooldown_mark
    x = step / num_train_steps  # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        if not cooldown_mark:
            cooldown_mark = True
            log_main(f"Starting cooldown, x={x}")
        w = (1 - x) / cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: get_lr(step))

model = DDP(model, device_ids=[LOCAL_RANK])
model: nn.Module = torch.compile(model, dynamic=False)


model.train()


def get_masking_rule(input_seq: torch.Tensor):
    docs = (input_seq == end_of_document_token_id)[0].cumsum(0)
    beacons = (input_seq == beacon_token_id)[0]

    def document_causal(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[q_idx] == docs[kv_idx]
        return causal_mask & document_mask

    def document_causal_beacon(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        is_beacon = beacons[kv_idx]
        is_recent = kv_idx >= q_idx - beacon_offset
        return causal_mask & (is_beacon | is_recent)

    if use_beacon:
        return document_causal_beacon
    else:
        return document_causal


scaler = torch.amp.GradScaler()

train_time_ms = 0.0

for step in range(num_train_steps):
    torch.cuda.synchronize()  # Ensure previous operations are complete
    t0 = time.perf_counter()  # Start timing for this step
    
    inputs, targets = next(train_loader)
    inputs = inputs[None, :]
    targets = targets[None, :]
    masking_rule = get_masking_rule(inputs)
    mask = create_block_mask(
        masking_rule,
        B=None,
        H=None,
        Q_LEN=inputs.size(1),
        KV_LEN=inputs.size(1),
        device=inputs.device,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(inputs, targets, mask)
    scaler.scale(loss).backward()
    
    # Learning rate scheduling
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    model.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()  # Ensure all operations are complete
    if step >= 10:
        step_time_ms = (time.perf_counter() - t0) * 1000
        train_time_ms += step_time_ms
        current_lr = scheduler.get_last_lr()
        log_main(
            f"[step {step}] loss {loss.item():.4f} / step_time {step_time_ms:.2f}ms / avg_time {train_time_ms / (step + 1):.2f}ms / lr {current_lr}"
        )