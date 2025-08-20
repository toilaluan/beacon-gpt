# Reference fully to https://github.com/KellerJordan/modded-nanogpt except modeling and masking rule for beacon

import os
import math
import time
import argparse
import yaml
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from muon import MuonWithAuxAdam
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.attention.flex_attention import create_block_mask
from datasets import load_dataset
import tiktoken
from torch.nn.attention.flex_attention import BlockMask
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from modeling_beacon_gpt import BeaconGPT
import torch.distributed as dist
import glob
from pathlib import Path
from loguru import logger
import uuid

log_uuid = str(uuid.uuid4())

logger.add(f"logs/train_{log_uuid}.log")

## Init distributed

dist.init_process_group(backend="nccl")
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
dist.barrier()


def log_main(message: str):
    if RANK == 0:
        logger.info(message)


num_train_steps = 1750
beacon_offset = 16
use_beacon = False
beacon_token_id = 50257
end_of_document_token_id = 50256
if use_beacon:
    vocab_size = 50258
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
    "data/fineweb10B/*train*.bin", local_batch_size * WORLD_SIZE, False
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


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [
                torch.zeros_like(params[-1])
            ] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(
                    dist.reduce_scatter(
                        grad,
                        grad_pad[base_i : base_i + world_size],
                        op=dist.ReduceOp.AVG,
                        async_op=True,
                    ).get_future()
                )

        idx = 0
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = (
                        group["lr"]
                        * max(1, p.size(-2) / p.size(-1)) ** 0.5
                        * getattr(p, "lr_mul", 1.0)
                    )
                    eff_weight_decay = (
                        group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    )
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                        async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(all_reduce_futures).wait()


class DistAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(
                        grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                    ).get_future()
                )
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            params = group["params"]
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]
                lr = group["lr"] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p_slice)
                    state["exp_avg_sq"] = torch.zeros_like(p_slice)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                )
        torch.futures.collect_all(all_reduce_futures).wait()


# Init GPT-2 model config
model: nn.Module = BeaconGPT(
    vocab_size=50257,
    hidden_size=768,
    n_layer=12,
    n_head=6,
    max_seq_len=1024,
    dropout=0.1,
    beacon_offset=beacon_offset,
    use_beacon=use_beacon,
)

# Use Muon only for hidden params

hidden_matrix_params = [
    p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n
]

embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=0.008,
    betas=(0.8, 0.95),
    eps=1e-10,
    weight_decay=0.0,
)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

cooldown_frac = 0.45
cooldown_mark = False


def get_lr(step: int):
    global cooldown_mark
    x = step / num_train_steps  # progress in training
    assert 0 <= x < 1
    if x < 1 - 0.45:
        return 1.0
    else:
        if not cooldown_mark:
            cooldown_mark = True
            log_main(f"Starting cooldown, x={x}")
        w = (1 - x) / 0.45
        return w * 1.0 + (1 - w) * 0.1


model: nn.Module = torch.compile(model, dynamic=False)


def get_masking_rule(input_seq: torch.Tensor):
    docs = (input_seq == end_of_document_token_id).cumsum(0)
    beacons = input_seq == beacon_token_id

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


training_time_ms = 0.0
torch.cuda.synchronize()
t0 = time.perf_counter()

for step in range(num_train_steps):
    inputs, targets = next(train_loader)
    masking_rule = get_masking_rule(inputs)
    mask = create_block_mask(
        masking_rule,
        B=None,
        H=None,
        Q_LEN=inputs.size(1),
        KV_LEN=inputs.size(1),
        device=inputs.device,
    )
    loss = model(inputs, targets, mask)
    loss.backward()
    for opt in optimizers:
        opt.step()
    torch.cuda.synchronize()
    training_time_ms += time.perf_counter() - t0
    t0 = time.perf_counter()

    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)  # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    log_main(
        f"[step {step}] loss {loss.item():.4f} / time {training_time_ms:.2f}ms / avg_step_time {training_time_ms / (step + 1):.2f}ms"
    )
