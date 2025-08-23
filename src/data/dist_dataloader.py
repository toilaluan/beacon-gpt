import torch
import torch.distributed as dist
from torch import Tensor
from pathlib import Path
import glob


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
    tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int
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
    filename_pattern: str, batch_size: int, align_to_bos: bool
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
        pos += batch_span
        yield buf
