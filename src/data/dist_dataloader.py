import time
import json
import glob
import random
from pathlib import Path
from typing import Tuple, Optional, List, Generator
from collections import deque

import numpy as np
import torch
from torch import Tensor


def floor_multiple_of_n(v: int, n: int) -> int:
    return (v // n) * n


def _load_shard(shard_path: Path, index_path: Path) -> Tuple[np.ndarray, dict]:
    tokens = np.fromfile(shard_path, dtype=np.uint32)
    with open(index_path, "r") as f:
        index = json.load(f)
    return tokens, index


def _deterministic_shuffle(items: List, seed: int) -> List:
    """Return a deterministically shuffled *copy* of items."""
    out = list(items)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def distributed_data_generator(
    dataset_path: Path,
    batch_size: int,
    prefix_tokens: Optional[List[int]] = None,
    postfix_tokens: Optional[List[int]] = None,
    local_rank: int = 0,
    world_size: int = 1,
    doc_multiple_of_n: int = 1,
    base_seed: int = 0,
) -> Generator[Tensor, None, None]:
    """
    Yields 1D torch.long tensors of length == batch_size for the given local_rank.
    Scheduling:
      - Each document becomes: prefix + doc_body + postfix.
      - Documents are assigned round-robin to rank buffers [0..world_size-1].
      - When *all* buffers have >= batch_size tokens, emit exactly batch_size
        from each (carry-over kept), returning only local_rank's slice.

    Notes:
      - `doc_multiple_of_n` applies to the *raw document body* only.
      - Deterministic shuffle per (epoch, shard) ensures each rank sees the same
        document order and thus different (but aligned) slices by rank.
    """
    assert world_size >= 1, "world_size must be >= 1"
    assert 0 <= local_rank < world_size, "local_rank must be in [0, world_size)"

    shards = sorted(glob.glob(str(dataset_path / "shard_*.bin")))
    indices = sorted(glob.glob(str(dataset_path / "shard_*.idx")))
    assert shards and indices and len(shards) == len(indices), "Missing shards or indices"

    doc_prefix = list(prefix_tokens) if prefix_tokens else []
    doc_postfix = list(postfix_tokens) if postfix_tokens else []
    epoch = 0

    while True:
        for shard_i, (shard_file, index_file) in enumerate(zip(shards, indices)):
            print(f"[rank {local_rank}] Loading shard: {shard_file}")
            t0 = time.time()
            tokens, index = _load_shard(Path(shard_file), Path(index_file))
            print(f"[rank {local_rank}] Load time: {time.time() - t0:.3f}s")

            # Deterministic per-(epoch, shard) shuffle
            docs = index["documents"]
            seed = (base_seed << 32) ^ (epoch * 1315423911) ^ (shard_i * 2654435761)
            docs = _deterministic_shuffle(docs, seed)

            rank_starts = [[] for _ in range(world_size)]
            rank_lengths = [0] * world_size

            processing_rank = 0

            for doc_pos in docs:
                start_doc = int(doc_pos["start"])
                end_doc = int(doc_pos["end"])
                raw_len = end_doc - start_doc
                if raw_len <= 0:
                    continue

                body_len = raw_len
                if doc_multiple_of_n > 1:
                    body_len = floor_multiple_of_n(body_len, doc_multiple_of_n)
                    if body_len <= 0:
                        continue
                rank_starts[processing_rank].append((start_doc, body_len))
                rank_lengths[processing_rank] += body_len
                
                if rank_lengths[processing_rank] > batch_size:
                    processing_rank += 1
                else:
                    continue
                if processing_rank == world_size:
                    # print("Local rank x", local_rank)
                    local_rank_docs = [tokens[start:start+length] for start, length in rank_starts[local_rank]]
                    return_ids = []
                    for doc in local_rank_docs:
                        return_ids.extend(doc_prefix)
                        return_ids.extend(doc)
                        return_ids.extend(doc_postfix)
                    yield torch.tensor(return_ids, dtype=torch.int32)
                    rank_starts = [[] for _ in range(world_size)]
                    rank_lengths = [0] * world_size
                    processing_rank = 0

        epoch += 1


if __name__ == "__main__":
    from transformers import AutoTokenizer

    dataset_path = Path("./tokenized_data")
    batch_size = 16*1024
    prefix_tokens = [50256]           # e.g., BOS
    postfix_tokens = []               # optional

    # Example: simulate rank 0 of 4. Launch 4 procs with local_rank=0..3 for real use.
    local_rank = 0
    world_size = 4

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    iters = 0
    for ids in distributed_data_generator(
        dataset_path=dataset_path,
        batch_size=batch_size+128,
        prefix_tokens=prefix_tokens,
        postfix_tokens=postfix_tokens,
        local_rank=2,
        world_size=8,
        doc_multiple_of_n=16,
        base_seed=1234,   # set the same across ranks for aligned ordering
    ):
        print(ids.shape)
        if iters > 10:
            break
        iters += 1
