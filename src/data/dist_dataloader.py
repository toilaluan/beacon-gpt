import torch
import torch.distributed as dist
from torch import Tensor
from pathlib import Path
import glob
import numpy as np
from typing import Tuple
import json
from typing import Optional, List, Generator
from itertools import cycle
import random


def floor_multiple_of_n(v: int, n: int) -> int:
    return (v // n) * n


def _load_shard(shard_path: Path, index_path: Path) -> Tuple[np.ndarray, dict]:
    tokens = np.fromfile(shard_path, dtype=np.uint32)
    with open(index_path, "r") as f:
        index = json.load(f)

    return tokens, index


def distributed_data_generator(
    dataset_path: Path,
    batch_size: int,
    prefix_tokens: Optional[List[int]] = None,
    local_rank: int = 0,
    world_size: int = 1,
    doc_multiple_of_n: int = 16,
):
    shards = sorted(glob.glob(str(dataset_path / "shard_*.bin")))[
        local_rank::world_size
    ]
    indices = sorted(glob.glob(str(dataset_path / "shard_*.idx")))[
        local_rank::world_size
    ]

    # freeze the prefix so we don't mutate the caller's list
    base_prefix = tuple(prefix_tokens) if prefix_tokens is not None else None

    def new_batch():
        return [] if base_prefix is None else list(base_prefix)

    for shard_file, index_file in cycle(zip(shards, indices)):
        tokens, index = _load_shard(Path(shard_file), Path(index_file))
        batch_tokens = new_batch()

        random.shuffle(index["documents"])

        for doc_pos in index["documents"]:
            start_doc = doc_pos["start"]
            end_doc = doc_pos["end"]
            length = floor_multiple_of_n(end_doc - start_doc, doc_multiple_of_n)
            if length <= 0:
                continue
            doc_tokens = tokens[start_doc : start_doc + length]

            # extend safely; batch_tokens is a fresh list for each batch
            batch_tokens.extend(doc_tokens.tolist())

            if len(batch_tokens) >= batch_size:
                yield torch.tensor(batch_tokens[:batch_size], dtype=torch.long)
                batch_tokens = new_batch()

        if len(batch_tokens) > 0:
            yield torch.tensor(batch_tokens[:batch_size], dtype=torch.long)


if __name__ == "__main__":
    import time
    from transformers import AutoTokenizer

    dataset_path = Path("./tokenized_data")
    batch_size = 1 * 1024
    prefix_tokens = [50256]
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    iters = 0
    for ids in distributed_data_generator(
        dataset_path, batch_size, prefix_tokens, local_rank=0, world_size=1
    ):
        start = time.perf_counter()
        print(ids[:16])
        print(tokenizer.decode(ids[:32]))
        print(f"Time taken: {time.perf_counter() - start} seconds")
        if iters > 10:
            break
        iters += 1
