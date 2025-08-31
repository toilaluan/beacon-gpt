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
    tokens = np.fromfile(shard_path, dtype=np.uint16)
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
) -> Generator[Tensor, None, None]:

    shards = glob.glob(str(dataset_path / "shard_*.bin"))[local_rank::world_size]
    indices = glob.glob(str(dataset_path / "shard_*.idx"))[local_rank::world_size]

    shards.sort()
    indices.sort()

    for shard_file, index_file in cycle(zip(shards, indices)):
        tokens, index = _load_shard(shard_file, index_file)

        batch_tokens = [] if prefix_tokens is None else prefix_tokens

        random.shuffle(index["documents"])

        for doc_pos in index["documents"]:
            start_doc = doc_pos["start"]
            end_doc = doc_pos["end"]
            length = end_doc - start_doc
            length = floor_multiple_of_n(length, doc_multiple_of_n)
            end_doc = start_doc + length

            doc_tokens = tokens[start_doc:end_doc]

            batch_tokens.extend(doc_tokens)

            if len(batch_tokens) >= batch_size:
                yield torch.tensor(batch_tokens, dtype=torch.int32)[:batch_size]
                batch_tokens = [] if prefix_tokens is None else prefix_tokens

        if len(batch_tokens) > 0:
            yield torch.tensor(batch_tokens, dtype=torch.int32)[:batch_size]


if __name__ == "__main__":
    import time

    dataset_path = Path("data/dclm")
    batch_size = 48 * 1024
    prefix_tokens = [50256]
    iters = 0
    for ids in distributed_data_generator(
        dataset_path, batch_size, prefix_tokens, local_rank=0, world_size=1
    ):
        start = time.perf_counter()
        print(ids.shape)
        print(f"Time taken: {time.perf_counter() - start} seconds")
        if iters > 10:
            break
        iters += 1
