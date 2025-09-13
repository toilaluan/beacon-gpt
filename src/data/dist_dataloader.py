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
    postfix_tokens: Optional[List[int]] = None,
    local_rank: int = 0,
    world_size: int = 1,
    doc_multiple_of_n: int = 16,
):
    """
    Yields 1D torch.long tensors of length <= batch_size.
    Each document is transformed to: prefix_tokens + doc_tokens + postfix_tokens
    before concatenation into the running buffer.

    Notes:
      - We carry over overflow (no token loss).
      - `doc_multiple_of_n` is applied to the *raw doc body* only.
    """
    shards = sorted(glob.glob(str(dataset_path / "shard_*.bin")))[local_rank::world_size]
    indices = sorted(glob.glob(str(dataset_path / "shard_*.idx")))[local_rank::world_size]

    doc_prefix = list(prefix_tokens) if prefix_tokens else []
    doc_postfix = list(postfix_tokens) if postfix_tokens else []

    buffer: List[int] = []

    for shard_file, index_file in cycle(zip(shards, indices)):
        print(f"Loading new shard: {shard_file}")
        tokens, index = _load_shard(Path(shard_file), Path(index_file))

        random.shuffle(index["documents"])

        for doc_pos in index["documents"]:
            start_doc = doc_pos["start"]
            end_doc = doc_pos["end"]

            # Truncate the **document body** to a multiple of n
            length = floor_multiple_of_n(end_doc - start_doc, doc_multiple_of_n)
            if length <= 0:
                continue

            doc_body = tokens[start_doc : start_doc + length].tolist()

            # Wrap each doc: [prefix] + body + [postfix]
            buffer.extend(doc_prefix)
            buffer.extend(doc_body)
            buffer.extend(doc_postfix)

            # Emit full batches; keep overflow for the next batch
            while len(buffer) >= batch_size:
                yield torch.tensor(buffer[:batch_size], dtype=torch.long)
                buffer = []

        # End of shard: flush any remainder (may be shorter than batch_size)
        if buffer:
            yield torch.tensor(buffer[:batch_size], dtype=torch.long)
            buffer = []


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
