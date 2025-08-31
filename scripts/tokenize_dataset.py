import argparse
import os
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Global variables for multiprocessing
_tokenizer = None


def init_worker(tokenizer_name: str):
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _tokenizer.model_max_length = int(1e9)
    if _tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} must have an EOS token.")


def tokenize_doc(doc) -> Tuple[np.ndarray, int]:
    """
    Tokenize a single document and append EOS token.

    Returns:
        Tuple of (tokenized array, document ID if available)
    """
    global _tokenizer
    text = doc.get("text", "")
    doc_id = doc.get("id", -1)  # Use -1 if no ID field

    if not text or not isinstance(text, str):
        return np.array([], dtype=np.uint16), doc_id

    tokens = _tokenizer.encode(text, add_special_tokens=False)
    tokens.append(_tokenizer.eos_token_id)

    tokens_array = np.array(tokens, dtype=np.uint32)

    # Validate token range
    if not ((0 <= tokens_array) & (tokens_array < 2**32)).all():
        raise ValueError(
            f"Token IDs exceed uint16 range. Vocab size: {_tokenizer.vocab_size}"
        )

    return tokens_array, doc_id


class ShardWriter:
    """Handles writing shards and their index files."""

    def __init__(self, output_dir: str, shard_size_mb: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert MB to approximate number of uint16 tokens
        # 1 MB = 1024 * 1024 bytes, uint16 = 2 bytes
        self.shard_size_tokens = (shard_size_mb * 1024 * 1024) // 2

        self.current_shard_idx = 0
        self.current_buffer = []
        self.current_buffer_size = 0
        self.current_index = []
        self.global_token_count = 0

    def add_document(self, tokens: np.ndarray, doc_id: Optional[int] = None) -> None:
        """Add a tokenized document to the current shard."""
        if len(tokens) == 0:
            return

        # Record index entry: (start_pos, end_pos, doc_id)
        start_pos = self.current_buffer_size
        end_pos = start_pos + len(tokens)

        self.current_index.append(
            {
                "start": start_pos,
                "end": end_pos,
                "length": len(tokens),
                "doc_id": doc_id if doc_id != -1 else None,
            }
        )

        self.current_buffer.append(tokens)
        self.current_buffer_size += len(tokens)
        self.global_token_count += len(tokens)

        # Check if we should write the shard
        if self.current_buffer_size >= self.shard_size_tokens:
            self._write_shard()

    def _write_shard(self) -> None:
        """Write current buffer as a shard with its index."""
        if not self.current_buffer:
            return

        # Concatenate all tokens in buffer
        shard_data = np.concatenate(self.current_buffer)

        # Write binary shard file
        shard_path = self.output_dir / f"shard_{self.current_shard_idx:06d}.bin"
        shard_data.tofile(shard_path)

        # Write index file
        index_path = self.output_dir / f"shard_{self.current_shard_idx:06d}.idx"
        index_data = {
            "shard_id": self.current_shard_idx,
            "total_tokens": self.current_buffer_size,
            "num_documents": len(self.current_index),
            "documents": self.current_index,
        }

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        tqdm.write(
            f"Wrote shard {self.current_shard_idx}: "
            f"{self.current_buffer_size:,} tokens, "
            f"{len(self.current_index)} documents"
        )

        # Reset for next shard
        self.current_shard_idx += 1
        self.current_buffer = []
        self.current_buffer_size = 0
        self.current_index = []

    def finalize(self) -> None:
        """Write any remaining data and create metadata file."""
        # Write remaining buffer if not empty
        if self.current_buffer:
            self._write_shard()

        # Write global metadata
        metadata = {
            "total_shards": self.current_shard_idx,
            "total_tokens": self.global_token_count,
            "shard_size_mb": (self.shard_size_tokens * 2) // (1024 * 1024),
            "dtype": "uint16",
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nTokenization complete!")
        print(f"Total shards: {self.current_shard_idx}")
        print(f"Total tokens: {self.global_token_count:,}")
        print(f"Output directory: {self.output_dir}")


def load_shard(shard_path: Path, index_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load a shard and its index.

    Returns:
        Tuple of (token array, index dictionary)
    """
    # Load binary data
    tokens = np.fromfile(shard_path, dtype=np.uint16)

    # Load index
    with open(index_path, "r") as f:
        index = json.load(f)

    return tokens, index


def get_document_from_shard(
    tokens: np.ndarray, index: dict, doc_idx: int
) -> np.ndarray:
    """
    Extract a specific document from a shard.

    Args:
        tokens: The shard's token array
        index: The shard's index dictionary
        doc_idx: Index of the document within the shard

    Returns:
        Token array for the specified document
    """
    if doc_idx >= len(index["documents"]):
        raise IndexError(f"Document index {doc_idx} out of range")

    doc_info = index["documents"][doc_idx]
    return tokens[doc_info["start"] : doc_info["end"]]


def main(args):
    """Main tokenization pipeline."""
    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max tokens: {args.max_tokens:,}")
    print(f"Shard size: {args.shard_size_mb} MB")
    print(f"Output directory: {args.output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(
        args.dataset, split=args.split, streaming=True, trust_remote_code=True
    )

    # Shuffle if requested
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    # Determine number of processes
    num_proc = args.num_proc
    if num_proc <= 0:
        # Use 80% of available CPUs
        num_proc = max(1, int(os.cpu_count() * 0.8))
    print(f"Using {num_proc} processes")

    # Initialize shard writer
    writer = ShardWriter(args.output_dir, args.shard_size_mb)

    # Process dataset with multiprocessing
    with mp.Pool(num_proc, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        with tqdm(total=args.max_tokens, unit="tokens", desc="Tokenizing") as pbar:
            for tokens, doc_id in pool.imap(
                tokenize_doc, iter(dataset), chunksize=args.chunk_size
            ):
                if writer.global_token_count >= args.max_tokens:
                    break

                if len(tokens) == 0:
                    continue

                # Truncate if exceeding max_tokens
                remaining = args.max_tokens - writer.global_token_count
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]

                writer.add_document(tokens, doc_id)
                pbar.update(len(tokens))

    # Finalize and write metadata
    writer.finalize()

    # Example: How to read back the data
    if args.verify:
        print("\nVerifying first shard...")
        first_shard = Path(args.output_dir) / "shard_000000.bin"
        first_index = Path(args.output_dir) / "shard_000000.idx"

        if first_shard.exists() and first_index.exists():
            tokens, index = load_shard(first_shard, first_index)
            print(f"First shard has {len(tokens)} tokens")
            print(f"First shard contains {len(index['documents'])} documents")

            # Get first document
            if index["documents"]:
                first_doc = get_document_from_shard(tokens, index, 0)
                print(f"First document has {len(first_doc)} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize dataset into sharded binary files with indexes"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        default="mlfoundations/dclm-baseline-1.0-parquet",
        help="HuggingFace dataset name",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument(
        "--tokenizer", default="gpt2", help="Transformers tokenizer name or path"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        default="./tokenized_data",
        help="Output directory for shards and indexes",
    )
    parser.add_argument(
        "--shard_size_mb",
        type=int,
        default=100,
        help="Size of each shard in MB (default: 100MB)",
    )

    # Processing arguments
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1_000_000_000,  # 1B tokens
        help="Maximum total tokens to process (default: 1B)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=0,
        help="Number of processes (0 for 80%% of CPUs)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Processing chunk size for multiprocessing",
    )

    # Dataset shuffling
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle the dataset before processing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=10000, help="Buffer size for shuffling"
    )

    # Verification
    parser.add_argument(
        "--verify", action="store_true", help="Verify the first shard after processing"
    )

    args = parser.parse_args()
    main(args)
