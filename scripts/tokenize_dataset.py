#!/usr/bin/env python3
import argparse
import os
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ----------------------------
# Globals (initialized in workers)
# ----------------------------
_tokenizer = None
_dtype = None  # np.dtype
_dtype_name = None  # str, e.g. "uint32"


# ----------------------------
# Worker init / tokenization
# ----------------------------
def init_worker(tokenizer_name: str, dtype_name: str, trust_remote_code: bool):
    """Initialize tokenizer and dtype in each worker process."""
    global _tokenizer, _dtype, _dtype_name
    _tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    _dtype_name = dtype_name
    _dtype = np.dtype(dtype_name)
    if _tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} must have an EOS token.")
    # Leave tokenizer.model_max_length alone; we explicitly avoid truncation.


def _pick_text(doc: Dict[str, Any], text_field: Optional[str]) -> str:
    """Find text in the doc."""
    if text_field:
        return doc.get(text_field, "")
    # Fall back through common field names
    for key in ("text", "content", "raw", "document", "body"):
        if key in doc and isinstance(doc[key], str):
            return doc[key]
    return ""


def tokenize_doc(
    args: Tuple[Dict[str, Any], Optional[str], Optional[str]],
) -> Tuple[np.ndarray, int]:
    """
    Tokenize a single document and append EOS token.

    Returns:
        (token_array (np.ndarray of selected dtype), doc_id (int or -1 if N/A))
    """
    global _tokenizer, _dtype
    if _tokenizer is None or _dtype is None:
        raise RuntimeError(
            "Worker not initialized correctly (tokenizer/dtype missing)."
        )

    doc, text_field, id_field = args
    text = _pick_text(doc, text_field)
    doc_id = doc.get(id_field, -1) if id_field else doc.get("id", -1)

    if not isinstance(text, str) or not text:
        return np.array([], dtype=_dtype), doc_id

    # Encode without truncation; rely on tokenizer to emit full sequence
    tokens: List[int] = _tokenizer.encode(text, add_special_tokens=False)
    tokens.append(int(_tokenizer.eos_token_id))

    tokens_array = np.asarray(tokens, dtype=_dtype)

    # Optional sanity check for overflow (mostly useful if using uint16)
    if tokens_array.dtype == np.uint16 and np.any(
        tokens_array > np.iinfo(np.uint16).max
    ):
        raise ValueError(
            "Token IDs exceed uint16 range; use --dtype uint32 or --dtype auto."
        )
    return tokens_array, int(doc_id) if isinstance(doc_id, int) else -1


# ----------------------------
# Shard Writer
# ----------------------------
class ShardWriter:
    """Handles writing shards and their index files."""

    def __init__(self, output_dir: str, shard_size_mb: int, dtype: np.dtype):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dtype = dtype
        self.bytes_per_token = int(np.dtype(dtype).itemsize)
        # Convert MB to number of tokens of this dtype
        self.shard_size_tokens = max(
            1, (shard_size_mb * 1024 * 1024) // self.bytes_per_token
        )

        self.current_shard_idx = 0
        self.current_buffer: List[np.ndarray] = []
        self.current_buffer_size = 0  # in tokens
        self.current_index: List[Dict[str, Optional[int]]] = []
        self.global_token_count = 0

    def add_document(self, tokens: np.ndarray, doc_id: Optional[int] = None) -> None:
        """Add a tokenized document to the current shard."""
        if tokens.size == 0:
            return
        if tokens.dtype != self.dtype:
            tokens = tokens.astype(self.dtype, copy=False)

        # Pre-flush to keep shard size closer to target when possible (no splitting docs)
        if (
            self.current_buffer_size > 0
            and (self.current_buffer_size + tokens.size) > self.shard_size_tokens
        ):
            self._write_shard()

        # Record index entry: positions are within this shard
        start_pos = self.current_buffer_size
        end_pos = start_pos + tokens.size

        self.current_index.append(
            {
                "start": start_pos,
                "end": end_pos,
                "length": int(tokens.size),
                "doc_id": int(doc_id)
                if (doc_id is not None and doc_id != -1)
                else None,
            }
        )

        self.current_buffer.append(tokens)
        self.current_buffer_size += tokens.size
        self.global_token_count += tokens.size

        if self.current_buffer_size >= self.shard_size_tokens:
            self._write_shard()

    def _write_shard(self) -> None:
        """Write current buffer as a shard with its index."""
        if not self.current_buffer:
            return

        shard_tokens = np.concatenate(self.current_buffer).astype(
            self.dtype, copy=False
        )

        shard_path = self.output_dir / f"shard_{self.current_shard_idx:06d}.bin"
        shard_tokens.tofile(shard_path)

        index_path = self.output_dir / f"shard_{self.current_shard_idx:06d}.idx"
        index_data = {
            "shard_id": self.current_shard_idx,
            "total_tokens": int(self.current_buffer_size),
            "num_documents": len(self.current_index),
            "dtype": str(self.dtype),
            "bytes_per_token": self.bytes_per_token,
            "documents": self.current_index,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        tqdm.write(
            f"Wrote shard {self.current_shard_idx}: "
            f"{self.current_buffer_size:,} tokens, "
            f"{len(self.current_index)} documents, "
            f"{(self.current_buffer_size * self.bytes_per_token) / (1024*1024):.1f} MB"
        )

        # Reset for next shard
        self.current_shard_idx += 1
        self.current_buffer = []
        self.current_buffer_size = 0
        self.current_index = []

    def finalize(self) -> None:
        """Write any remaining data and create metadata file."""
        if self.current_buffer:
            self._write_shard()

        metadata = {
            "total_shards": self.current_shard_idx,
            "total_tokens": int(self.global_token_count),
            "target_shard_size_mb": (self.shard_size_tokens * self.bytes_per_token)
            // (1024 * 1024),
            "dtype": str(self.dtype),
            "bytes_per_token": self.bytes_per_token,
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nTokenization complete!")
        print(f"Total shards: {self.current_shard_idx}")
        print(f"Total tokens: {self.global_token_count:,}")
        print(f"Output directory: {self.output_dir}")


# ----------------------------
# Read utilities
# ----------------------------
def load_shard(
    shard_path: Path, index_path: Path, dtype: np.dtype
) -> Tuple[np.ndarray, dict]:
    """Load a shard and its index."""
    tokens = np.fromfile(shard_path, dtype=dtype)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    return tokens, index


def get_document_from_shard(
    tokens: np.ndarray, index: dict, doc_idx: int
) -> np.ndarray:
    """Extract a specific document from a shard."""
    docs = index.get("documents", [])
    if doc_idx < 0 or doc_idx >= len(docs):
        raise IndexError(f"Document index {doc_idx} out of range")
    info = docs[doc_idx]
    return tokens[int(info["start"]) : int(info["end"])]


# ----------------------------
# Helpers
# ----------------------------
def resolve_dtype_name(
    tokenizer_name: str, dtype_arg: str, trust_remote_code: bool
) -> str:
    """
    Decide dtype name: "uint32", "uint16", or "auto".
    - "auto": choose uint16 if vocab and eos fit; else uint32
    """
    if dtype_arg in ("uint16", "uint32"):
        return dtype_arg
    if dtype_arg != "auto":
        raise ValueError(f"Unsupported dtype: {dtype_arg}")

    tok = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    # Heuristic: if vocab size and eos_token_id within uint16 range, use uint16
    try:
        vmax = np.iinfo(np.uint16).max
        if (
            tok.vocab_size is not None
            and tok.vocab_size - 1 <= vmax
            and (tok.eos_token_id or 0) <= vmax
        ):
            return "uint16"
    except Exception:
        pass
    return "uint32"


# ----------------------------
# Main
# ----------------------------
def main(args):
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"DType: {args.dtype}")
    print(f"Max tokens: {args.max_tokens:,}")
    print(f"Shard size: {args.shard_size_mb} MB")
    print(f"Output directory: {args.output_dir}")
    if args.text_field:
        print(f"Text field: {args.text_field}")
    if args.id_field:
        print(f"ID field: {args.id_field}")
    print(f"Trust remote code (tokenizer): {args.trust_remote_code}")

    # Decide dtype name (parent process, once) so we can pass it to workers
    dtype_name = resolve_dtype_name(args.tokenizer, args.dtype, args.trust_remote_code)
    dtype = np.dtype(dtype_name)
    print(f"Resolved dtype: {dtype_name} ({dtype.itemsize * 8} bits)")

    # Load dataset (streaming)
    print("\nLoading dataset (streaming)...")
    dataset = load_dataset(
        args.dataset, split=args.split, streaming=True, trust_remote_code=True
    )

    # Shuffle if requested (buffered reservoir shuffle for streaming datasets)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    # Determine number of processes
    num_proc = args.num_proc
    if num_proc <= 0:
        num_proc = max(1, int(os.cpu_count() * 0.8))
    print(f"Using {num_proc} processes")

    # Initialize shard writer
    writer = ShardWriter(args.output_dir, args.shard_size_mb, dtype=dtype)

    # Process dataset with multiprocessing
    # Note: we iterate items in the parent and feed them to workers; no dataset access inside workers.
    with mp.Pool(
        processes=num_proc,
        initializer=init_worker,
        initargs=(args.tokenizer, dtype_name, args.trust_remote_code),
    ) as pool:
        # Prepare an argument iterator that includes field preferences
        def arg_iter():
            for ex in dataset:
                yield (ex, args.text_field, args.id_field)

        processed_tokens = 0
        with tqdm(total=args.max_tokens, unit="tokens", desc="Tokenizing") as pbar:
            try:
                for tokens, doc_id in pool.imap(
                    tokenize_doc, arg_iter(), chunksize=args.chunk_size
                ):
                    if writer.global_token_count >= args.max_tokens:
                        break
                    if tokens.size == 0:
                        continue

                    # Truncate last doc if it would exceed max_tokens
                    remaining = args.max_tokens - writer.global_token_count
                    if tokens.size > remaining:
                        tokens = tokens[:remaining]

                    writer.add_document(tokens, doc_id)
                    processed_tokens += int(tokens.size)
                    pbar.update(int(tokens.size))
            finally:
                # Pool will be terminated by context manager if we exit early
                pass

    # Finalize and write metadata
    writer.finalize()

    # Optional verification
    if args.verify:
        print("\nVerifying first shard...")
        first_shard = Path(args.output_dir) / "shard_000000.bin"
        first_index = Path(args.output_dir) / "shard_000000.idx"

        if first_shard.exists() and first_index.exists():
            tokens, index = load_shard(first_shard, first_index, dtype=dtype)
            print(f"First shard has {len(tokens)} tokens (dtype={tokens.dtype})")
            print(f"First shard contains {len(index.get('documents', []))} documents")

            if index.get("documents"):
                first_doc = get_document_from_shard(tokens, index, 0)
                print(f"First document has {len(first_doc)} tokens")
        else:
            print("No shard_000000.* found to verify.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize a dataset into sharded binary files with per-shard indexes."
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/finepdfs",
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (e.g., train)"
    )
    parser.add_argument(
        "--tokenizer", default="gpt2", help="Transformers tokenizer name or path"
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Trust remote code for tokenizer",
    )
    parser.add_argument(
        "--text-field",
        default=None,
        help="Explicit text field name in dataset (optional)",
    )
    parser.add_argument(
        "--id-field",
        default=None,
        help="Explicit doc id field name in dataset (optional)",
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
        help="Target size of each shard in MB (default: 100)",
    )

    # Processing arguments
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1_000_000_000,
        help="Maximum total tokens to process (default: 1B)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=0,
        help="Number of processes (0 for ~80%% of CPUs)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Multiprocessing chunk size (docs/task)",
    )

    # Dataset shuffling
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before processing (streaming-safe)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10_000,
        help="Buffer size for streaming shuffle",
    )

    # Storage dtype
    parser.add_argument(
        "--dtype",
        choices=("uint16", "uint32", "auto"),
        default="uint32",
        help="Storage dtype for tokens. 'auto' uses uint16 if safe, else uint32.",
    )

    # Verification
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After writing, verify first shard & show basic stats",
    )

    args = parser.parse_args()
    main(args)
