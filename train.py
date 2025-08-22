import tiktoken
from src.modeling.transformer import Transformer
from src.data.dist_dataloader import distributed_data_generator
from src.data.beacon_injecting import inject_beacon_to_docs
import torch
import wandb
import torch.distributed as dist
import os
from loguru import logger

# Distributed Init

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
dist.init_process_group(
    backend="nccl", init_method="env://", world_size=WORLD_SIZE, rank=RANK
)
IS_MASTER = RANK == 0


def log_master(message):
    if IS_MASTER:
        logger.info(message)


TRAIN_DATA_PATTERN = "data/fineweb10B/fineweb_train_%06d.bin"
VAL_DATA_PATTERN = "data/fineweb10B/fineweb_val_%06d.bin"
BATCH_SIZE = 48 * 1024


TOKENIZER = tiktoken.get_encoding("gpt2")

TOKENIZER._special_tokens = {}

train_loader = distributed_data_generator(
    TRAIN_DATA_PATTERN, batch_size=BATCH_SIZE * WORLD_SIZE, align_to_bos=True
)
