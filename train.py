import os
import math
import time
import argparse
import yaml
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.attention.flex_attention import create_block_mask
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from modeling_beacon_gpt import BeaconGPT
import torch.distributed as dist
import glob
from pathlib import Path


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:  # single-GPU
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML"""

    # Model config
    vocab_size: int = 50257  # GPT-2 vocab size
    hidden_size: int = 768
    n_layer: int = 12
    n_head: int = 12
    max_seq_len: int = 128
    dropout: float = 0.1

    # Training config
    batch_size: int = 1  # Micro batch size (flex attention limitation)
    gradient_accumulation_steps: int = (
        8  # Effective batch size = batch_size * grad_accum
    )
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 10000

    # Logging and evaluation
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    sample_interval: int = 500

    # Data config
    dataset_bin_pattern: str = "./data/*train*.bin"

    # System config
    device: str = "auto"  # "auto", "cuda", "cpu"
    compile_model: bool = True
    mixed_precision: bool = True

    # Paths
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Filter out any keys that aren't in the dataclass
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_config)

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2)

    def resolve_device(self) -> str:
        """Resolve auto device to actual device"""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


class DataLoader:
    """Streaming data loader for training"""

    def __init__(self, config: TrainingConfig, tokenizer, local_rank, world_size):
        self.config = config
        self.tokenizer = tokenizer
        self.eot_token = tokenizer._special_tokens["<|endoftext|>"]
        self.local_rank = local_rank
        self.world_size = world_size
        # Load streaming dataset
        self.bin_files = [Path(file) for file in glob.glob(config.dataset_bin_pattern)]
        self.bin_files_iter = iter(self.bin_files)
        self.max_seq_len = config.max_seq_len * config.batch_size
        self.dataset_iter = self._gen_valid_sentence_ids()

    def _load_data_shard(self,file: Path):
        header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        num_tokens = int(header[2]) # number of tokens (claimed)
        with file.open("rb", buffering=0) as f:
            tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
            f.seek(256 * 4)
            nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
            assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
        return tokens

    def _gen_valid_sentence_ids(self) -> torch.Tensor:
        for bin_file in self.bin_files_iter:
            tokens = self._load_data_shard(bin_file)
            tokens_per_rank = tokens.shape[0] // self.world_size
            rank_start = tokens_per_rank * self.local_rank
            rank_end = rank_start + tokens_per_rank
            valid_sentence_ids = tokens[rank_start:rank_end]
            
            index = 0
            while index < valid_sentence_ids.shape[0]:
                # Look for EOT token in current window
                segment = valid_sentence_ids[index:index+self.max_seq_len]
                eot_token_idx = (segment == self.eot_token).nonzero(as_tuple=True)[0]
                
                if len(eot_token_idx) == 0:
                    # No EOT token found in this segment, skip ahead
                    index += self.max_seq_len
                    continue
                
                # Start from the first EOT token found
                eot_start = index + eot_token_idx[0].item()
                eot_end = min(eot_start + self.max_seq_len, valid_sentence_ids.shape[0])
                
                segment = valid_sentence_ids[eot_start:eot_end]
                
                # Only yield if we have a full sequence or this is the last possible segment
                if segment.shape[0] == self.max_seq_len or eot_end == valid_sentence_ids.shape[0]:
                    yield segment
                
                index = eot_start + self.max_seq_len

    def get_batch(self) -> torch.Tensor:
        """Get a batch of token IDs"""
        t0 = time.time()
        ids = next(self.dataset_iter)
        labels = ids[1:].clone()
        ids = ids[:-1].clone()
        device = self.config.resolve_device()
        ids = ids.to(dtype=torch.int32, device=device, non_blocking=True)
        labels = labels.to(dtype=torch.int64, device=device, non_blocking=True)
        t1 = time.time()
        return ids, labels, t1 - t0


class Trainer:
    """Main trainer class"""

    def __init__(self, config: TrainingConfig):
        setup_distributed()
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.local_rank == 0:
            wandb.init(project="beacon-gpt")

        print(f"local_rank: {self.local_rank}, world_size: {self.world_size}")
        self.config = config
        self.step = 0

        # Resolve device
        self.device = config.resolve_device()
        print(f"Using device: {self.device}")

        # Setup directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Save config to output directory
        config_save_path = Path(config.output_dir) / "config.yaml"
        config.save_yaml(config_save_path)
        print(f"Saved config to {config_save_path}")

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        print(f"Tokenizer vocab size: {self.tokenizer.n_vocab}")

        # Initialize model
        self.model = BeaconGPT(
            vocab_size=self.tokenizer.n_vocab,
            hidden_size=config.hidden_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            max_seq_len=config.max_seq_len * config.batch_size,
            dropout=config.dropout,
        )

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Move to device
        self.model.to(self.device)
        for param in self.model.parameters():
            dist.broadcast(param.detach(), 0)
        self.model = DDP(self.model, device_ids=[self.local_rank])

        # Compile model if requested
        if config.compile_model:
            print("Compiling model...")
            self.model = torch.compile(self.model)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._get_scheduler()

        # Initialize data loader
        self.data_loader = DataLoader(config, self.tokenizer, self.local_rank, self.world_size)

        # Initialize logging
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Mixed precision scaler
        self.scaler = (
            torch.amp.GradScaler()
            if config.mixed_precision and self.device == "cuda"
            else None
        )

        # Test prompt for generation
        self.test_prompt = "Hello, my name is"
        self.test_prompt_ids = torch.tensor(
            [self.tokenizer.encode(self.test_prompt)],
            dtype=torch.long,
            device=self.device,
        )

    def _get_scheduler(self):
        """Get learning rate scheduler with warmup"""

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                # Cosine decay after warmup
                progress = (step - self.config.warmup_steps) / (
                    self.config.max_steps - self.config.warmup_steps
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _create_document_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask that prevents attention across document boundaries"""
        eot_token = self.tokenizer._special_tokens["<|endoftext|>"]

        # Find document boundaries
        docs = (input_ids == eot_token)[0].cumsum(dim=0)

        def mask_mod(b, h, q_idx, kv_idx):
            is_causal = q_idx >= kv_idx
            is_same_doc = docs[q_idx] == docs[kv_idx]
            return is_causal & is_same_doc

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=input_ids.size(1),
            KV_LEN=input_ids.size(1),
            device=self.device,
        )

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        t0 = time.time()
        mask = self._create_document_mask(input_ids)
        t1 = time.time()
        mask_time = t1 - t0

        # Prepare labels (shifted input for next token prediction)
        # labels = input_ids.clone()
        t0 = time.time()
        if self.config.mixed_precision and self.scaler is not None:
            with torch.amp.autocast(device_type=self.device):
                logits, loss = self.model(input_ids, labels=labels, mask=mask)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            logits, loss = self.model(input_ids, labels=labels, mask=mask)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
        t1 = time.time()
        loss_time = t1 - t0
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "lr": self.scheduler.get_last_lr()[0],
            "mask_time": mask_time,
            "loss_time": loss_time,
        }

    def generate_sample(self) -> str:
        """Generate a sample from the model"""
        self.model.eval()
        with torch.no_grad():
            generated_tokens = self.model.module.generate(
                self.test_prompt_ids,
                do_sample=False,
            )
            generated_text = self.test_prompt + self.tokenizer.decode(generated_tokens)
        self.model.train()
        return generated_text

    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from {checkpoint_path}, step {self.step}")

    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Configuration:")
        print(f"  Max steps: {self.config.max_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(
            f"  Model: {self.config.hidden_size}d, {self.config.n_layer}L, {self.config.n_head}H"
        )
        print(f"  Max sequence length: {self.config.max_seq_len}")
        print(f"  Device: {self.device}")
        print()

        self.model.train()

        accumulated_loss = 0.0
        start_time = time.time()

        # Progress bar
        pbar = tqdm(total=self.config.max_steps, initial=self.step, desc="Training")

        total_tokens = 0

        while self.step < self.config.max_steps:
            # Get batch
            input_ids, labels, token_time = self.data_loader.get_batch()
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)
            total_tokens += input_ids.shape[1]

            # Training step
            metrics = self.train_step(input_ids, labels)
            accumulated_loss += metrics["loss"]

            # Update weights every gradient_accumulation_steps
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    # Gradient clipping with mixed precision
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

            # Logging
            if self.step % self.config.log_interval == 0 and self.local_rank == 0:
                avg_loss = accumulated_loss / self.config.log_interval
                elapsed_time = time.time() - start_time
                tokens_per_sec = (
                    self.config.log_interval * self.config.max_seq_len
                ) / elapsed_time

                print(
                    f"Step {self.step}: loss={avg_loss:.4f}, lr={metrics['lr']:.2e}, "
                    f"tok/s={tokens_per_sec:.0f}, mask_time={metrics['mask_time']:.4f}, loss_time={metrics['loss_time']:.4f}, input_shape={input_ids.shape}, total_tokens={total_tokens}, token_time={token_time:.4f}"
                )
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": metrics["lr"],
                    "train/tokens_per_second": tokens_per_sec,
                    "train/mask_time": metrics["mask_time"],
                    "train/read_token_time": token_time,
                    "train/loss_time": metrics["loss_time"],
                })

                accumulated_loss = 0.0
                start_time = time.time()

            # Generate samples
            if self.step % self.config.sample_interval == 0 and self.step > 0 and self.local_rank == 0:
                sample_text = self.generate_sample()
                print(f"Sample: {sample_text}")
                self.writer.add_text("samples/generated", sample_text, self.step)
                wandb.log({
                    "samples/generated": sample_text,
                })

            # Save checkpoint
            if self.step % self.config.save_interval == 0 and self.step > 0 and self.local_rank == 0:
                self.save_checkpoint(self.step)

            self.step += 1
            pbar.update(1)

        pbar.close()
        print("Training completed!")

        # Final checkpoint
        self.save_checkpoint(self.step)
        self.writer.close()


def create_default_config(output_path: str):
    """Create a default configuration file"""
    config = TrainingConfig()
    config.save_yaml(output_path)
    print(f"Created default configuration at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train BeaconGPT model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--create-config",
        type=str,
        help="Create a default config file at specified path and exit",
    )

    args = parser.parse_args()

    # Create default config if requested
    if args.create_config:
        create_default_config(args.create_config)
        return

    # Load config from YAML file
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found!")
        print(
            f"Create a default config with: python {__file__} --create-config {args.config}"
        )
        return

    config = TrainingConfig.from_yaml(args.config)
    print(f"Loaded configuration from {args.config}")

    # Create trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
