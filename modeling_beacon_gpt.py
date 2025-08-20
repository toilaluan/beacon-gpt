import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    create_block_mask,
)
from typing import Callable, Optional, List, Tuple

# flex_attention = torch.compile(flex_attention)


def norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def create_beacon_mask(max_seq_len: int, beacon_offset: int) -> Callable:
    def create_mask(b, h, q_idx, kv_idx):
        is_causal = q_idx >= kv_idx
        is_beacon = kv_idx % beacon_offset == 0
        is_recent = kv_idx >= q_idx - beacon_offset
        return is_causal & (is_beacon | is_recent)

    return create_mask


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 1024, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        angles = torch.outer(torch.arange(max_seq_len), inv_freq)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x: torch.Tensor, seq_index: Optional[int] = None) -> torch.Tensor:
        if seq_index is None:
            seq_len = x.size(2)
            cos = self.cos[:seq_len, :]
            sin = self.sin[:seq_len, :]
        else:
            cos = self.cos[seq_index, :].unsqueeze(0)
            sin = self.sin[seq_index, :].unsqueeze(0)

        x_odd = x[:, :, :, 1::2].float()
        x_even = x[:, :, :, ::2].float()
        y_odd = x_odd * cos - x_even * sin
        y_even = x_odd * sin + x_even * cos
        y = torch.stack([y_odd, y_even], dim=-1)
        return y.flatten(start_dim=3).type_as(x)


class KVCache:
    """Convenient KV cache container for managing attention states"""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: str = "cpu",
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.current_length = 0

        # Pre-allocate cache tensors
        self.keys = [
            torch.empty(
                1, n_heads, max_seq_len, head_dim, device=device, dtype=torch.float16
            )
            for _ in range(n_layers)
        ]
        self.values = [
            torch.empty(
                1, n_heads, max_seq_len, head_dim, device=device, dtype=torch.float16
            )
            for _ in range(n_layers)
        ]

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        position: Optional[int] = None,
    ):
        """Update cache with new key-value pairs"""
        if position is None:
            # Batch update (for prefill)
            seq_len = k.size(2)
            self.keys[layer_idx][:, :, :seq_len, :] = k.to(dtype=torch.float16)
            self.values[layer_idx][:, :, :seq_len, :] = v.to(dtype=torch.float16)
            self.current_length = seq_len
        else:
            # Single position update (for generation)
            self.keys[layer_idx][:, :, position, :] = k.squeeze(2).to(
                dtype=torch.float16
            )
            self.values[layer_idx][:, :, position, :] = v.squeeze(2).to(
                dtype=torch.float16
            )
            if position >= self.current_length:
                self.current_length = position + 1

    def get_kv(
        self, layer_idx: int, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get key-value pairs for a specific layer"""
        if seq_len is None:
            seq_len = self.current_length
        return (
            self.keys[layer_idx][:, :, :seq_len, :].to(dtype=torch.float32),
            self.values[layer_idx][:, :, :seq_len, :].to(dtype=torch.float32),
        )

    def clear(self):
        """Clear the cache"""
        self.current_length = 0
        for layer_idx in range(self.n_layers):
            self.keys[layer_idx].zero_()
            self.values[layer_idx].zero_()


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, hidden_size: int, max_seq_len: int):
        super().__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        assert hidden_size % n_head == 0, "hidden_size must be divisible by n_head"
        self.head_dim = hidden_size // n_head
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim, max_seq_len=max_seq_len, base=10000
        )
        self.to_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                std = 0.5 * (in_features**-0.5)
                bound = 3**0.5 * std
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound)
        self.out_proj.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        block_mask: BlockMask,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False,
        position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, L, _ = x.shape
        q, k, v = self.to_qkv(x).split(self.hidden_size, dim=2)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        q = norm(q)
        k = norm(k)

        if position is not None:
            q = self.rotary_emb(q, seq_index=position)
            k = self.rotary_emb(k, seq_index=position)
        else:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        # Handle KV cache
        if use_cache and kv_cache is not None and layer_idx is not None:
            if position is not None:
                # Generation mode - use cached KV
                kv_cache.update(layer_idx, k, v, position)
                cached_k, cached_v = kv_cache.get_kv(layer_idx)
                out = flex_attention(q, cached_k, cached_v, block_mask=block_mask)
            else:
                # Prefill mode - cache new KV
                kv_cache.update(layer_idx, k, v)
                out = flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Standard forward without cache
            out = flex_attention(q, k, v, block_mask=block_mask)

        out = out.transpose(1, 2).reshape(B, L, self.hidden_size)
        return self.out_proj(out), k, v


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.init_weights()

    def init_weights(self):
        self.w2.weight.detach().zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)).square())


class Block(nn.Module):
    def __init__(self, n_head: int, hidden_size: int, max_seq_len: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, hidden_size, max_seq_len)
        self.mlp = MLP(hidden_size, 4 * hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        block_mask: BlockMask,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False,
        position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x_attn, k, v = self.attn(
            norm(x), block_mask, kv_cache, layer_idx, use_cache, position
        )
        x = x + x_attn
        x = x + self.mlp(norm(x))
        return x, k, v


class BeaconGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_layer: int,
        n_head: int,
        max_seq_len: int,
        use_beacon: bool = False,
        beacon_offset: int = 16,
    ):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=16)
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.n_layer = n_layer
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_head
        self.max_seq_len = max_seq_len
        self.blocks = nn.ModuleList(
            [Block(n_head, hidden_size, max_seq_len) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.use_beacon = use_beacon
        self.beacon_offset = beacon_offset

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                std = 0.5 * (in_features**-0.5)
                bound = 3**0.5 * std
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound)
        self.lm_head.weight.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[BlockMask] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert input_ids.size(0) == 1, "Only batch size 1 is supported with flex-attn"

        x = self.wte(input_ids)
        x = norm(x)

        # Create default causal mask if none provided
        if mask is None:
            seq_len = input_ids.size(1)
            mask = create_block_mask(
                lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len if position is None else kv_cache.current_length + 1,
                device=input_ids.device,
            )

        for i, block in enumerate(self.blocks):
            x, _, _ = block(x, mask, kv_cache, i, use_cache, position)

        x = norm(x)
        logits = self.lm_head(x).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

        return logits, loss

    def create_kv_cache(self, device: str = "cpu") -> KVCache:
        """Create a new KV cache instance"""
        return KVCache(
            n_layers=self.n_layer,
            n_heads=self.n_head,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            device=device,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
    ) -> List[int]:
        """Improved generation with proper KV caching"""
        assert input_ids.size(0) == 1, "Only batch size 1 is supported for generate"
        device = input_ids.device

        # Create KV cache
        kv_cache = self.create_kv_cache(device=str(device))

        # Prefill phase
        seq_len = input_ids.size(1)
        prefill_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

        logits, _ = self.forward(
            input_ids, mask=prefill_mask, kv_cache=kv_cache, use_cache=True
        )

        # Sample first token
        next_token = self._sample_token(
            logits[:, -1:, :], temperature, top_k, top_p, do_sample
        )
        generated_ids = [next_token.item()]

        # Generation phase
        for step in range(max_new_tokens - 1):
            position = kv_cache.current_length

            # Create mask for single token generation
            gen_mask = create_block_mask(
                lambda b, h, q_idx, kv_idx: q_idx + position >= kv_idx,
                B=None,
                H=None,
                Q_LEN=1,
                KV_LEN=position + 1,
                device=device,
            )
            logits, _ = self.forward(
                next_token,
                mask=gen_mask,
                kv_cache=kv_cache,
                use_cache=True,
                position=position,
            )

            next_token = self._sample_token(
                logits, temperature, top_k, top_p, do_sample
            )
            generated_ids.append(next_token.item())

        return generated_ids

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:
        """Sample next token from logits"""
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            # Top-k filtering
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if top_p is not None:
            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token


if __name__ == "__main__":
    # Example usage
    model = BeaconGPT(
        vocab_size=512,
        hidden_size=128,
        n_layer=4,
        n_head=4,
        max_seq_len=128,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test generation
    x = torch.randint(0, 512, (1, 16))
    print("Input:", x.shape)

    # Generate with sampling
    generated = model.generate(
        x, max_new_tokens=16, temperature=0.8, do_sample=True, top_k=50
    )
    print("Generated tokens:", generated)

    # Test KV cache directly
    kv_cache = model.create_kv_cache()
    print("KV cache created:", kv_cache.current_length)
