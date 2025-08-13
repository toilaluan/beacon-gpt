import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    create_block_mask,
)
from typing import Callable


def norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def create_beacon_mask(max_seq_len: int, beacon_offset: int) -> Callable:
    def create_mask(b, h, q_idx, kv_idx):
        is_causal = q_idx <= kv_idx
        is_beacon = kv_idx % beacon_offset == 0
        is_recent = kv_idx >= q_idx - beacon_offset
        return is_causal & (is_beacon | is_recent)

    return create_mask


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000, max_seq_len: int = 2048):
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

    def forward(self, x: torch.Tensor, seq_index: int = None) -> torch.Tensor:
        if seq_index is None:
            seq_len = x.size(2)
            assert (
                seq_len <= self.max_seq_len
            ), "input sequence length exceeds max_seq_len"
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


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, hidden_size: int, max_seq_len: int):
        super().__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        assert hidden_size % n_head == 0, "hidden_size must be divisible by n_head"
        self.head_dim = hidden_size // n_head
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim, max_seq_len=max_seq_len
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

    def forward(self, x: torch.Tensor, block_mask: BlockMask) -> torch.Tensor:
        B, L, _ = x.shape
        q, k, v = self.to_qkv(x).split(self.hidden_size, dim=2)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        q = norm(q)
        k = norm(k)
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).reshape(B, L, self.hidden_size)
        return self.out_proj(out), k, v


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.GELU()
        self.init_weights()

    def init_weights(self):
        # modded nanogpt init
        self.w1.weight.detach().zero_()
        self.w2.weight.detach().zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class Block(nn.Module):
    def __init__(self, n_head: int, hidden_size: int, max_seq_len: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, hidden_size, max_seq_len)
        self.mlp = MLP(hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor, block_mask: BlockMask) -> torch.Tensor:
        x_attn, k, v = self.attn(norm(x), block_mask)
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
        dropout: float = 0.1,
        use_beacon: bool = False,
        beacon_offset: int = 16,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_head, hidden_size, max_seq_len) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.use_beacon = use_beacon
        self.beacon_offset = beacon_offset
        if self.use_beacon:
            self.mask = create_block_mask(
                create_beacon_mask(max_seq_len, beacon_offset),
                B=None,
                H=None,
                Q_LEN=max_seq_len,
                KV_LEN=max_seq_len,
                device="cpu",
            )
        else:
            self.mask = create_block_mask(
                lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
                B=None,
                H=None,
                Q_LEN=max_seq_len,
                KV_LEN=max_seq_len,
                device="cpu",
            )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                std = 0.5 * (in_features**-0.5)
                bound = 3**0.5 * std
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        assert input_ids.size(0) == 1, "Only batch size 1 is supported with flex-attn"
        x = self.wte(input_ids)
        x = norm(x)
        for block in self.blocks:
            x = x + block(x, self.mask)[0]
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            logits = logits.float()
            labels = F.pad(labels, (0, 1), value=-100)
            labels = labels[:, 1:]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return logits, loss

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 16
    ) -> torch.Tensor:
        assert input_ids.size(0) == 1, "Only batch size 1 is supported for generate"
        promise_kv_len = input_ids.size(1) + max_new_tokens
        c_k = [
            torch.empty(1, self.n_head, promise_kv_len, self.head_dim, device="cpu")
            for _ in range(self.n_layer)
        ]
        c_v = [
            torch.empty(1, self.n_head, promise_kv_len, self.head_dim, device="cpu")
            for _ in range(self.n_layer)
        ]
        print("pre-allocated kv cache", c_k[0].shape, c_v[0].shape)
        generated_ids = []
        x = self.wte(input_ids)
        x = norm(x)
        mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=None,
            H=None,
            Q_LEN=input_ids.size(1),
            KV_LEN=input_ids.size(1),
            device="cpu",
        )
        for i, block in enumerate(self.blocks):
            x, k, v = block(x, mask)
            c_k[i][:, :, : k.size(2), :] = k
            c_v[i][:, :, : v.size(2), :] = v

        seq_index = input_ids.size(1)
        mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx + seq_index >= kv_idx,
            B=None,
            H=None,
            Q_LEN=1,
            KV_LEN=promise_kv_len,
            device="cpu",
        )
        logits = self.lm_head(x)
        new_token_id = torch.argmax(logits[0, 0, :], dim=-1, keepdim=True)
        generated_ids.append(new_token_id.item())
        for _ in range(max_new_tokens - 1):
            x = self.wte(new_token_id.unsqueeze(1))
            x = norm(x)
            for i, block in enumerate(self.blocks):
                B, L, _ = x.shape
                q, k, v = block.attn.to_qkv(x).split(block.attn.hidden_size, dim=2)
                q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
                v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
                q = norm(q)
                k = norm(k)
                q = block.attn.rotary_emb(q, seq_index=seq_index)
                k = block.attn.rotary_emb(k, seq_index=seq_index)
                c_k[i][:, :, seq_index, :] = k.squeeze(2)
                c_v[i][:, :, seq_index, :] = v.squeeze(2)
                out = flex_attention(q, c_k[i], c_v[i], block_mask=mask)
                out = out.transpose(1, 2).reshape(B, L, block.attn.hidden_size)
                x = x + out
            seq_index += 1
            x = x + block.mlp(norm(x))
            logits = self.lm_head(x)
            new_token_id = torch.argmax(logits[0, 0, :], dim=-1, keepdim=True)
            generated_ids.append(new_token_id.item())
        return generated_ids


if __name__ == "__main__":
    model = BeaconGPT(
        vocab_size=512,
        hidden_size=128,
        n_layer=4,
        n_head=4,
        max_seq_len=128,
    )
    print(model)

    x = torch.randint(0, 512, (1, 64))
    print(x)
    out = model.generate(x, max_new_tokens=16)
    print(out)
