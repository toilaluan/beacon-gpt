import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import flex_attention
from typing import Tuple, Optional

class CastedLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.type_as(x))

class KVCache:
    def __init__(
        self,
        n_layers: int,
        head_dim: int,
        n_heads: int,
        max_seq_len: int,
        device: str = "cpu",
        beacon_stride: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.beacon_stride = beacon_stride
        self.dtype = dtype

        self.keys = [
            torch.empty(1, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.values = [
            torch.empty(1, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.not_beacon_counter = 0
        self.current_length = 0

    def need_new_beacon(self) -> bool:
        return self.beacon_stride == self.not_beacon_counter

    def get_new_length_after_reducing(self):
        return self.current_length + 1 - self.beacon_stride

    def get_current_beacon_count(self):
        return self.current_length - self.not_beacon_counter

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        prefill: bool = False,
        not_beacon_counter: int = 0,
        is_beacon: bool = False,
    ):
        _prev = (
            self.current_length,
            self.get_current_beacon_count(),
            self.not_beacon_counter,
        )
        if prefill:
            seq_len = k.size(2)
            self.keys[layer_idx][:, :, :seq_len, :] = k.to(dtype=self.dtype)
            self.values[layer_idx][:, :, :seq_len, :] = v.to(dtype=self.dtype)
            self.not_beacon_counter = not_beacon_counter
            self.current_length = seq_len
        elif self.need_new_beacon() and is_beacon:
            new_beacon_index = self.get_new_length_after_reducing()
            self.keys[layer_idx][:, :, new_beacon_index, :] = k.squeeze(2).to(
                dtype=self.dtype
            )
            self.values[layer_idx][:, :, new_beacon_index, :] = v.squeeze(2).to(
                dtype=self.dtype
            )
            if layer_idx == 0:
                self.not_beacon_counter = 0
                self.current_length = new_beacon_index
        else:
            self.keys[layer_idx][:, :, self.current_length, :] = k.squeeze(2).to(
                dtype=self.dtype
            )
            self.values[layer_idx][:, :, self.current_length, :] = v.squeeze(2).to(
                dtype=self.dtype
            )
            if layer_idx == 0:
                self.current_length += 1
                if is_beacon:
                    self.not_beacon_counter += 1
        
        _after = (
            self.current_length,
            self.get_current_beacon_count(),
            self.not_beacon_counter,
        )
        # if layer_idx == 0:
        #     print(
        #         f"KV Cache (current_length, beacon_count, not_beacon_counter): {_prev} -> {_after}"
        #     )

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.keys[layer_idx][:, :, : self.current_length, :],
            self.values[layer_idx][:, :, : self.current_length, :],
        )


def get_causal_document_mask_mod(input_ids: torch.Tensor, bos_id: int):
    assert input_ids.ndim == 1, "input_ids must be a 1D tensor"
    doc_identifiers = (input_ids == bos_id).cumsum(0)

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_identifiers[q_idx] == doc_identifiers[kv_idx]
        return causal_mask & document_mask

    return mask_mod


def reduce_beacon_ids_decoding(input_ids: torch.Tensor, beacon_id: int):
    is_beacons = input_ids == beacon_id
    last_beacon_index = is_beacons.nonzero(as_tuple=True)[0][-1]
    n_beacons = is_beacons.sum()
    return torch.cat(
        [
            torch.tensor([beacon_id] * n_beacons, device=input_ids.device),
            input_ids[last_beacon_index + 1 :],
        ]
    )


def get_beacon_causal_document_mask_mod(
    input_ids: torch.Tensor, bos_id: int, beacon_id: int
):
    assert input_ids.ndim == 1, "input_ids must be a 1D tensor"
    doc_identifiers = (input_ids == bos_id).cumsum(0)
    is_beacons = input_ids == beacon_id
    beacon_identifiers = is_beacons.long().cumsum(0) - is_beacons.long()

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        is_beacon = is_beacons[kv_idx]
        is_same_beacon_part = beacon_identifiers[q_idx] == beacon_identifiers[kv_idx]
        document_mask = doc_identifiers[q_idx] == doc_identifiers[kv_idx]
        return causal_mask & document_mask & (is_beacon | is_same_beacon_part)

    return mask_mod


def get_block_mask_for_decoding(kv_cache: KVCache):
    offset = kv_cache.current_length
    return lambda b, h, q_idx, kv_idx: q_idx + offset >= kv_idx


def create_block_mask(
    input_ids: torch.Tensor,
    bos_id: int,
    beacon_id: int = None,
    mask_type: str = "causal_document",
    decoding: bool = False,
):
    if mask_type == "causal_document":
        mask_mod = get_causal_document_mask_mod(input_ids, bos_id)
    elif mask_type == "beacon_causal_document":
        mask_mod = get_beacon_causal_document_mask_mod(input_ids, bos_id, beacon_id)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")

    return flex_attention.create_block_mask(
        mask_mod=mask_mod,
        B=None,
        H=None,
        Q_LEN=input_ids.size(0) if not decoding else 1,
        KV_LEN=input_ids.size(0),
        device=input_ids.device,
    )


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


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
        angles = torch.outer(torch.arange(self.max_seq_len), inv_freq)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x: torch.Tensor, position: Optional[int] = None) -> torch.Tensor:
        """
        x: torch.Tensor [1, H, L, D]
        position: int, optional, use for generation
        """
        if position is None:
            seq_len = x.size(2)
            cos = self.cos[:seq_len, :]
            sin = self.sin[:seq_len, :]
        else:
            assert x.size(2) == 1, "x must have sequence length 1 for generation"
            cos = self.cos[position, :].unsqueeze(0)
            sin = self.sin[position, :].unsqueeze(0)

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
            head_dim=self.head_dim, max_seq_len=max_seq_len, base=10000
        )
        self.to_qkv = CastedLinear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, CastedLinear):
                in_features = module.in_features
                std = 0.5 * (in_features**-0.5)
                bound = 3**0.5 * std
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound)
        self.out_proj.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        block_mask: flex_attention.BlockMask,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        kv_cache_args: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        x: [1, L, D], prev layer output or input embedding
        block_mask: BlockMask, block mask for flex attention
        kv_cache: KV Cache manager for generation
        layer_idx: int, layer index for KV cache
        kv_cache_args: dict, arguments for KV cache
        """
        B, L, _ = x.shape
        q, k, v = self.to_qkv(x).split(self.hidden_size, dim=2)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        q = rms_norm(q)
        k = rms_norm(k)

        if kv_cache is not None:
            if kv_cache_args["prefill"]:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
                if kv_cache_args["beacon_mask"] is not None:
                    k = k[:, :, kv_cache_args["beacon_mask"], :]
                    v = v[:, :, kv_cache_args["beacon_mask"], :]
                kv_cache.update(
                    layer_idx,
                    k,
                    v,
                    prefill=True,
                    not_beacon_counter=kv_cache_args["not_beacon_counter"],
                    is_beacon=kv_cache_args["is_beacon"],
                )
                k, v = kv_cache.get_kv(layer_idx)
            else:
                q = self.rotary_emb(q, position=kv_cache.current_length)
                k = self.rotary_emb(k, position=kv_cache.current_length)
                kv_cache.update(
                    layer_idx,
                    k,
                    v,
                    prefill=False,
                )
                k, v = kv_cache.get_kv(layer_idx)
        else:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        out = flex_attention.flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).reshape(B, L, self.hidden_size)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = CastedLinear(hidden_size, intermediate_size, bias=False)
        self.w2 = CastedLinear(intermediate_size, hidden_size, bias=False)
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
        block_mask: flex_attention.BlockMask,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        kv_cache_args: Optional[dict] = None,
    ) -> torch.Tensor:
        x_attn = self.attn(
            rms_norm(x),
            block_mask,
            kv_cache,
            layer_idx,
            kv_cache_args,
        )
        x = x + x_attn
        x = x + self.mlp(rms_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_head: int,
        hidden_size: int,
        max_seq_len: int,
        n_layer: int,
        beacon_id: int,
        bos_id: int,
        beacon_stride: int = 0,
    ):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=16)
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_head
        self.max_seq_len = max_seq_len
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [Block(n_head, hidden_size, max_seq_len) for _ in range(n_layer)]
        )
        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)
        self.beacon_stride = beacon_stride
        self.beacon_id = beacon_id
        self.bos_id = bos_id

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[flex_attention.BlockMask] = None,
        kv_cache: Optional[KVCache] = None,
        kv_cache_args: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert (
            input_ids.ndim == 1
        ), f"input_ids must be a 1D tensor, but got {input_ids}"
        _kv_cache_args = {
            "prefill": False,
            "is_beacon": False,
        }
        if kv_cache_args is not None:
            _kv_cache_args.update(kv_cache_args)
        if (
            kv_cache is not None
            and _kv_cache_args.get("prefill")
            and _kv_cache_args.get("is_beacon")
        ):
            beacon_mask = input_ids == self.beacon_id
            last_beacon_index = beacon_mask.nonzero(as_tuple=True)[0][-1]
            not_beacon_counter = int(input_ids.size(0) - last_beacon_index - 1)
            beacon_mask[last_beacon_index:] = True
        else:
            beacon_mask = None
            not_beacon_counter = 0
        _kv_cache_args.update(
            {
                "beacon_mask": beacon_mask,
                "not_beacon_counter": not_beacon_counter,
            }
        )
        x = self.embed_tokens(input_ids.unsqueeze(0))
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                block_mask=mask,
                kv_cache=kv_cache,
                layer_idx=i,
                kv_cache_args=_kv_cache_args,
            )

        x = rms_norm(x)
        logits = self.lm_head(x).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        is_beacon: bool = False,
    ):
        assert input_ids.ndim == 1, "input_ids must be a 1D tensor"
        # pre-allocate KV cache
        kv_length = next_multiple_of_n(input_ids.size(0) + max_new_tokens, n=16)
        kv_cache = KVCache(
            n_layers=self.n_layer,
            head_dim=self.head_dim,
            n_heads=self.n_head,
            max_seq_len=kv_length,
            beacon_stride=self.beacon_stride,
            device=input_ids.device,
            dtype=self.embed_tokens.weight.dtype,
        )
        block_mask = create_block_mask(
            input_ids=input_ids,
            bos_id=self.bos_id,
            beacon_id=self.beacon_id,
            mask_type="beacon_causal_document" if is_beacon else "causal_document",
            decoding=False,
        )
        # Prefill phase
        kv_cache_args = {
            "prefill": True,
            "beacon_reduction": False,
        }
        logits, _ = self.forward(
            input_ids, mask=None, kv_cache=kv_cache, kv_cache_args=kv_cache_args
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=False)
        decoded_tokens = input_ids
        input_ids = next_token
        decoded_tokens = torch.cat([decoded_tokens, next_token], dim=0)
        decode_kv_cache_args = {
            "prefill": False,
            "beacon_reduction": False,
            "is_beacon": is_beacon,
        }

        # Decoding phase
        for i in range(max_new_tokens):
            block_mask = flex_attention.create_block_mask(
                mask_mod=get_block_mask_for_decoding(kv_cache),
                B=None,
                H=None,
                Q_LEN=1,
                KV_LEN=kv_cache.current_length + 1,
                device=input_ids.device,
            )
            logits, _ = self.forward(
                input_ids,
                mask=block_mask,
                kv_cache=kv_cache,
                kv_cache_args=decode_kv_cache_args,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=False)
            decoded_tokens = torch.cat([decoded_tokens, next_token], dim=0)
            input_ids = next_token
            if kv_cache.need_new_beacon():
                input_ids = torch.tensor([self.beacon_id], device=input_ids.device)
                block_mask = flex_attention.create_block_mask(
                    mask_mod=get_block_mask_for_decoding(kv_cache),
                    B=None,
                    H=None,
                    Q_LEN=1,
                    KV_LEN=kv_cache.get_new_length_after_reducing(),
                    device=input_ids.device,
                )
                logits, _ = self.forward(
                    input_ids,
                    mask=block_mask,
                    kv_cache=kv_cache,
                    kv_cache_args=decode_kv_cache_args,
                )
        return decoded_tokens


if __name__ == "__main__":
    import time

    model = Transformer(
        vocab_size=100,
        n_head=4,
        hidden_size=128,
        max_seq_len=1024,
        n_layer=1,
        beacon_id=24,
        bos_id=23,
        beacon_stride=4,
    )
    input_ids = torch.randint(0, 100, (10,))
    labels = torch.randint(0, 100, (10,))
    mask = create_block_mask(
        input_ids,
        bos_id=23,
        beacon_id=24,
        mask_type="beacon_causal_document",
    )
    logits, loss = model(input_ids, labels, mask)
    print(logits.shape)
    print(loss)

    # Generation
    print("Generation")
    input_ids = torch.tensor(
        [23, 1, 2, 2, 24, 2, 1, 3, 5, 24, 2, 2], device=input_ids.device
    )
    start = time.time()
    output = model.generate(input_ids, max_new_tokens=256, is_beacon=False)
    print(output)
    print(f"Causal time taken: {time.time() - start} seconds")

    start = time.time()
    output = model.generate(input_ids, max_new_tokens=256, is_beacon=True)
    print(output)
    print(f"Beacon time taken: {time.time() - start} seconds")

    # ! Beacon is faster
