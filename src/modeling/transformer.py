import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.nn.attention import flex_attention

flex_attention.flex_attention = torch.compile(flex_attention.flex_attention)


class DTypeLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.type_as(x))


class ScaledRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(1, dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._norm(x.float())
        y = y * (1.0 + self.weight[0].float())
        return y.type_as(x)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


def repeat_kv_heads(x: torch.Tensor, repeats: int) -> torch.Tensor:
    bsz, n_kv, slen, hdim = x.shape
    if repeats == 1:
        return x
    x = x[:, :, None, :, :].expand(bsz, n_kv, repeats, slen, hdim)
    return x.reshape(bsz, n_kv * repeats, slen, hdim)


class KVCache:
    def __init__(
        self,
        num_hidden_layers: int,
        head_dim: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        device: str = "cpu",
        beacon_stride: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        use_beacon: bool = False,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.device = device
        self.beacon_stride = beacon_stride
        self.dtype = dtype
        self.use_beacon = use_beacon

        self.keys = [
            torch.empty(
                1,
                num_key_value_heads,
                max_position_embeddings,
                head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_hidden_layers)
        ]
        self.values = [
            torch.empty(
                1,
                num_key_value_heads,
                max_position_embeddings,
                head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_hidden_layers)
        ]
        self.not_beacon_counter = 0
        self.current_length = 0
        self.uncompressed_length = 0

    def need_new_beacon(self) -> bool:
        if not self.use_beacon or self.beacon_stride is None or self.beacon_stride <= 0:
            return False
        return self.not_beacon_counter >= self.beacon_stride

    def get_current_beacon_count(self) -> int:
        return (self.current_length - self.not_beacon_counter) if self.use_beacon else 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        prefill: bool = False,
        not_beacon_counter: int = 0,
        prefill_length: int = 0,
    ):
        before = (
            self.current_length,
            self.get_current_beacon_count(),
            self.not_beacon_counter,
            self.uncompressed_length,
        )
        if prefill:
            seqlen = k.size(2)
            self.keys[layer_idx][:, :, :seqlen, :] = k.to(dtype=self.dtype)
            self.values[layer_idx][:, :, :seqlen, :] = v.to(dtype=self.dtype)
            self.not_beacon_counter = not_beacon_counter
            self.current_length = seqlen
            self.uncompressed_length = prefill_length
        else:
            idx = self.current_length if layer_idx == 0 else self.current_length - 1
            self.keys[layer_idx][:, :, idx, :] = k.squeeze(2).to(dtype=self.dtype)
            self.values[layer_idx][:, :, idx, :] = v.squeeze(2).to(dtype=self.dtype)
            if layer_idx == 0:
                self.current_length += 1
                self.uncompressed_length += 1
                self.not_beacon_counter += 1

        after = (
            self.current_length,
            self.get_current_beacon_count(),
            self.not_beacon_counter,
            self.uncompressed_length,
        )
        if layer_idx == 0:
            print(
                f"KV Cache (current_length, beacon_count, not_beacon_counter, uncompressed_length): {before} -> {after}"
            )

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.keys[layer_idx][:, :, : self.current_length, :],
            self.values[layer_idx][:, :, : self.current_length, :],
        )

    def merge_to_beacon(self):
        assert self.not_beacon_counter - 1 == self.beacon_stride
        print(
            f"Merging to beacon, current_length: {self.current_length}, not_beacon_counter: {self.not_beacon_counter}"
        )
        new_beacon_index = self.current_length - self.not_beacon_counter
        for i in range(self.num_hidden_layers):
            self.keys[i][:, :, new_beacon_index, :] = self.keys[i][
                :, :, self.current_length - 1, :
            ]
            self.values[i][:, :, new_beacon_index, :] = self.values[i][
                :, :, self.current_length - 1, :
            ]
        self.not_beacon_counter = 0
        self.current_length = new_beacon_index + 1
        print(
            f"Merged to beacon, current_length: {self.current_length}, not_beacon_counter: {self.not_beacon_counter}"
        )


def mask_mod_document_causal(input_ids: torch.Tensor, bos_token_id: int):
    assert input_ids.ndim == 1
    docs = (input_ids == bos_token_id).cumsum(0)

    def fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx])

    return fn


def reduce_beacon_ids_for_decoding(input_ids: torch.Tensor, beacon_token_id: int):
    is_beacon = input_ids == beacon_token_id
    last_beacon_idx = is_beacon.nonzero(as_tuple=True)[0][-1]
    n_beacon = is_beacon.sum()
    return torch.cat(
        [
            torch.tensor([beacon_token_id] * n_beacon, device=input_ids.device),
            input_ids[last_beacon_idx + 1 :],
        ]
    )


def mask_mod_beacon_document_causal(
    input_ids: torch.Tensor, bos_token_id: int, beacon_token_id: int
):
    assert input_ids.ndim == 1
    docs = (input_ids == bos_token_id).cumsum(0)
    is_beacon = input_ids == beacon_token_id
    beacon_ids = is_beacon.long().cumsum(0) - is_beacon.long()

    def fn(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_beacon = beacon_ids[q_idx] == beacon_ids[kv_idx]
        same_doc = docs[q_idx] == docs[kv_idx]
        return causal & same_doc & (is_beacon[kv_idx] | same_beacon)

    return fn


def decoding_mask_from_cache(kv_cache: KVCache):
    offset = kv_cache.current_length
    return lambda b, h, q_idx, kv_idx: q_idx + offset >= kv_idx


def make_block_mask(
    input_ids: torch.Tensor,
    bos_token_id: int,
    beacon_token_id: Optional[int] = None,
    mask_type: str = "causal_document",
    decoding: bool = False,
    return_block_mask: bool = True,
):
    if mask_type == "causal_document":
        mask_mod = mask_mod_document_causal(input_ids, bos_token_id)
    elif mask_type == "beacon_causal_document":
        mask_mod = mask_mod_beacon_document_causal(
            input_ids, bos_token_id, beacon_token_id
        )
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")
    if return_block_mask:
        return flex_attention.create_block_mask(
            mask_mod=mask_mod,
            B=None,
            H=None,
            Q_LEN=input_ids.size(0) if not decoding else 1,
            KV_LEN=input_ids.size(0),
            device=input_ids.device,
        )
    return mask_mod


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def round_up_to_multiple(v: int | float, *, n: int) -> int:
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 100_000, max_seq_len: int = 2048):
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
        if position is None:
            seqlen = x.size(2)
            cos = self.cos[:seqlen, :]
            sin = self.sin[:seqlen, :]
        else:
            assert x.size(2) == 1
            cos = self.cos[position, :].unsqueeze(0)
            sin = self.sin[position, :].unsqueeze(0)

        x_odd = x[:, :, :, 1::2].float()
        x_even = x[:, :, :, ::2].float()
        y_odd = x_odd * cos - x_even * sin
        y_even = x_odd * sin + x_even * cos
        y = torch.stack([y_odd, y_even], dim=-1)
        return y.flatten(start_dim=3).type_as(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        max_position_embeddings: int,
        head_dim: int,
        num_key_value_heads: int = 1,
        rope_theta: int = 100_000,
        rms_norm_eps: float = 1e-6,
        query_pre_attn_scalar: int = 256,
    ):
        super().__init__()
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.rotary = RotaryEmbedding(
            head_dim=self.head_dim, max_seq_len=max_position_embeddings, base=rope_theta
        )
        self.q_proj = DTypeLinear(
            hidden_size, num_attention_heads * head_dim, bias=False
        )
        self.k_proj = DTypeLinear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = DTypeLinear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = DTypeLinear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )
        self.q_norm = ScaledRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = ScaledRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.scaling = query_pre_attn_scalar**-0.5
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, DTypeLinear):
                in_features = m.in_features
                std = 0.5 * (in_features**-0.5)
                bound = (3.0**0.5) * std
                with torch.no_grad():
                    m.weight.uniform_(-bound, bound)
        self.o_proj.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        block_mask: flex_attention.BlockMask,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        kv_cache_args: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, seqlen, _ = x.shape
        q = (
            self.q_proj(x)
            .view(bsz, seqlen, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        if kv_cache is not None:
            if kv_cache_args["prefill"]:
                q = self.rotary(q)
                k = self.rotary(k)
                if kv_cache_args["beacon_mask"] is not None:
                    _k = k[:, :, kv_cache_args["beacon_mask"], :]
                    _v = v[:, :, kv_cache_args["beacon_mask"], :]
                else:
                    _k = k
                    _v = v
                kv_cache.update(
                    layer_idx,
                    _k,
                    _v,
                    prefill=True,
                    not_beacon_counter=kv_cache_args["not_beacon_counter"],
                    prefill_length=q.size(2),
                )
            else:
                q = self.rotary(q, position=kv_cache.uncompressed_length)
                k = self.rotary(k, position=kv_cache.uncompressed_length)
                kv_cache.update(layer_idx, k, v, prefill=False)
                k, v = kv_cache.get_kv(layer_idx)
        else:
            q = self.rotary(q)
            k = self.rotary(k)

        k = repeat_kv_heads(k, self.num_attention_heads // self.num_key_value_heads)
        v = repeat_kv_heads(v, self.num_attention_heads // self.num_key_value_heads)

        out = flex_attention.flex_attention(
            q, k, v, block_mask=block_mask, scale=self.scaling
        )
        out = out.transpose(1, 2).reshape(
            bsz, seqlen, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(out)


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = DTypeLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = DTypeLinear(intermediate_size, hidden_size, bias=False)
        self.gate_proj = DTypeLinear(hidden_size, intermediate_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        self.down_proj.weight.detach().zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.up_proj(x) * F.gelu(self.gate_proj(x), approximate="tanh")
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        max_position_embeddings: int,
        head_dim: int,
        intermediate_size: int,
        num_key_value_heads: int = 1,
        rope_theta: int = 100_000,
        rms_norm_eps: float = 1e-6,
        query_pre_attn_scalar: int = 256,
    ):
        super().__init__()
        self.input_layernorm = ScaledRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = ScaledRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = ScaledRMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = ScaledRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = MultiHeadSelfAttention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            query_pre_attn_scalar=query_pre_attn_scalar,
        )
        self.mlp = GatedMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        block_mask: flex_attention.BlockMask,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        kv_cache_args: Optional[dict] = None,
    ) -> torch.Tensor:
        residual = x
        x_attn = self.self_attn(
            self.input_layernorm(x),
            block_mask,
            kv_cache,
            layer_idx,
            kv_cache_args,
        )
        x = residual + self.post_attention_layernorm(x_attn)
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


@dataclass
class TransformerConfig:
    bos_token_id: int
    eos_token_id: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    initializer_range: float = 0.02
    query_pre_attn_scalar: int = 256
    vocab_size: int = 32000
    rope_scaling: Optional[dict] = None


class TransformerModel(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        beacon_token_id: int,
        *,
        beacon_stride: int = 0,
    ):
        super().__init__()
        vocab_size = round_up_to_multiple(config.vocab_size + 1, n=16)
        self.config = config
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    num_attention_heads=config.num_attention_heads,
                    hidden_size=config.hidden_size,
                    max_position_embeddings=config.max_position_embeddings,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    num_key_value_heads=config.num_key_value_heads,
                    rope_theta=int(config.rope_theta),
                    rms_norm_eps=config.rms_norm_eps,
                    query_pre_attn_scalar=config.query_pre_attn_scalar,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.lm_head = DTypeLinear(config.hidden_size, vocab_size, bias=False)
        self.beacon_stride = beacon_stride
        self.beacon_token_id = beacon_token_id
        self.norm = ScaledRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        print(self.embed_tokens.weight.mean(), self.embed_tokens.weight.std())

    def resize_token_embeddings(self, new_num_tokens: int):
        old_weight = self.embed_tokens.weight
        if new_num_tokens > self.vocab_size:
            self.embed_tokens = nn.Embedding(new_num_tokens, self.config.hidden_size)
            self.embed_tokens.weight.data[: self.vocab_size] = old_weight
        elif new_num_tokens < self.vocab_size:
            self.embed_tokens.weight.data = self.embed_tokens.weight.data[
                :new_num_tokens
            ]
        self.vocab_size = new_num_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[flex_attention.BlockMask] = None,
        kv_cache: Optional[KVCache] = None,
        kv_cache_args: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert input_ids.ndim == 1
        _kv_args = {
            "prefill": False,
            "use_beacon": False,
            "beacon_mask": None,
            "not_beacon_counter": 0,
        }
        if kv_cache_args is not None:
            _kv_args.update(kv_cache_args)

        if kv_cache is not None and _kv_args.get("prefill") and kv_cache.use_beacon:
            beacon_mask = input_ids == self.beacon_token_id
            nz = beacon_mask.nonzero(as_tuple=True)[0]
            last_beacon_index = nz[-1] if nz.size(0) != 0 else -1
            not_beacon_counter = int(input_ids.size(0) - last_beacon_index - 1)
            beacon_mask[last_beacon_index + 1 :] = True
            _kv_args.update(
                {
                    "beacon_mask": beacon_mask,
                    "not_beacon_counter": not_beacon_counter,
                }
            )

        x = self.embed_tokens(input_ids.unsqueeze(0))
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                block_mask=mask,
                kv_cache=kv_cache,
                layer_idx=i,
                kv_cache_args=_kv_args,
            )
            # print(f"layer-{i}(mean={x.mean()}, std={x.std()})")
        x = self.norm(x)
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
        use_beacon: bool = False,
    ):
        assert input_ids.ndim == 1
        kv_length = round_up_to_multiple(input_ids.size(0) + max_new_tokens, n=16)
        kv_cache = KVCache(
            num_hidden_layers=self.config.num_hidden_layers,
            head_dim=self.config.head_dim,
            num_key_value_heads=self.config.num_key_value_heads,
            max_position_embeddings=kv_length,
            beacon_stride=self.beacon_stride,
            device=input_ids.device,
            dtype=self.embed_tokens.weight.dtype,
            use_beacon=use_beacon,
        )
        pre_mask = make_block_mask(
            input_ids=input_ids,
            bos_token_id=self.config.bos_token_id,
            beacon_token_id=self.beacon_token_id,
            mask_type="beacon_causal_document" if use_beacon else "causal_document",
            decoding=False,
        )
        logits, _ = self.forward(
            input_ids, mask=pre_mask, kv_cache=kv_cache, kv_cache_args={"prefill": True}
        )
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=False)
        decoded = torch.cat([input_ids, next_tok], dim=0)
        cur = next_tok

        for _ in range(max_new_tokens):
            if kv_cache.need_new_beacon():
                old_len = kv_cache.current_length
                beacon_mask = flex_attention.create_block_mask(
                    mask_mod=lambda b, h, q_idx, kv_idx: old_len + 1 >= kv_idx,
                    B=None,
                    H=None,
                    Q_LEN=1,
                    KV_LEN=old_len + 1,
                    device=cur.device,
                )
                logits, _ = self.forward(
                    torch.tensor([self.beacon_token_id], device=cur.device),
                    mask=beacon_mask,
                    kv_cache=kv_cache,
                    kv_cache_args={"prefill": False},
                )
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=False)
                decoded = torch.cat([decoded[:-1], next_tok], dim=0)
                cur = next_tok
                kv_cache.merge_to_beacon()
                continue

            old_len = kv_cache.current_length
            step_mask = flex_attention.create_block_mask(
                mask_mod=lambda b, h, q_idx, kv_idx: old_len + 1 >= kv_idx,
                B=None,
                H=None,
                Q_LEN=1,
                KV_LEN=old_len + 1,
                device=cur.device,
            )
            logits, _ = self.forward(
                cur, mask=step_mask, kv_cache=kv_cache, kv_cache_args={"prefill": False}
            )
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=False)
            decoded = torch.cat([decoded, next_tok], dim=0)
            cur = next_tok

        return decoded


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    num_tok_added = tokenizer.add_special_tokens({"cls_token": "<|beacon|>"})
    print(tokenizer.vocab_size, tokenizer.bos_token_id, tokenizer.cls_token_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TransformerConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        head_dim=256,
        hidden_size=640,
        intermediate_size=2048,
        max_position_embeddings=32768,
        num_attention_heads=4,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        initializer_range=0.02,
        query_pre_attn_scalar=256,
        vocab_size=tokenizer.vocab_size,
        rope_scaling=None,
    )

    model = TransformerModel(
        config=config,
        beacon_token_id=tokenizer.cls_token_id,
        beacon_stride=16,
    ).to(device)

    try:
        state_dict = load_file("./ckpt/model.safetensors")
        renamed_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(renamed_state_dict, strict=False)
    except Exception as e:
        print(e)
        raise SystemExit(1)

    new_vocab = ((tokenizer.vocab_size + num_tok_added + 16) // 16) * 16
    model.resize_token_embeddings(new_vocab)
    print(model.embed_tokens.weight.shape)
    print(model)

    input_ids = torch.randint(0, 100, (10,), device=device)
    labels = torch.randint(0, 100, (10,), device=device)
    mask = make_block_mask(
        input_ids=input_ids,
        bos_token_id=tokenizer.bos_token_id,
        beacon_token_id=tokenizer.cls_token_id,
        mask_type="causal_document",
    )
    logits, loss = model(input_ids, labels, mask)
    print(logits.shape)
    print(loss)

    sample_text = "In the US, "
    print("Generation")
    input_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids, device=device)
    print(input_ids)

    output = model.generate(input_ids, max_new_tokens=4, use_beacon=False)
    print(output)
    print(tokenizer.decode(output))
