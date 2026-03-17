"""Inference-only LLaMA model compatible with HuggingFace weights."""

from typing import Optional

import torch
from torch import nn
from transformers import Qwen3Config

from cache import KVCache

from flashinfer import (
    silu_and_mul,
    fused_add_rmsnorm,
    rmsnorm,
    apply_rope_pos_ids,
    single_prefill_with_kv_cache,
    single_decode_with_kv_cache,
)

__all__ = [
    "Qwen3Model",
]


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
        else:
            residual = x
            x = rmsnorm(x, self.weight, self.variance_epsilon)
        return x, residual


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.gate_proj(x)
        x2 = self.up_proj(x)
        x = torch.cat([x1, x2], dim=-1)
        x = silu_and_mul(x)
        x = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_idx: int,
        rope_theta: float = 10000,
        max_position_embeddings: int = 8192,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.num_heads
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        # self.qkv_proj = nn.Linear(
        #     hidden_size,
        #     (self.num_heads + self.num_kv_heads * 2) * self.head_dim,
        #     bias=False,
        # )
        self.q_proj = nn.Linear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        self.apply_rope = lambda q, k, pos_ids: apply_rope_pos_ids(
            q, k, pos_ids, rotary_dim=self.head_dim, rope_theta=rope_theta
        )

        self.cache = KVCache(
            self.num_kv_heads, self.head_dim, self.max_position_embeddings
        )
        assert config.rms_norm_eps is not None
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # qkv = self.qkv_proj(hidden_states)
        # q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)
        q, k = apply_rope_pos_ids(
            q,
            k,
            positions,
            rotary_dim=self.head_dim,
            rope_theta=self.rope_theta,
        )
        old_cache_len = self.cache.cur_seq_len
        input_len = q.shape[0]
        self.cache.store_kv_cache(k, v)
        if old_cache_len == 0:  # Prefill
            o = single_prefill_with_kv_cache(
                q, k, v, causal=True, sm_scale=self.scaling
            )
        elif input_len == 1:  # Decode
            k_cache, v_cache = self.cache.get_kv_cache(self.layer_idx)
            assert k_cache is not None
            assert v_cache is not None
            q = q.view(self.num_heads, self.head_dim)
            o = single_decode_with_kv_cache(
                q, k_cache, v_cache, sm_scale=self.scaling
            )
        else:  # Extend
            o = q

        o = o.view(-1, self.num_heads * self.head_dim)
        output = self.o_proj(o)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        assert config.num_attention_heads is not None
        assert config.hidden_size is not None
        assert config.intermediate_size is not None
        assert config.rms_norm_eps is not None
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        self.layer_idx = layer_idx

        max_position_embeddings = getattr(
            config, "max_position_embeddings", 8192
        )

        self.self_attn = Qwen3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            layer_idx=layer_idx,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        assert residual is not None
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()

        self.config = config
        assert config.num_hidden_layers is not None
        assert config.hidden_size is not None
        assert config.vocab_size is not None
        assert config.rms_norm_eps is not None
        self.vocab_size = config.vocab_size
        self.layer_indices = list(range(config.num_hidden_layers))
        self.layers = nn.ModuleDict()
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        for i in self.layer_indices:
            self.layers[str(i)] = Qwen3DecoderLayer(
                config=config,
                layer_idx=i,
            )

    def clear_kv_cache(self) -> None:
        for layer in self.layers.values():
            assert isinstance(layer, Qwen3DecoderLayer)
            layer.self_attn.cache.clear_kv_cache()

    def forward(
        self,
        positions: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positions = positions.view(-1)
        residual = None
        hidden_states = None
        if input_embeds is not None:
            hidden_states = input_embeds
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)

        assert hidden_states is not None

        for i in self.layer_indices:
            hidden_states, residual = self.layers[str(i)](
                positions, hidden_states, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
