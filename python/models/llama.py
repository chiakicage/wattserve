"""Inference-only LLaMA model compatible with HuggingFace weights."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import LlamaConfig

from flashinfer import (
    silu_and_mul,
    fused_add_rmsnorm,
    rmsnorm,
    apply_rope_pos_ids,
    single_prefill_with_kv_cache,
)

__all__ = [
    "LlamaAblationConfig",
    "LlamaModel",
]


@dataclass(frozen=True)
class LlamaAblationConfig:
    replace_ln: bool = False
    replace_attention: bool = False
    replace_rope: bool = False
    replace_activation: bool = False


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        ablation_config: LlamaAblationConfig | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.ablation_config = ablation_config or LlamaAblationConfig()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ablation_config.replace_ln:
            residual = x
            return x, residual
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
        else:
            residual = x
            x = rmsnorm(x, self.weight, self.variance_epsilon)
        return x, residual


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ablation_config: LlamaAblationConfig | None = None,
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.ablation_config = ablation_config or LlamaAblationConfig()
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
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.ablation_config.replace_activation:
            x = up
        else:
            x = torch.cat([gate, up], dim=-1)
            x = silu_and_mul(x)
        x = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        ablation_config: LlamaAblationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.ablation_config = ablation_config or LlamaAblationConfig()
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.num_heads
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(
            hidden_size,
            self.q_size,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            self.kv_size,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.q_size,
            self.hidden_size,
            bias=False,
        )

        assert config.rms_norm_eps is not None
        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            ablation_config=self.ablation_config,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            ablation_config=self.ablation_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)
        if not self.ablation_config.replace_rope:
            q, k = apply_rope_pos_ids(
                q,
                k,
                positions,
                rotary_dim=self.head_dim,
                rope_theta=self.rope_theta,
            )
        if self.ablation_config.replace_attention:
            o = q
        else:
            o = single_prefill_with_kv_cache(
                q, k, v, causal=True, sm_scale=self.scaling
            )
        o = o.reshape(-1, self.q_size)
        output = self.o_proj(o)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        ablation_config: LlamaAblationConfig | None = None,
    ) -> None:
        super().__init__()
        assert config.hidden_size is not None
        assert config.intermediate_size is not None
        assert config.num_attention_heads is not None
        assert config.rms_norm_eps is not None
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        self.layer_idx = layer_idx
        self.ablation_config = ablation_config or LlamaAblationConfig()

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,
            ablation_config=self.ablation_config,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            ablation_config=self.ablation_config,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            ablation_config=self.ablation_config,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            ablation_config=self.ablation_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        assert residual is not None
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        replace_ln: bool = False,
        replace_attention: bool = False,
        replace_rope: bool = False,
        replace_activation: bool = False,
    ):
        super().__init__()

        self.config = config
        self.ablation_config = LlamaAblationConfig(
            replace_ln=replace_ln,
            replace_attention=replace_attention,
            replace_rope=replace_rope,
            replace_activation=replace_activation,
        )
        assert config.num_hidden_layers is not None
        assert config.hidden_size is not None
        assert config.rms_norm_eps is not None
        assert config.vocab_size is not None
        self.vocab_size = config.vocab_size
        self.layer_indices = list(range(config.num_hidden_layers))
        self.layers = nn.ModuleDict()
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            ablation_config=self.ablation_config,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        for i in self.layer_indices:
            self.layers[str(i)] = LlamaDecoderLayer(
                config=config,
                layer_idx=i,
                ablation_config=self.ablation_config,
            )

    def clear_kv_cache(self) -> None:
        # TTFT-only benchmarking does not keep KV cache state.
        return None

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
