from transformers import LlamaConfig


LLAMA2_VOCAB_SIZE = 32000
LLAMA2_MAX_POSITION_EMBEDDINGS = 16384
# Keep some headroom for CUDA context, allocator fragmentation, and
# runtime workspaces on an A100 40GB card.
A100_40G_TOTAL_MEMORY_GIB = 40.0
A100_40G_RESERVED_MEMORY_GIB = 7.0

BF16_BYTES = 2
FP16_BYTES = 2


LLAMA2_MODEL_SPECS = {
    "7B": {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    },
    "13B": {
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 40,
    },
    # Llama 2 does not have an official 34B checkpoint. We use the common
    # Llama-family 34B tensor shape here and trim only the layer count when
    # fitting to the A100 40GB budget.
    "34B": {
        "hidden_size": 8192,
        "intermediate_size": 22016,
        "num_hidden_layers": 48,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
    },
    "70B": {
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
    },
}


def _get_head_dim(config: LlamaConfig) -> int:
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    assert config.hidden_size is not None
    assert config.num_attention_heads is not None
    return config.hidden_size // config.num_attention_heads


def calculate_llama_parameter_count(
    config: LlamaConfig,
    include_lm_head: bool = False,
    num_hidden_layers: int | None = None,
) -> int:
    """Estimate parameter count for the current LlamaModel implementation."""
    assert config.hidden_size is not None
    assert config.intermediate_size is not None
    assert config.num_attention_heads is not None
    assert config.num_key_value_heads is not None
    assert config.vocab_size is not None

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_layers = (
        config.num_hidden_layers
        if num_hidden_layers is None
        else num_hidden_layers
    )
    assert num_layers is not None

    head_dim = _get_head_dim(config)
    q_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim

    attn_params = (
        hidden_size * q_size
        + hidden_size * kv_size
        + hidden_size * kv_size
        + q_size * hidden_size
    )
    mlp_params = 3 * hidden_size * intermediate_size
    norm_params = 2 * hidden_size + 2 * head_dim
    params_per_layer = attn_params + mlp_params + norm_params

    embedding_params = config.vocab_size * hidden_size
    final_norm_params = hidden_size

    total_params = (
        embedding_params + num_layers * params_per_layer + final_norm_params
    )
    if include_lm_head:
        total_params += hidden_size * config.vocab_size

    return total_params


def calculate_llama_prefill_flops(
    config: LlamaConfig,
    prompt_len: int,
    num_hidden_layers: int | None = None,
    replace_ln: bool = False,
    replace_attention: bool = False,
    replace_rope: bool = False,
    replace_activation: bool = False,
) -> int:
    """
    Estimate prefill FLOPs for one forward pass.

    This follows the benchmark's simplified accounting:
    - projections and MLP dominate linear algebra cost
    - attention score/value cost is approximated with hidden_size
    - elementwise ops such as RMSNorm are intentionally omitted
    - only replace_attention changes the counted matmul path
    """
    assert config.hidden_size is not None
    assert config.intermediate_size is not None
    assert config.num_attention_heads is not None
    assert config.num_key_value_heads is not None
    del replace_ln, replace_rope, replace_activation

    num_layers = (
        config.num_hidden_layers
        if num_hidden_layers is None
        else num_hidden_layers
    )
    assert num_layers is not None

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    head_dim = _get_head_dim(config)
    q_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim
    seq_len = prompt_len

    qkv_flops = seq_len * hidden_size * (q_size + 2 * kv_size)
    if replace_attention:
        attn_scores_flops = 0
        attn_values_flops = 0
    else:
        attn_scores_flops = seq_len * seq_len * q_size
        attn_values_flops = seq_len * seq_len * q_size
    out_proj_flops = seq_len * q_size * hidden_size
    ffn_flops = 3 * seq_len * hidden_size * intermediate_size

    flops_per_layer = (
        qkv_flops
        + attn_scores_flops
        + attn_values_flops
        + out_proj_flops
        + ffn_flops
    )

    return 2 * flops_per_layer * num_layers


def calculate_llama_runtime_memory_bytes(
    config: LlamaConfig,
    include_lm_head: bool = True,
    num_hidden_layers: int | None = None,
    weight_dtype_bytes: int = BF16_BYTES,
) -> int:
    """
    Estimate persistent runtime memory for TTFT benchmarking.

    This includes weights and lm_head, but excludes transient activations,
    workspaces, and KV cache because the TTFT-only benchmark does not keep
    persistent KV state.
    """
    num_layers = (
        config.num_hidden_layers
        if num_hidden_layers is None
        else num_hidden_layers
    )
    assert num_layers is not None

    parameter_count = calculate_llama_parameter_count(
        config,
        include_lm_head=include_lm_head,
        num_hidden_layers=num_layers,
    )
    return parameter_count * weight_dtype_bytes


def fit_llama_num_hidden_layers_to_memory(
    config: LlamaConfig,
    total_memory_gib: float = A100_40G_TOTAL_MEMORY_GIB,
    reserved_memory_gib: float = A100_40G_RESERVED_MEMORY_GIB,
    include_lm_head: bool = True,
) -> int:
    """Reduce layer count until persistent memory fits the target budget."""
    assert config.num_hidden_layers is not None
    if total_memory_gib <= reserved_memory_gib:
        raise ValueError(
            "total_memory_gib must be larger than reserved_memory_gib"
        )

    available_bytes = int((total_memory_gib - reserved_memory_gib) * (1024**3))
    for num_layers in range(config.num_hidden_layers, 0, -1):
        estimated_bytes = calculate_llama_runtime_memory_bytes(
            config,
            include_lm_head=include_lm_head,
            num_hidden_layers=num_layers,
        )
        if estimated_bytes <= available_bytes:
            return num_layers
    return 1


def _build_llama2_config(model_size: str) -> LlamaConfig:
    spec = LLAMA2_MODEL_SPECS[model_size]
    config = LlamaConfig()
    config.vocab_size = LLAMA2_VOCAB_SIZE
    config.hidden_size = spec["hidden_size"]
    config.intermediate_size = spec["intermediate_size"]
    config.num_hidden_layers = spec["num_hidden_layers"]
    config.num_attention_heads = spec["num_attention_heads"]
    config.num_key_value_heads = spec["num_key_value_heads"]
    config.hidden_act = "silu"
    config.max_position_embeddings = LLAMA2_MAX_POSITION_EMBEDDINGS
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-5
    config.use_cache = False
    config.tie_word_embeddings = False
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.pad_token_id = 0

    canonical_num_hidden_layers = spec["num_hidden_layers"]
    fitted_num_hidden_layers = fit_llama_num_hidden_layers_to_memory(config)
    setattr(config, "canonical_model_size", model_size)
    setattr(config, "canonical_num_hidden_layers", canonical_num_hidden_layers)
    config.num_hidden_layers = fitted_num_hidden_layers
    setattr(config, "target_gpu", "A100 40GB")
    setattr(config, "total_memory_gib", A100_40G_TOTAL_MEMORY_GIB)
    setattr(config, "reserved_memory_gib", A100_40G_RESERVED_MEMORY_GIB)
    setattr(
        config,
        "canonical_parameter_count",
        calculate_llama_parameter_count(
            config,
            num_hidden_layers=canonical_num_hidden_layers,
        ),
    )
    setattr(
        config,
        "canonical_parameter_count_with_lm_head",
        calculate_llama_parameter_count(
            config,
            include_lm_head=True,
            num_hidden_layers=canonical_num_hidden_layers,
        ),
    )
    setattr(config, "parameter_count", calculate_llama_parameter_count(config))
    setattr(
        config,
        "parameter_count_with_lm_head",
        calculate_llama_parameter_count(
            config,
            include_lm_head=True,
        ),
    )
    setattr(
        config,
        "estimated_runtime_memory_bytes",
        calculate_llama_runtime_memory_bytes(
            config,
            include_lm_head=True,
        ),
    )
    return config


def get_llama_config_7B() -> LlamaConfig:
    return _build_llama2_config("7B")


def get_llama_config_13B() -> LlamaConfig:
    return _build_llama2_config("13B")


def get_llama_config_34B() -> LlamaConfig:
    return _build_llama2_config("34B")


def get_llama_config_70B() -> LlamaConfig:
    return _build_llama2_config("70B")
