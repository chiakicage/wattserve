from transformers import LlamaConfig


def get_llama_config_8B():
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128001,  # Llama 3 通常 pad 等于 eos
    )
    return config


def get_llama_config_13B():
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return config


def get_llama_config_32B():
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=6144,  # 增加 hidden size
        intermediate_size=16384,  # 增加 FFN size
        num_hidden_layers=60,  # 增加层数以达到约 32B 的参数量
        num_attention_heads=48,
        num_key_value_heads=8,  # 依然保持 GQA (分组查询注意力)
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128001,
    )
    return config
