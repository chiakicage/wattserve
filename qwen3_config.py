from transformers import Qwen3Config


def get_qwen3_config_4B():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=24,
        attention_dropout=0.0,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151643,
    )
    return config


def get_qwen3_config_8B():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151643,
    )
    return config


def get_qwen3_config_14B():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=32,
        attention_dropout=0.0,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151643,
    )
    return config


def get_qwen3_config_32B():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=8192,
        intermediate_size=44032,
        num_hidden_layers=56,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=48,
        attention_dropout=0.0,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151643,
    )
    return config


if __name__ == "__main__":
    import json

    configs = {
        "qwen3_4B": get_qwen3_config_4B().to_dict(),
        "qwen3_8B": get_qwen3_config_8B().to_dict(),
        "qwen3_14B": get_qwen3_config_14B().to_dict(),
        "qwen3_32B": get_qwen3_config_32B().to_dict(),
    }

    for name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"{name} Config:")
        print(f"{'='*50}")
        print(json.dumps(config, indent=2))