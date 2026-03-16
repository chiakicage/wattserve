from transformers import Qwen3Config


def get_qwen3_config_4B():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=36,
        attention_dropout=0.0,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    return config


def get_qwen3_config_8B():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=36,
        attention_dropout=0.0,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    return config


def get_qwen3_config_14B():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
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
        eos_token_id=151645,
    )
    return config


def get_qwen3_config_32B():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=25600,
        num_hidden_layers=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=64,
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
