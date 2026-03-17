from typing import Optional
import torch


class KVCache:
    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.cur_seq_len = 0

        # Initialize the cache for keys and values
        self.data = torch.zeros(
            (2, max_seq_len, num_kv_heads, head_dim),
            dtype=torch.float16,
        )

    def store_kv_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        seq_len = keys.shape[0]
        self.data[0, self.cur_seq_len : self.cur_seq_len + seq_len] = keys
        self.data[1, self.cur_seq_len : self.cur_seq_len + seq_len] = values
        self.cur_seq_len += seq_len

    def get_kv_cache(
        self, layer_idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.cur_seq_len == 0:
            return None, None
        keys = self.data[0, : self.cur_seq_len]
        values = self.data[1, : self.cur_seq_len]
        return keys, values

    def clear_kv_cache(self) -> None:
        self.cur_seq_len = 0

    def set_cache_seq_len(self, s: int) -> None:
        assert s <= self.max_seq_len
        self.cur_seq_len = s
