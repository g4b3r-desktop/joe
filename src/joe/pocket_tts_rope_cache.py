"""Experimental cached RoPE tables for the CUDA Pocket TTS path.

The upstream Pocket TTS 4.0.1 implementation rebuilds frequencies and trigonometric
values for every attention call.  This adapter caches FP32 cos/sin tables per shared
RotaryEmbedding instance.  It is opt-in and restores every patched method on close.
"""
from __future__ import annotations

import math
from functools import wraps


class _SingleRotaryCache:
    def __init__(self, rope, torch, *, initial_positions: int) -> None:
        self.rope = rope
        self.torch = torch
        self.max_period = float(rope.max_period)
        self.capacity = max(1, int(initial_positions))
        self.tables: dict[tuple[object, int, float], tuple[object, object]] = {}
        self.calls = 0
        self.fallbacks = 0

        namespace = vars(rope)
        self._had_instance_forward = "forward" in namespace
        self._original_instance_forward = namespace.get("forward")
        self._original_forward = rope.forward

        @wraps(self._original_forward)
        def cached_forward(q, k, offset):
            return self.forward(q, k, offset)

        self._patched_forward = cached_forward
        setattr(rope, "forward", cached_forward)

    def _build_table(self, device, head_dim: int, capacity: int):
        torch = self.torch
        half_dim = head_dim // 2
        dimensions = torch.arange(half_dim, device=device, dtype=torch.float32)
        frequencies = torch.exp(
            dimensions * (-math.log(self.max_period) * 2.0 / head_dim)
        )
        positions = torch.arange(capacity, device=device, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]
        cos_table = torch.cos(angles).contiguous()
        sin_table = torch.sin(angles).contiguous()
        return cos_table, sin_table

    def _tables_for(self, q):
        key = (q.device, int(q.shape[-1]), self.max_period)
        tables = self.tables.get(key)
        if tables is None:
            tables = self._build_table(q.device, int(q.shape[-1]), self.capacity)
            self.tables[key] = tables
        return key, tables

    def ensure_capacity(self, required_positions: int) -> None:
        """Grow tables before Pocket TTS expands its autoregressive KV cache."""
        required = max(1, int(required_positions))
        if required <= self.capacity:
            return
        new_capacity = max(required, self.capacity * 2)
        for key in tuple(self.tables):
            device, head_dim, _max_period = key
            self.tables[key] = self._build_table(device, head_dim, new_capacity)
        self.capacity = new_capacity

    def forward(self, q, k, offset):
        torch = self.torch
        if (
            not isinstance(q, torch.Tensor)
            or not isinstance(k, torch.Tensor)
            or q.ndim != 4
            or k.ndim != 4
        ):
            self.fallbacks += 1
            return self._original_forward(q, k, offset)

        batch, time_steps, _heads, head_dim = q.shape
        key_batch, key_steps, _key_heads, key_dim = k.shape
        if (
            batch != key_batch
            or time_steps != key_steps
            or head_dim != key_dim
            or head_dim <= 0
            or head_dim % 2
            or q.device != k.device
            or q.dtype != k.dtype
            or not math.isfinite(self.max_period)
            or self.max_period <= 0
        ):
            self.fallbacks += 1
            return self._original_forward(q, k, offset)

        positions = torch.arange(time_steps, device=q.device, dtype=torch.long)
        if isinstance(offset, torch.Tensor):
            if offset.numel() != 1 or offset.dtype not in (torch.int32, torch.int64):
                self.fallbacks += 1
                return self._original_forward(q, k, offset)
            if offset.device.type != q.device.type:
                self.fallbacks += 1
                return self._original_forward(q, k, offset)
            positions = positions + offset.to(device=q.device, dtype=torch.long).reshape(())
        elif isinstance(offset, int):
            if offset < 0 or offset + time_steps > self.capacity:
                self.fallbacks += 1
                return self._original_forward(q, k, offset)
            positions = positions + offset
        else:
            self.fallbacks += 1
            return self._original_forward(q, k, offset)

        _, (cos_table, sin_table) = self._tables_for(q)
        # _expand_kv_cache is patched at the model boundary to grow the tables
        # before each generation chunk. This keeps CUDA offsets on-device and
        # avoids a .item() synchronization on every autoregressive attention call.
        if time_steps > self.capacity:
            self.fallbacks += 1
            return self._original_forward(q, k, offset)

        try:
            cos = cos_table.index_select(0, positions).view(1, time_steps, 1, head_dim // 2)
            sin = sin_table.index_select(0, positions).view(1, time_steps, 1, head_dim // 2)
        except IndexError:
            # Python offsets can be checked without a device sync. CUDA tensor offsets
            # are guaranteed by the model's pre-generation KV-cache expansion hook.
            self.fallbacks += 1
            return self._original_forward(q, k, offset)

        q_pairs = q.view(batch, time_steps, q.shape[2], head_dim // 2, 2)
        k_pairs = k.view(batch, time_steps, k.shape[2], head_dim // 2, 2)
        qr, qi = q_pairs[..., 0].float(), q_pairs[..., 1].float()
        kr, ki = k_pairs[..., 0].float(), k_pairs[..., 1].float()

        q_real = qr * cos - qi * sin
        q_imag = qr * sin + qi * cos
        k_real = kr * cos - ki * sin
        k_imag = kr * sin + ki * cos

        dtype = q.dtype
        q_rotated = torch.stack((q_real.to(dtype), q_imag.to(dtype)), dim=-1)
        k_rotated = torch.stack((k_real.to(dtype), k_imag.to(dtype)), dim=-1)
        self.calls += 1
        return (
            q_rotated.view(batch, time_steps, q.shape[2], head_dim),
            k_rotated.view(batch, time_steps, k.shape[2], head_dim),
        )

    @property
    def allocated_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tables in self.tables.values()
            for tensor in tables
        )

    def close(self) -> None:
        if getattr(self.rope, "forward", None) is self._patched_forward:
            if self._had_instance_forward:
                setattr(self.rope, "forward", self._original_instance_forward)
            else:
                delattr(self.rope, "forward")
        self.tables.clear()


class PocketTTSRopeCache:
    """Own and restore the shared RoPE and KV-cache-expansion patches."""

    def __init__(self, model, torch, *, initial_positions: int = 8192) -> None:
        flow_lm = getattr(model, "flow_lm", None)
        transformer = getattr(flow_lm, "transformer", None)
        candidates = []
        transformer_rope = getattr(transformer, "rope", None)
        if transformer_rope is not None:
            candidates.append(transformer_rope)
        for layer in tuple(getattr(transformer, "layers", ())):
            rope = getattr(getattr(layer, "self_attn", None), "rope", None)
            if rope is not None:
                candidates.append(rope)

        unique_ropes = []
        seen_ids = set()
        for rope in candidates:
            if id(rope) not in seen_ids and callable(getattr(rope, "forward", None)):
                seen_ids.add(id(rope))
                unique_ropes.append(rope)
        if not unique_ropes:
            raise RuntimeError(
                "A versão instalada do Pocket TTS não expõe o RotaryEmbedding do Flow LM."
            )

        self._caches = [
            _SingleRotaryCache(rope, torch, initial_positions=initial_positions)
            for rope in unique_ropes
        ]
        self._model = model
        self._closed = False
        self._original_expand = getattr(model, "_expand_kv_cache", None)
        if not callable(self._original_expand):
            for cache in reversed(self._caches):
                cache.close()
            raise RuntimeError(
                "A versão instalada do Pocket TTS não expõe _expand_kv_cache; "
                "não é seguro usar o cache RoPE sem garantir a capacidade das posições."
            )
        self._had_instance_expand = "_expand_kv_cache" in vars(model)
        self._original_instance_expand = vars(model).get("_expand_kv_cache")
        original_expand = self._original_expand

        @wraps(original_expand)
        def expand_with_rope_cache(model_state, sequence_length: int):
            for cache in self._caches:
                cache.ensure_capacity(int(sequence_length))
            return original_expand(model_state, sequence_length)

        self._patched_expand = expand_with_rope_cache
        setattr(model, "_expand_kv_cache", expand_with_rope_cache)

    @property
    def calls(self) -> int:
        return sum(cache.calls for cache in self._caches)

    @property
    def fallbacks(self) -> int:
        return sum(cache.fallbacks for cache in self._caches)

    @property
    def capacity_positions(self) -> int:
        return max((cache.capacity for cache in self._caches), default=0)

    @property
    def allocated_bytes(self) -> int:
        return sum(cache.allocated_bytes for cache in self._caches)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if (
            self._patched_expand is not None
            and getattr(self._model, "_expand_kv_cache", None) is self._patched_expand
        ):
            if self._had_instance_expand:
                setattr(self._model, "_expand_kv_cache", self._original_instance_expand)
            else:
                delattr(self._model, "_expand_kv_cache")
        for cache in reversed(self._caches):
            cache.close()


def install_cuda_rope_cache(
    model, torch, *, initial_positions: int = 8192
) -> PocketTTSRopeCache:
    """Install the experimental cache for a single benchmark/generation scope."""
    return PocketTTSRopeCache(model, torch, initial_positions=initial_positions)
