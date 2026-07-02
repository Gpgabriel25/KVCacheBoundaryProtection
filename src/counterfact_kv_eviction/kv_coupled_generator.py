from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import time

from .interfaces import KVBlockMetadata, KVPolicy
from .policies import AdaKVFaithfulPolicy, LRUPolicy, QUESTFaithfulPolicy


@dataclass
class KVPositionMetadata:
    """Metadata tracked per logical KV position."""

    position: int
    token_id: int
    insert_step: int
    last_access_step: int
    access_count: int
    attention_score: float = 0.0
    max_attention_score: float = 0.0
    cumulative_attention_score: float = 0.0
    prefill_attention_score: float = 0.0


@dataclass
class KVCoupledGeneratorResult:
    """Container for Stage 1 KV-coupled generation state."""

    generated_token_ids: List[int]
    generated_text: str
    eviction_count: int
    positions_retained: int
    positions_evicted: List[int]
    retention_mask: List[bool]
    policy_call_steps: List[int]
    retained_counts_by_step: List[int]
    prefill_position_count: int
    faithful_union_retained_max: int = 0
    faithful_union_overflow_max: int = 0
    faithful_union_overflow_steps: int = 0
    policy_compute_time_s: float = 0.0
    prefill_wall_s: float = 0.0
    decode_phase_wall_s: float = 0.0
    decode_step_n: int = 0
    decode_step_mean_ms: float = 0.0
    decode_step_p50_ms: float = 0.0
    decode_step_p95_ms: float = 0.0
    decode_step_p99_ms: float = 0.0


def _decode_step_percentiles_ms(step_times_ms: List[float]) -> Tuple[int, float, float, float, float]:
    """Return (n_steps, mean_ms, p50_ms, p95_ms, p99_ms) for one generation's decode forwards."""
    if not step_times_ms:
        return 0, 0.0, 0.0, 0.0, 0.0
    a = np.asarray(step_times_ms, dtype=np.float64)
    n = int(a.size)
    return (
        n,
        float(np.mean(a)),
        float(np.percentile(a, 50)),
        float(np.percentile(a, 95)),
        float(np.percentile(a, 99)),
    )


def position_meta_to_block(position_meta: KVPositionMetadata) -> KVBlockMetadata:
    """Convert position metadata into existing policy metadata shape."""

    return KVBlockMetadata(
        key=str(position_meta.position),
        last_access_step=position_meta.last_access_step,
        insert_step=position_meta.insert_step,
        access_count=position_meta.access_count,
        attention_score=position_meta.attention_score,
        max_attention_score=position_meta.max_attention_score,
        cumulative_attention_score=position_meta.cumulative_attention_score,
        prefill_attention_score=position_meta.prefill_attention_score,
    )


class KVCoupledQwen35Generator:
    """Manual decode loop with policy-driven retention masking."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        policy: KVPolicy,
        capacity: int,
        evict_every_n_steps: int = 8,
        n_kv_heads: int = 0,
        eviction_regime: str = "decode",
    ) -> None:
        """Initialise the generator.

        Args:
            eviction_regime: When to apply the eviction policy.
                ``'decode'`` (default) — evicts after prefill *and* at each
                ``evict_every_n_steps`` decode step (existing behaviour).
                ``'prefill'`` — evicts once after prefill only; no decode-time
                eviction.  Matches PagedAttention / SnapKV reference semantics.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if evict_every_n_steps <= 0:
            raise ValueError("evict_every_n_steps must be > 0")
        if eviction_regime not in {"decode", "prefill"}:
            raise ValueError("eviction_regime must be 'decode' or 'prefill'")
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy
        self.capacity = int(capacity)
        self.evict_every_n_steps = int(evict_every_n_steps)
        self.eviction_regime = eviction_regime
        # Detect per-head mode through wrapper (e.g. ProtectedPolicyWrapper)
        _inner = getattr(policy, 'inner', policy)
        self._per_head_mode = isinstance(_inner, (AdaKVFaithfulPolicy, QUESTFaithfulPolicy))
        self._per_head_policy = _inner if self._per_head_mode else None
        self._per_head_uses_current = isinstance(_inner, QUESTFaithfulPolicy)
        self._n_kv_heads = n_kv_heads
        # Extract protection bounds from wrapper if present
        self._prefix_protect = 0
        self._suffix_protect = 0
        if self._per_head_mode and hasattr(policy, 'prefix_frac'):
            cap = getattr(policy, 'capacity', 0) or capacity
            if getattr(policy, 'prefix_frac', 0) > 0:
                self._prefix_protect = max(4, int(cap * policy.prefix_frac))
            if getattr(policy, 'suffix_frac', 0) > 0:
                self._suffix_protect = max(4, int(cap * policy.suffix_frac))
        self._orig_prompt_len = 0  # set in generate_with_kv_control

    def generate_with_kv_control(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: set[int] | None = None,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> KVCoupledGeneratorResult:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not input_ids:
            raise ValueError("input_ids must be non-empty")

        generated_ids = [int(tok) for tok in input_ids]
        position_metadata: List[KVPositionMetadata] = []
        retention_mask: List[bool] = []
        positions_evicted: List[int] = []
        policy_call_steps: List[int] = []
        retained_counts_by_step: List[int] = []
        faithful_union_retained_max = 0
        faithful_union_overflow_max = 0
        faithful_union_overflow_steps = 0
        self._policy_time_s = 0.0
        _decode_step_ms: List[float] = []

        _t_prefill0 = time.perf_counter()
        # LRU policy uses only recency (insert_step/last_access_step), not
        # attention scores.  Skip output_attentions to avoid allocating the
        # full [n_layers, n_q_heads, chunk, max_cache_len] tensor (~10GB/device
        # on v5e at 33K context), which causes OOM on 16GB-HBM chips.
        _inner_policy = getattr(self.policy, "inner", self.policy)  # unwrap ProtectedPolicyWrapper
        _need_attn = not isinstance(_inner_policy, LRUPolicy)
        prefill_output = self.model(
            input_ids=list(input_ids),
            output_attentions=_need_attn,
            use_cache=True,
        )
        prefill_wall_s = time.perf_counter() - _t_prefill0
        _t_after_prefill = time.perf_counter()
        past_key_values = getattr(prefill_output, "past_key_values", None)
        prefill_scores = _attention_max_per_key(getattr(prefill_output, "attentions", None))

        n_prefill = len(input_ids)
        self._orig_prompt_len = n_prefill  # Track original prompt for static suffix protection
        for pos, token_id in enumerate(input_ids):
            score = prefill_scores[pos] if pos < len(prefill_scores) else 0.0
            position_metadata.append(
                KVPositionMetadata(
                    position=pos,
                    token_id=int(token_id),
                    insert_step=0,
                    last_access_step=0,
                    access_count=1,
                    attention_score=score,
                    max_attention_score=score,
                    cumulative_attention_score=score,
                    prefill_attention_score=score,
                )
            )
            retention_mask.append(True)

        # ── Preallocate numpy arrays for per-step metadata hot path ──────────
        # Avoids an O(N_positions) Python loop every decode step by doing the
        # attention-score / access-count updates with vectorised numpy ops.
        # Synced back to position_metadata only at policy-call cadence.
        _max_alloc = n_prefill + max_new_tokens + 2
        _attn_score = np.zeros(_max_alloc, dtype=np.float32)
        _max_attn = np.zeros(_max_alloc, dtype=np.float32)
        _cumul_attn = np.zeros(_max_alloc, dtype=np.float64)
        _last_access = np.zeros(_max_alloc, dtype=np.int32)
        _access_count = np.ones(_max_alloc, dtype=np.int32)
        for p, score in enumerate(prefill_scores[:n_prefill]):
            _attn_score[p] = score
            _max_attn[p] = score
            _cumul_attn[p] = score

        # ── Parallel float list for attention-mask (avoids list comprehension) ─
        # Maintained in sync with retention_mask: set to 0.0 when pos evicted.
        _retention_float: List[float] = [1.0] * n_prefill
        _prev_evict_count = 0

        # ── Per-head tracking for faithful Ada-KV ────────────────────────────
        _per_head_cumul: np.ndarray | None = None
        _per_head_current: np.ndarray | None = None
        _per_head_retention: np.ndarray | None = None  # (H, n_pos) bool mask
        _per_head_current_ready = False
        if self._per_head_mode and self._n_kv_heads > 0:
            _per_head_cumul = np.zeros(
                (self._n_kv_heads, _max_alloc), dtype=np.float64)
            _per_head_current = np.zeros(
                (self._n_kv_heads, _max_alloc), dtype=np.float32)
            _per_head_retention = np.ones(
                (self._n_kv_heads, _max_alloc), dtype=bool)

        prefill_position_count = len(position_metadata)
        retained_counts_by_step.append(_retained_count(retention_mask))
        if not self._per_head_mode:
            self._maybe_evict(
                step=0,
                position_metadata=position_metadata,
                retention_mask=retention_mask,
                positions_evicted=positions_evicted,
                policy_call_steps=policy_call_steps,
                force_if_over_capacity=True,
            )
        else:
            # ── Per-head post-prefill eviction (AdaKV / QUEST) ──────────────
            # Mirrors the global _maybe_evict(step=0, force_if_over_capacity=True)
            # path.  Runs for BOTH eviction_regime values when n_prefill exceeds
            # capacity.  The decode-loop per-head call stays gated on
            # eviction_regime == "decode" (see decode loop below).
            if n_prefill > self.capacity and _per_head_retention is not None:
                # Seed per-head signal arrays from prefill attention output.
                _prefill_attn_raw = getattr(prefill_output, "attentions", None)
                if _prefill_attn_raw is not None and len(_prefill_attn_raw) > 0:
                    _ph_arr = np.asarray(_prefill_attn_raw[0], dtype=np.float32)
                    if _ph_arr.ndim == 2:
                        # (H, n_prefill) — compact per-head format from jax bridge
                        _ph_hh = min(_ph_arr.shape[0],
                                     _per_head_cumul.shape[0] if _per_head_cumul is not None else 0)
                        _ph_cc = min(_ph_arr.shape[1], n_prefill)
                    elif _ph_arr.ndim == 4:
                        # (batch, H, q_len, k_len) — standard HuggingFace format;
                        # use last query position's attention as per-head signal.
                        _ph_arr = _ph_arr[0, :, -1, :]  # (H, k_len)
                        _ph_hh = min(_ph_arr.shape[0],
                                     _per_head_cumul.shape[0] if _per_head_cumul is not None else 0)
                        _ph_cc = min(_ph_arr.shape[1], n_prefill)
                    else:
                        _ph_hh, _ph_cc = 0, 0
                    if _ph_hh > 0 and _ph_cc > 0:
                        if _per_head_cumul is not None:
                            _per_head_cumul[:_ph_hh, :_ph_cc] += _ph_arr[:_ph_hh, :_ph_cc]
                        if _per_head_current is not None:
                            _per_head_current[:_ph_hh, :_ph_cc] = _ph_arr[:_ph_hh, :_ph_cc]
                            _per_head_current_ready = _ph_cc > 0
                _has_ph_signal = (
                    (_per_head_cumul is not None and not self._per_head_uses_current)
                    or (_per_head_current is not None
                        and self._per_head_uses_current
                        and _per_head_current_ready)
                )
                if _has_ph_signal:
                    _policy_t0 = time.perf_counter()
                    per_head_signal = (
                        _per_head_current if self._per_head_uses_current else _per_head_cumul
                    )
                    ph_mask = self._per_head_policy.select_per_head_retention(
                        per_head_signal[:, :n_prefill],
                        retention_mask,
                        self.capacity,
                        n_protected=self._prefix_protect,
                        n_suffix_protected=self._suffix_protect,
                        orig_prompt_len=self._orig_prompt_len,
                    )
                    union_mask = ph_mask.any(axis=0)
                    union_retained = int(union_mask.sum())
                    faithful_union_retained_max = max(
                        faithful_union_retained_max, union_retained)
                    union_overflow = max(0, union_retained - self.capacity)
                    faithful_union_overflow_max = max(
                        faithful_union_overflow_max, union_overflow)
                    if union_overflow > 0:
                        faithful_union_overflow_steps += 1
                    # ── Overflow guardrail (same logic as decode-loop per-head) ──
                    if union_retained > self.capacity:
                        _prot = np.zeros(n_prefill, dtype=bool)
                        _prot[:self._prefix_protect] = True
                        if self._suffix_protect > 0:
                            _ret_arr = np.array(retention_mask[:n_prefill], dtype=bool)
                            _ret_pos = np.where(_ret_arr)[0]
                            if len(_ret_pos) >= self._suffix_protect:
                                _prot[_ret_pos[-self._suffix_protect:]] = True
                        _evict_cand = np.where(union_mask & ~_prot)[0]
                        if len(_evict_cand) > 0:
                            _votes = ph_mask[:, _evict_cand].sum(axis=0).astype(
                                np.float64)
                            _attn = per_head_signal[:, _evict_cand].sum(axis=0)
                            _order = np.lexsort((_attn, _votes))
                            _n_trim = min(union_retained - self.capacity,
                                          len(_evict_cand))
                            _trim_idx = _evict_cand[_order[:_n_trim]]
                            union_mask[_trim_idx] = False
                            ph_mask[:, _trim_idx] = False
                    _per_head_retention[:, :n_prefill] = ph_mask
                    for pos in range(n_prefill):
                        if retention_mask[pos] and not union_mask[pos]:
                            retention_mask[pos] = False
                            positions_evicted.append(pos)
                    policy_call_steps.append(0)
                    self._policy_time_s += (time.perf_counter() - _policy_t0)
        # Sync newly-evicted positions to _retention_float and per-head mask
        for pos in positions_evicted[_prev_evict_count:]:
            _retention_float[pos] = 0.0
            if _per_head_retention is not None:
                _per_head_retention[:, pos] = False
        _prev_evict_count = len(positions_evicted)
        retained_counts_by_step.append(_retained_count(retention_mask))

        last_logits = np.asarray(getattr(prefill_output, "logits"))

        for step in range(1, max_new_tokens + 1):
            logit_vec = last_logits[0, -1]
            if temperature > 0.0 and rng is not None:
                scaled = logit_vec / temperature
                scaled -= scaled.max()  # numerical stability
                probs = np.exp(scaled) / np.exp(scaled).sum()
                next_token_id = int(rng.choice(len(probs), p=probs))
            else:
                next_token_id = int(np.argmax(logit_vec))
            generated_ids.append(next_token_id)

            if eos_token_ids and next_token_id in eos_token_ids:
                break

            new_position = len(position_metadata)
            position_metadata.append(
                KVPositionMetadata(
                    position=new_position,
                    token_id=next_token_id,
                    insert_step=step,
                    last_access_step=step,
                    access_count=1,
                    attention_score=0.0,
                    max_attention_score=0.0,
                )
            )
            retention_mask.append(True)
            _retention_float.append(1.0)
            _last_access[new_position] = step  # _access_count already 1 from alloc

            should_call_policy = step % self.evict_every_n_steps == 0
            # Sync numpy → dataclass only when the policy will actually run
            if should_call_policy:
                _sync_numpy_to_meta(
                    position_metadata, _attn_score, _max_attn,
                    _cumul_attn, _last_access, _access_count,
                )
            if (
                self._per_head_mode
                and should_call_policy
                and self.eviction_regime == "decode"
                and _per_head_retention is not None
                and (
                    (_per_head_cumul is not None and not self._per_head_uses_current)
                    or (_per_head_current is not None and self._per_head_uses_current and _per_head_current_ready)
                )
            ):
                _policy_t0 = time.perf_counter()
                n_pos = len(position_metadata)
                per_head_signal = _per_head_current if self._per_head_uses_current else _per_head_cumul
                ph_mask = self._per_head_policy.select_per_head_retention(
                    per_head_signal[:, :n_pos],
                    retention_mask,
                    self.capacity,
                    n_protected=self._prefix_protect,
                    n_suffix_protected=self._suffix_protect,
                    orig_prompt_len=self._orig_prompt_len,
                )
                # Global retention = union of all heads' retention
                union_mask = ph_mask.any(axis=0)
                union_retained = int(union_mask.sum())
                faithful_union_retained_max = max(faithful_union_retained_max, union_retained)
                union_overflow = max(0, union_retained - self.capacity)
                faithful_union_overflow_max = max(faithful_union_overflow_max, union_overflow)
                if union_overflow > 0:
                    faithful_union_overflow_steps += 1

                # ── Overflow guardrail: trim union to hard global capacity ──
                # Per-head policies can produce a union >> capacity because
                # each head independently selects positions.  Trim by evicting
                # the lowest-consensus positions (fewest heads selected them),
                # breaking ties by lowest aggregate attention signal.
                if union_retained > self.capacity:
                    _prot = np.zeros(n_pos, dtype=bool)
                    _prot[:self._prefix_protect] = True
                    if self._suffix_protect > 0:
                        _ret_arr = np.array(retention_mask[:n_pos], dtype=bool)
                        _ret_pos = np.where(_ret_arr)[0]
                        if len(_ret_pos) >= self._suffix_protect:
                            _prot[_ret_pos[-self._suffix_protect:]] = True
                    _evict_cand = np.where(union_mask & ~_prot)[0]
                    if len(_evict_cand) > 0:
                        _votes = ph_mask[:, _evict_cand].sum(axis=0).astype(np.float64)
                        _attn = per_head_signal[:, _evict_cand].sum(axis=0)
                        _order = np.lexsort((_attn, _votes))  # ascending
                        _n_trim = min(union_retained - self.capacity,
                                      len(_evict_cand))
                        _trim_idx = _evict_cand[_order[:_n_trim]]
                        union_mask[_trim_idx] = False
                        ph_mask[:, _trim_idx] = False

                _per_head_retention[:, :n_pos] = ph_mask
                for pos in range(n_pos):
                    if retention_mask[pos] and not union_mask[pos]:
                        retention_mask[pos] = False
                        positions_evicted.append(pos)
                policy_call_steps.append(step)
                self._policy_time_s += (time.perf_counter() - _policy_t0)
            elif self.eviction_regime == "decode":
                self._maybe_evict(
                    step=step,
                    position_metadata=position_metadata,
                    retention_mask=retention_mask,
                    positions_evicted=positions_evicted,
                    policy_call_steps=policy_call_steps,
                    force_if_over_capacity=False,
                )
            # Maintain _retention_float in sync with any evictions
            for pos in positions_evicted[_prev_evict_count:]:
                _retention_float[pos] = 0.0
                if _per_head_retention is not None:
                    _per_head_retention[:, pos] = False
            _prev_evict_count = len(positions_evicted)
            retained_counts_by_step.append(_retained_count(retention_mask))

            if step == max_new_tokens:
                break

            # Build attention mask: per-head (2D ndarray) or global (1D list).
            # When n_kv_heads > 0, broadcast global retention to (H, n_pos) so the
            # JAX adapter uses the same per-head decode JIT as Ada-KV/QUEST faithful.
            # A 1D mask alone selects a different compiled specialization; wall-clock
            # decode-step timing would not be comparable (LRU looked artificially slower).
            if self._per_head_mode and _per_head_retention is not None:
                n_pos = len(position_metadata)
                _attn_mask_arg = _per_head_retention[:, :n_pos].astype(np.float32)
            elif self._n_kv_heads > 0:
                n_pos = len(position_metadata)
                rf = np.asarray(_retention_float[:n_pos], dtype=np.float32)
                _attn_mask_arg = np.broadcast_to(rf, (self._n_kv_heads, n_pos)).copy()
            else:
                _attn_mask_arg = _retention_float

            _t_dec0 = time.perf_counter()
            decode_output = self.model(
                input_ids=[next_token_id],
                past_key_values=past_key_values,
                attention_mask=_attn_mask_arg,
                output_attentions=True,
                use_cache=True,
            )
            _decode_step_ms.append((time.perf_counter() - _t_dec0) * 1000.0)
            past_key_values = getattr(decode_output, "past_key_values", past_key_values)
            last_logits = np.asarray(getattr(decode_output, "logits"))

            # Extract attention: per-head (H, C) array or global per-key list
            raw_attn = getattr(decode_output, "attentions", None)
            if self._per_head_mode and raw_attn is not None and len(raw_attn) > 0:
                per_head_arr = np.asarray(raw_attn[0], dtype=np.float32)  # (H, C)
                n_pos = len(position_metadata)
                if _per_head_cumul is not None and per_head_arr.ndim == 2:
                    hh = min(per_head_arr.shape[0], _per_head_cumul.shape[0])
                    cc = min(per_head_arr.shape[1], n_pos)
                    _per_head_cumul[:hh, :cc] += per_head_arr[:hh, :cc]
                if _per_head_current is not None and per_head_arr.ndim == 2:
                    hh = min(per_head_arr.shape[0], _per_head_current.shape[0])
                    cc = min(per_head_arr.shape[1], n_pos)
                    _per_head_current[:, :n_pos] = 0.0
                    _per_head_current[:hh, :cc] = per_head_arr[:hh, :cc]
                    _per_head_current_ready = cc > 0
                # Also update global scores for logging consistency
                key_scores = per_head_arr.max(axis=0).tolist()
            else:
                key_scores = _attention_max_per_key(raw_attn)
            # Threshold-based access: only mark positions as "accessed" if their
            # attention score clears a relative threshold. This prevents access_count
            # and last_access_step from degenerating to "age in cache" (which would
            # collapse LRU→FIFO and break H2O/SnapKV baselines).
            if key_scores:
                n_pos = len(position_metadata)
                ks = np.asarray(key_scores[:n_pos], dtype=np.float32)
                rm = np.asarray(retention_mask[:n_pos], dtype=bool)
                max_score = float(ks.max()) if n_pos > 0 else 0.0
                access_threshold = np.float32(0.05 * max_score)
                # Vectorised update: evicted positions (rm=False) are left unchanged.
                _attn_score[:n_pos] = np.where(rm, ks, _attn_score[:n_pos])
                np.maximum(_max_attn[:n_pos], rm * ks, out=_max_attn[:n_pos])
                _cumul_attn[:n_pos] += rm * ks  # bool*float32 broadcast, cumul is float64
                accessed = rm & (ks > access_threshold)
                _last_access[:n_pos] = np.where(accessed, np.int32(step), _last_access[:n_pos])
                _access_count[:n_pos] += accessed.astype(np.int32)

        generated_text = ""
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            generated_text = str(self.tokenizer.decode(generated_ids))

        decode_phase_wall_s = time.perf_counter() - _t_after_prefill

        _ds_n, _ds_m, _ds_p50, _ds_p95, _ds_p99 = _decode_step_percentiles_ms(_decode_step_ms)

        return KVCoupledGeneratorResult(
            generated_token_ids=generated_ids,
            generated_text=generated_text,
            eviction_count=len(positions_evicted),
            positions_retained=_retained_count(retention_mask),
            positions_evicted=positions_evicted,
            retention_mask=list(retention_mask),
            policy_call_steps=policy_call_steps,
            retained_counts_by_step=retained_counts_by_step,
            prefill_position_count=prefill_position_count,
            faithful_union_retained_max=faithful_union_retained_max,
            faithful_union_overflow_max=faithful_union_overflow_max,
            faithful_union_overflow_steps=faithful_union_overflow_steps,
            policy_compute_time_s=float(self._policy_time_s),
            prefill_wall_s=float(prefill_wall_s),
            decode_phase_wall_s=float(decode_phase_wall_s),
            decode_step_n=int(_ds_n),
            decode_step_mean_ms=float(_ds_m),
            decode_step_p50_ms=float(_ds_p50),
            decode_step_p95_ms=float(_ds_p95),
            decode_step_p99_ms=float(_ds_p99),
        )

    def _maybe_evict(
        self,
        step: int,
        position_metadata: List[KVPositionMetadata],
        retention_mask: List[bool],
        positions_evicted: List[int],
        policy_call_steps: List[int],
        force_if_over_capacity: bool,
    ) -> None:
        retained = _retained_positions(retention_mask)
        over_capacity = len(retained) > self.capacity
        should_call = (step % self.evict_every_n_steps == 0) or (force_if_over_capacity and over_capacity)
        if not should_call:
            return

        evict_count = max(0, len(retained) - self.capacity)
        blocks = {
            str(pos): position_meta_to_block(position_metadata[pos])
            for pos in retained
        }
        if self.policy.name.startswith("causal_") and any(not block.has_estimate for block in blocks.values()):
            import logging as _gen_log
            _gen_log.getLogger(__name__).debug(
                "KV-coupled: causal policy %r running without estimate annotations "
                "(non-causal fallback), step=%d",
                self.policy.name,
                step,
            )
        _policy_t0 = time.perf_counter()
        selected = self.policy.select_evictions(blocks=blocks, evict_count=evict_count, step=step)
        self._policy_time_s += (time.perf_counter() - _policy_t0)
        policy_call_steps.append(step)

        for key in selected:
            pos = int(key)
            if 0 <= pos < len(retention_mask) and retention_mask[pos]:
                retention_mask[pos] = False
                positions_evicted.append(pos)

        # Guardrail: always honor hard capacity, even if policy-rate limits
        # produce fewer evictions than currently required.
        retained_after = _retained_positions(retention_mask)
        overflow = len(retained_after) - self.capacity
        if overflow > 0:
            fallback_order = sorted(
                retained_after,
                key=lambda pos: (
                    int(position_metadata[pos].last_access_step),
                    int(position_metadata[pos].insert_step),
                    int(pos),
                ),
            )
            for pos in fallback_order[:overflow]:
                if retention_mask[pos]:
                    retention_mask[pos] = False
                    positions_evicted.append(pos)


def _retained_positions(retention_mask: Sequence[bool]) -> List[int]:
    return [idx for idx, keep in enumerate(retention_mask) if keep]


def _retained_count(retention_mask: Sequence[bool]) -> int:
    return sum(1 for keep in retention_mask if keep)


def _sync_numpy_to_meta(
    position_metadata: List[KVPositionMetadata],
    attn_score: np.ndarray,
    max_attn: np.ndarray,
    cumul_attn: np.ndarray,
    last_access: np.ndarray,
    access_count: np.ndarray,
) -> None:
    """Bulk-sync numpy hot-path arrays → KVPositionMetadata dataclasses.

    Called only at policy-eviction cadence (~evict_every_n_steps) rather than
    every decode step, so the per-position Python loop runs ~16× not 128×.
    """
    for p, pm in enumerate(position_metadata):
        pm.attention_score = float(attn_score[p])
        pm.max_attention_score = float(max_attn[p])
        pm.cumulative_attention_score = float(cumul_attn[p])
        pm.last_access_step = int(last_access[p])
        pm.access_count = int(access_count[p])


def _attention_max_per_key(attentions: Any) -> List[float]:
    if attentions is None:
        return []

    arrays = [np.asarray(layer, dtype=np.float32) for layer in attentions]
    if not arrays:
        return []

    key_len = int(arrays[0].shape[-1])
    try:
        # Flatten batch/head/query dims across all layers and reduce in one pass.
        # After jax_inference Patch-A the common case is a single pre-reduced layer,
        # making this a trivial reshape+max.  For prefill (already fake_layers) it
        # also avoids the sequential np.maximum accumulation loop.
        stacked = np.concatenate([a.reshape(-1, key_len) for a in arrays], axis=0)
        return stacked.max(axis=0).tolist()
    except (ValueError, Exception):
        # Fallback: shapes differ, reduce layer-by-layer
        per_key = np.zeros(key_len, dtype=np.float32)
        for arr in arrays:
            if arr.shape[-1] == key_len:
                np.maximum(per_key, arr.reshape(-1, key_len).max(axis=0), out=per_key)
        return per_key.tolist()
