from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from .estimator import OnlineCausalCreditEstimator
from .interfaces import KVBlockMetadata, build_feature_vector
from .kv_coupled_generator import (
    KVCoupledGeneratorResult,
    KVCoupledQwen35Generator,
    KVPositionMetadata,
    _attention_max_per_key,
    _retained_count,
    _retained_positions,
    _sync_numpy_to_meta,
    position_meta_to_block,
)
from .policies import CausalCreditsPolicy, LRUPolicy
from .wrappers import UncertaintyGatedPolicy, annotate_blocks_with_estimates


@dataclass
class OnlineCreditGeneratorResult(KVCoupledGeneratorResult):
    """Extended result with online credit learning stats."""

    estimator_update_count: int = 0
    gate_opened_steps: List[int] = field(default_factory=list)
    gate_fallback_steps: List[int] = field(default_factory=list)
    mean_log_prob: float = 0.0
    quality_deltas: List[float] = field(default_factory=list)
    demotion_count: int = 0
    positions_demoted: List[int] = field(default_factory=list)


@dataclass
class _PendingCreditAssignment:
    """Tracks an eviction event waiting for quality-after observation."""

    eviction_step: int
    quality_before: float
    evicted_features: List[List[float]]
    evicted_attention_weights: List[float]  # for proportional attribution


class OnlineCreditGenerator:
    """Wraps KVCoupledQwen35Generator with online counterfactual credit learning."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        capacity: int,
        evict_every_n_steps: int = 8,
        estimator: Optional[OnlineCausalCreditEstimator] = None,
        uncertainty_threshold: float = 0.35,
        min_warmup_updates: int = 8,
        credit_lookback: int = 8,
        credit_lookahead: int = 6,
        noise_threshold: float = 0.003,
        prefix_protect_frac: float = 0.10,
        suffix_protect_frac: float = 0.10,
        reference_capacity: int = 128,
        demotion_ratio: float = 0.0,
        ablation_random_credit: bool = False,
        ablation_no_priors: bool = False,
        force_credit_policy: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if evict_every_n_steps <= 0:
            raise ValueError("evict_every_n_steps must be > 0")

        self.model = model
        self.tokenizer = tokenizer
        self.capacity = int(capacity)
        self.reference_capacity = int(reference_capacity)

        # Fidelity ladder: at a fixed memory budget (capacity * bf16),
        # trade precision for quantity by demoting low-credit positions to int8.
        # demotion_ratio = fraction of budget reallocated from bf16 to int8.
        # Example: capacity=256, demotion_ratio=0.25 → 192 bf16 + 128 int8 = 320 total.
        self.demotion_ratio = max(0.0, min(1.0, float(demotion_ratio)))
        if self.demotion_ratio > 0:
            self.bf16_slots = int(self.capacity * (1.0 - self.demotion_ratio))
            self.int8_slots = int(self.capacity * self.demotion_ratio * 2)
            self.total_retained = self.bf16_slots + self.int8_slots
        else:
            self.bf16_slots = self.capacity
            self.int8_slots = 0
            self.total_retained = self.capacity

        # V4c adaptive scaling was tested and found harmful (DECISIONS.md 2026-03-09).
        # Use fixed hyperparameters regardless of capacity.
        self.evict_every_n_steps = int(evict_every_n_steps)

        self.estimator = estimator if estimator is not None else OnlineCausalCreditEstimator(
            warmup_min_updates=min_warmup_updates,
        )
        # Ablation: remove informative priors (zero-init weights)
        if ablation_no_priors:
            self.estimator.weights = [0.0] * len(self.estimator.weights)
            self.estimator.bias = 0.0
        self.ablation_random_credit = ablation_random_credit
        self.force_credit_policy = force_credit_policy
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.credit_lookback = int(credit_lookback)
        self.credit_lookahead = int(credit_lookahead)
        self.noise_threshold = float(noise_threshold)

        # Token protection: reserve capacity for prefix (question/instruction)
        # and suffix (recent working memory) positions.
        # Allow explicit zero for ablation studies (no minimum clamp).
        if prefix_protect_frac == 0.0:
            self.prefix_protect = 0
        else:
            self.prefix_protect = max(4, int(self.capacity * prefix_protect_frac))
        if suffix_protect_frac == 0.0:
            self.suffix_protect = 0
        else:
            self.suffix_protect = max(4, int(self.capacity * suffix_protect_frac))

        candidate = CausalCreditsPolicy(
            uncertainty_penalty=0.2,
            on_missing_estimates="fallback_lru",
        )
        fallback = LRUPolicy()
        self.policy = UncertaintyGatedPolicy(
            candidate_policy=candidate,
            fallback_policy=fallback,
            uncertainty_threshold=self.uncertainty_threshold,
            min_samples=3,
        )

    def generate_with_online_credit(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: set[int] | None = None,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> OnlineCreditGeneratorResult:
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

        # Online credit tracking
        log_probs: List[float] = []
        pending_credits: deque[_PendingCreditAssignment] = deque()
        gate_opened_steps: List[int] = []
        gate_fallback_steps: List[int] = []
        quality_deltas: List[float] = []
        estimator_update_count = 0

        # Fidelity ladder tracking
        already_demoted: Set[int] = set()
        positions_demoted: List[int] = []
        pending_demotions: List[int] = []

        # --- Prefill ---
        prefill_output = self.model(
            input_ids=list(input_ids),
            output_attentions=True,
            use_cache=True,
        )
        past_key_values = getattr(prefill_output, "past_key_values", None)
        prefill_scores = _attention_max_per_key(getattr(prefill_output, "attentions", None))

        n_prefill = len(input_ids)
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
                )
            )
            retention_mask.append(True)

        # ── Preallocate numpy arrays for per-step metadata hot path ──────────
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

        # Parallel float list for attention-mask (avoids [1.0 if k else 0.0 for k in …])
        _retention_float: List[float] = [1.0] * n_prefill
        _prev_evict_count = 0

        prefill_position_count = len(position_metadata)
        retained_counts_by_step.append(_retained_count(retention_mask))

        # Initial eviction if over capacity
        self._evict_with_gating(
            step=0,
            position_metadata=position_metadata,
            retention_mask=retention_mask,
            positions_evicted=positions_evicted,
            policy_call_steps=policy_call_steps,
            gate_opened_steps=gate_opened_steps,
            gate_fallback_steps=gate_fallback_steps,
            pending_credits=pending_credits,
            log_probs=log_probs,
            force_if_over_capacity=True,
            pending_demotions=pending_demotions,
            already_demoted=already_demoted,
        )
        # Sync retention_float for any initial evictions
        for pos in positions_evicted[_prev_evict_count:]:
            _retention_float[pos] = 0.0
        _prev_evict_count = len(positions_evicted)
        # Apply initial demotions to KV cache
        if pending_demotions and past_key_values is not None:
            past_key_values = self._demote_kv_positions(
                past_key_values, pending_demotions, already_demoted, positions_demoted,
            )
            pending_demotions.clear()
        retained_counts_by_step.append(_retained_count(retention_mask))

        last_logits = np.asarray(getattr(prefill_output, "logits"))

        # --- Decode loop ---
        for step in range(1, max_new_tokens + 1):
            next_token_id = int(np.argmax(last_logits[0, -1]))
            # NOTE: online credit always uses greedy for credit estimation fidelity
            generated_ids.append(next_token_id)

            if eos_token_ids and next_token_id in eos_token_ids:
                break

            # Extract log-prob of the selected token
            step_log_prob = _log_prob_of_token(last_logits, next_token_id)
            log_probs.append(step_log_prob)

            # Process pending credit assignments that have enough lookahead
            estimator_update_count += self._process_pending_credits(
                pending_credits=pending_credits,
                log_probs=log_probs,
                current_step=step,
                quality_deltas=quality_deltas,
            )

            # Add new position
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
            _last_access[new_position] = step  # _access_count preinit to 1

            # Eviction: only at policy steps. Unlike the base generator, we do NOT
            # evict at every step — we let positions accumulate between policy calls
            # so each eviction event is a meaningful batch (better credit signal).
            # The attention mask still correctly marks all retained positions.
            should_call_policy = step % self.evict_every_n_steps == 0
            if should_call_policy:
                # Sync numpy arrays → position_metadata before credit/policy logic
                _sync_numpy_to_meta(
                    position_metadata, _attn_score, _max_attn,
                    _cumul_attn, _last_access, _access_count,
                )
                self._evict_with_gating(
                    step=step,
                    position_metadata=position_metadata,
                    retention_mask=retention_mask,
                    positions_evicted=positions_evicted,
                    policy_call_steps=policy_call_steps,
                    gate_opened_steps=gate_opened_steps,
                    gate_fallback_steps=gate_fallback_steps,
                    pending_credits=pending_credits,
                    log_probs=log_probs,
                    force_if_over_capacity=False,
                    pending_demotions=pending_demotions,
                    already_demoted=already_demoted,
                )
                # Sync retention_float for newly evicted positions
                for pos in positions_evicted[_prev_evict_count:]:
                    _retention_float[pos] = 0.0
                _prev_evict_count = len(positions_evicted)
                # Apply demotions to KV cache before next decode step
                if pending_demotions and past_key_values is not None:
                    past_key_values = self._demote_kv_positions(
                        past_key_values, pending_demotions, already_demoted, positions_demoted,
                    )
                    pending_demotions.clear()
            retained_counts_by_step.append(_retained_count(retention_mask))

            if step == max_new_tokens:
                break

            decode_output = self.model(
                input_ids=[next_token_id],
                past_key_values=past_key_values,
                attention_mask=_retention_float,
                output_attentions=True,
                use_cache=True,
            )
            past_key_values = getattr(decode_output, "past_key_values", past_key_values)
            last_logits = np.asarray(getattr(decode_output, "logits"))

            key_scores = _attention_max_per_key(getattr(decode_output, "attentions", None))
            # Threshold-based access: only mark positions as "accessed" if their
            # attention score clears a relative threshold. Prevents access_count
            # and last_access_step from degenerating to "age in cache".
            if key_scores:
                n_pos = len(position_metadata)
                ks = np.asarray(key_scores[:n_pos], dtype=np.float32)
                rm = np.asarray(retention_mask[:n_pos], dtype=bool)
                max_score = float(ks.max()) if n_pos > 0 else 0.0
                access_threshold = np.float32(0.05 * max_score)
                _attn_score[:n_pos] = np.where(rm, ks, _attn_score[:n_pos])
                np.maximum(_max_attn[:n_pos], rm * ks, out=_max_attn[:n_pos])
                _cumul_attn[:n_pos] += rm * ks
                accessed = rm & (ks > access_threshold)
                _last_access[:n_pos] = np.where(accessed, np.int32(step), _last_access[:n_pos])
                _access_count[:n_pos] += accessed.astype(np.int32)

        # Flush remaining pending credits
        estimator_update_count += self._process_pending_credits(
            pending_credits=pending_credits,
            log_probs=log_probs,
            current_step=max_new_tokens + 1,
            quality_deltas=quality_deltas,
            flush_all=True,
        )

        generated_text = ""
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            generated_text = str(self.tokenizer.decode(generated_ids))

        mean_lp = float(np.mean(log_probs)) if log_probs else 0.0

        return OnlineCreditGeneratorResult(
            generated_token_ids=generated_ids,
            generated_text=generated_text,
            eviction_count=len(positions_evicted),
            positions_retained=_retained_count(retention_mask),
            positions_evicted=positions_evicted,
            retention_mask=list(retention_mask),
            policy_call_steps=policy_call_steps,
            retained_counts_by_step=retained_counts_by_step,
            prefill_position_count=prefill_position_count,
            estimator_update_count=estimator_update_count,
            gate_opened_steps=gate_opened_steps,
            gate_fallback_steps=gate_fallback_steps,
            mean_log_prob=mean_lp,
            quality_deltas=quality_deltas,
            demotion_count=len(positions_demoted),
            positions_demoted=positions_demoted,
        )

    def _compute_blend_factor(self, blocks: Dict[str, KVBlockMetadata]) -> float:
        """Compute how much to trust OC vs LRU (0 = pure LRU, 1 = pure OC).

        Three multiplicative factors ensure OC only dominates when it has
        earned trust AND the decision matters:
          1. Uncertainty confidence: estimator must have low prediction error
          2. Capacity discount: per-eviction impact is ~1/capacity, so require
             proportionally more confidence at larger caches
          3. Training maturity: estimator must have enough updates
        """
        if not blocks or len(blocks) < 3:
            return 0.0

        warmup = max(1, self.estimator.warmup_min_updates)

        # During warmup: no credit influence (estimator has too few updates)
        if self.estimator.update_count < warmup:
            return 0.0

        # Post-warmup: progressive blend from 0 to max_blend.
        # Ramps linearly from warmup to 4×warmup updates, then caps.
        # This replaces the previous triple-gated approach (uncertainty^2 ×
        # capacity_discount × maturity) which was too conservative and
        # resulted in the gate never opening in practice.
        max_blend = 0.40
        ramp_length = 3.0 * warmup  # reach max_blend after 3× more updates
        progress = min(1.0, (self.estimator.update_count - warmup) / ramp_length)
        blend = progress * max_blend

        # Safety check: if average uncertainty is extremely high (> 2× threshold),
        # the estimator is clearly unreliable — fall back to LRU.
        avg_unc = sum(b.uncertainty for b in blocks.values()) / len(blocks)
        if avg_unc > 2.0 * self.uncertainty_threshold:
            return 0.0

        return blend

    def _blended_eviction(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
        blend: float,
    ) -> List[str]:
        """Select evictions using rank-blended LRU + OC scoring.

        Each block gets a blended retention rank:
            rank = blend * oc_rank + (1 - blend) * lru_rank
        Lowest-ranked blocks are evicted first.
        """
        items = list(blocks.values())
        n = len(items)
        if n <= evict_count:
            return [b.key for b in items]

        # LRU ranking: 0 = most evictable (least recently used)
        lru_order = sorted(items, key=lambda b: (b.last_access_step, b.insert_step, b.key))
        lru_rank: Dict[str, float] = {}
        denom = max(1, n - 1)
        for rank, b in enumerate(lru_order):
            lru_rank[b.key] = rank / denom

        # OC ranking: 0 = most evictable (lowest credit)
        # Uses same scoring as CausalCreditsPolicy for consistency
        def _oc_score(b: KVBlockMetadata) -> float:
            base = b.estimated_credit - 0.2 * b.uncertainty
            if b.access_count <= 1 and max(b.attention_score, b.max_attention_score) >= 0.70:
                base += 0.25
            return base

        oc_order = sorted(items, key=lambda b: (_oc_score(b), b.last_access_step, b.key))
        oc_rank: Dict[str, float] = {}
        for rank, b in enumerate(oc_order):
            oc_rank[b.key] = rank / denom

        # Blend: lower score = evict first
        scored: Dict[str, float] = {}
        for b in items:
            scored[b.key] = blend * oc_rank[b.key] + (1.0 - blend) * lru_rank[b.key]

        sorted_keys = sorted(scored.keys(), key=lambda k: (scored[k], k))
        return sorted_keys[:evict_count]

    def _evict_with_gating(
        self,
        step: int,
        position_metadata: List[KVPositionMetadata],
        retention_mask: List[bool],
        positions_evicted: List[int],
        policy_call_steps: List[int],
        gate_opened_steps: List[int],
        gate_fallback_steps: List[int],
        pending_credits: deque[_PendingCreditAssignment],
        log_probs: List[float],
        force_if_over_capacity: bool,
        pending_demotions: Optional[List[int]] = None,
        already_demoted: Optional[Set[int]] = None,
    ) -> None:
        retained = _retained_positions(retention_mask)
        over_capacity = len(retained) > self.total_retained
        should_call = (step % self.evict_every_n_steps == 0) or (force_if_over_capacity and over_capacity)
        if not should_call:
            return

        # With fidelity ladder: total_retained > capacity when demotion_ratio > 0.
        # We evict down to total_retained (not capacity), keeping extra demoted positions.
        evict_count = max(0, len(retained) - self.total_retained)
        if evict_count == 0 and self.int8_slots == 0:
            return

        blocks: Dict[str, KVBlockMetadata] = {
            str(pos): position_meta_to_block(position_metadata[pos])
            for pos in retained
        }

        # Protect prefix (question/instruction) and suffix (recent) tokens
        # from eviction. This prevents catastrophic loss at tight budgets.
        protected_keys: set = set()
        sorted_retained = sorted(retained)
        # Protect earliest positions (question/instruction tokens)
        for pos in sorted_retained[:self.prefix_protect]:
            protected_keys.add(str(pos))
        # Protect most recent positions (working memory)
        for pos in sorted_retained[-self.suffix_protect:]:
            protected_keys.add(str(pos))

        evictable_blocks = {k: v for k, v in blocks.items() if k not in protected_keys}
        # If protection leaves too few candidates, reduce protection
        if len(evictable_blocks) < evict_count:
            evictable_blocks = blocks  # fall back to full set

        # Annotate blocks with current estimator predictions
        if self.ablation_random_credit:
            import random
            for block in evictable_blocks.values():
                block.estimated_credit = random.gauss(0.0, 1.0)
                block.uncertainty = 0.1
                block.has_estimate = True
        else:
            annotate_blocks_with_estimates(evictable_blocks, self.estimator, step, capacity=self.capacity)

        # Blended scoring: smoothly interpolate between LRU and OC based on
        # confidence, capacity pressure, and training maturity.
        # force_credit_policy bypasses the blend computation entirely (ablation).
        if self.force_credit_policy:
            blend = 1.0
        else:
            blend = self._compute_blend_factor(evictable_blocks)
        if blend >= 0.01:
            gate_opened_steps.append(step)
        else:
            gate_fallback_steps.append(step)

        if blend < 0.01:
            selected = LRUPolicy().select_evictions(
                blocks=evictable_blocks, evict_count=evict_count, step=step,
            ) if evict_count > 0 else []
        else:
            selected = self._blended_eviction(
                evictable_blocks, evict_count, step, blend,
            ) if evict_count > 0 else []
        policy_call_steps.append(step)

        # Collect features and attention weights for credit assignment before evicting
        evicted_features: List[List[float]] = []
        evicted_attention_weights: List[float] = []
        for key in selected:
            pos = int(key)
            if 0 <= pos < len(retention_mask) and retention_mask[pos]:
                evicted_features.append(
                    build_feature_vector(evictable_blocks[key], step, capacity=self.capacity)
                )
                # Record current attention for proportional attribution
                # (using current score, not lifetime max, to avoid stale spikes)
                evicted_attention_weights.append(
                    float(evictable_blocks[key].attention_score)
                )
                retention_mask[pos] = False
                positions_evicted.append(pos)

        # Queue pending credit assignment if we evicted anything
        if evicted_features and log_probs:
            lookback_start = max(0, len(log_probs) - self.credit_lookback)
            quality_before = float(np.mean(log_probs[lookback_start:]))
            pending_credits.append(_PendingCreditAssignment(
                eviction_step=step,
                quality_before=quality_before,
                evicted_features=evicted_features,
                evicted_attention_weights=evicted_attention_weights,
            ))

        # Fidelity demotion: among non-evicted positions, the lowest-credit
        # positions (up to int8_slots) are demoted to int8 precision.
        # Only positions not already demoted are considered, and we cap the
        # total living demoted count at int8_slots to prevent unbounded
        # accumulation over long generations.
        if self.int8_slots > 0 and pending_demotions is not None:
            evicted_set = set(str(pos) for pos in positions_evicted)
            surviving_blocks = {
                k: v for k, v in evictable_blocks.items()
                if k not in evicted_set and retention_mask[int(k)]
            }
            if surviving_blocks:
                # Count how many surviving positions are already demoted
                living_demoted = sum(
                    1 for b in surviving_blocks.values()
                    if int(b.key) in already_demoted
                )
                slots_available = max(0, self.int8_slots - living_demoted)
                if slots_available > 0:
                    def _credit_score(b: KVBlockMetadata) -> float:
                        return b.estimated_credit - 0.2 * b.uncertainty

                    # Only consider non-demoted survivors
                    candidates = [
                        b for b in surviving_blocks.values()
                        if int(b.key) not in already_demoted
                    ]
                    ranked = sorted(
                        candidates,
                        key=lambda b: (_credit_score(b), b.last_access_step, b.key),
                    )
                    for b in ranked[:slots_available]:
                        pos = int(b.key)
                        if pos not in set(pending_demotions):
                            pending_demotions.append(pos)

        # Hard-capacity guardrail (uses total_retained, not capacity)
        self._hard_capacity_evict(step, position_metadata, retention_mask, positions_evicted)

    def _hard_capacity_evict(
        self,
        step: int,
        position_metadata: List[KVPositionMetadata],
        retention_mask: List[bool],
        positions_evicted: List[int],
    ) -> None:
        retained_after = _retained_positions(retention_mask)
        overflow = len(retained_after) - self.total_retained
        if overflow <= 0:
            return
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

    def _demote_kv_positions(
        self,
        past_key_values: Any,
        positions: List[int],
        already_demoted: Set[int],
        positions_demoted: List[int],
    ) -> Any:
        """Lossy demotion: quantize KV vectors at specified positions to int8 precision.

        Performs an absmax int8 round-trip (bf16 → int8 → bf16) per head, destroying
        ~7 bits of mantissa precision while preserving the approximate direction of
        each KV vector. Positions that have already been demoted are skipped
        (re-quantizing is approximately idempotent but wasteful).

        Returns the updated past_key_values tuple.
        """
        if not _HAS_JAX or not positions:
            return past_key_values

        new_positions = [p for p in positions if p not in already_demoted]
        if not new_positions:
            return past_key_values

        cache_keys, cache_values, cache_pos = past_key_values
        # Ensure lists (not tuples) so we can mutate individual layers
        if isinstance(cache_keys, tuple):
            cache_keys = list(cache_keys)
        if isinstance(cache_values, tuple):
            cache_values = list(cache_values)
        pos_arr = jnp.array(new_positions, dtype=jnp.int32)

        for layer_idx in range(len(cache_keys)):
            ck = cache_keys[layer_idx]   # (1, num_kv_heads, max_cache_len, head_dim)
            cv = cache_values[layer_idx]

            # Extract slices for all demoted positions at once
            # Fancy index: ck[0, :, pos_arr, :] → (num_kv_heads, n_pos, head_dim)
            k_slice = ck[0, :, pos_arr, :]
            v_slice = cv[0, :, pos_arr, :]

            # Per-head-per-position absmax quantization to int8 range
            k_scale = jnp.max(jnp.abs(k_slice), axis=-1, keepdims=True) / 127.0
            v_scale = jnp.max(jnp.abs(v_slice), axis=-1, keepdims=True) / 127.0

            k_deq = jnp.clip(
                jnp.round(k_slice / jnp.maximum(k_scale, 1e-10)), -127, 127
            ) * k_scale
            v_deq = jnp.clip(
                jnp.round(v_slice / jnp.maximum(v_scale, 1e-10)), -127, 127
            ) * v_scale

            # Write back to cache (scatter update)
            cache_keys[layer_idx] = ck.at[0, :, pos_arr, :].set(
                k_deq.astype(jnp.bfloat16)
            )
            cache_values[layer_idx] = cv.at[0, :, pos_arr, :].set(
                v_deq.astype(jnp.bfloat16)
            )

        already_demoted.update(new_positions)
        positions_demoted.extend(new_positions)
        return (cache_keys, cache_values, cache_pos)

    def _process_pending_credits(
        self,
        pending_credits: deque[_PendingCreditAssignment],
        log_probs: List[float],
        current_step: int,
        quality_deltas: List[float],
        flush_all: bool = False,
    ) -> int:
        updates = 0
        while pending_credits:
            assignment = pending_credits[0]
            steps_since = current_step - assignment.eviction_step
            if not flush_all and steps_since < self.credit_lookahead:
                break
            pending_credits.popleft()

            # Find log-probs after eviction
            # log_probs is indexed by decode step (1-based offset into the list)
            eviction_lp_idx = assignment.eviction_step  # log_probs[i] corresponds to step i+1
            after_start = eviction_lp_idx
            after_end = min(len(log_probs), after_start + self.credit_lookahead)
            if after_start >= len(log_probs):
                # Not enough post-eviction data — skip update to avoid
                # biasing estimator with zero-signal observations.
                continue

            quality_after = float(np.mean(log_probs[after_start:after_end]))

            # Guard against NaN/Inf from unstable model steps
            if not np.isfinite(quality_after) or not np.isfinite(assignment.quality_before):
                continue

            quality_delta = quality_after - assignment.quality_before
            quality_deltas.append(quality_delta)

            # Skip near-zero deltas that are just noise
            if abs(quality_delta) < self.noise_threshold:
                continue

            # Negative delta means quality dropped → positions were important → positive credit
            total_impact = -quality_delta

            # Attention-weighted proportional attribution:
            # Blocks that had higher attention before eviction are more likely
            # responsible for any quality change, so they get proportionally
            # more credit. This beats uniform attribution which dilutes signal.
            attn_weights = assignment.evicted_attention_weights
            weight_sum = sum(attn_weights) if attn_weights else 0.0
            n_evicted = len(assignment.evicted_features)

            for idx, features in enumerate(assignment.evicted_features):
                if weight_sum > 1e-9 and n_evicted > 1:
                    # Proportional: each block gets credit proportional to its attention.
                    # sum(proportions) = 1.0, so sum(block_impacts) = total_impact.
                    proportion = attn_weights[idx] / weight_sum
                    block_impact = total_impact * proportion
                else:
                    # All attention is zero or single block → split evenly
                    block_impact = total_impact / float(max(1, n_evicted))
                self.estimator.update(features, block_impact)
            # Commit batch-level stats (update_count, EMA, feature stats) once
            # per eviction event, not per-token.
            self.estimator.finish_batch()
            updates += 1

        return updates


def _log_prob_of_token(logits: np.ndarray, token_id: int) -> float:
    """Extract log-probability of a single token efficiently.

    Operates in float32 (not float64) to halve the temporary-array allocation
    for large vocabularies (Qwen vocab ≈ 152 K tokens → saves ~1.2 MB/step).
    Numerical stability preserved via max-subtraction before exp.
    """
    row = logits[0, -1]
    if not np.all(np.isfinite(row)):
        return 0.0
    row = row.astype(np.float32, copy=False)  # no-op if already float32
    max_val = float(row.max())
    # log-sum-exp in float32 — precision is adequate for log-prob tracking
    lse = max_val + float(np.log(np.sum(np.exp(row - max_val))))
    result = float(row[token_id]) - lse
    return result if np.isfinite(result) else 0.0
