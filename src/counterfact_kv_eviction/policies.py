from __future__ import annotations

from dataclasses import dataclass, field
from heapq import nlargest, nsmallest
from typing import Dict, List

import numpy as np

from .interfaces import KVBlockMetadata


def _smallest_items(
    items,
    k: int,
    *,
    key,
):
    k = max(0, k)
    if k <= 0:
        return []
    if not isinstance(items, list):
        items = list(items)
    if not items:
        return []
    if k >= len(items):
        return sorted(items, key=key)
    return nsmallest(k, items, key=key)


def _largest_items(
    items,
    k: int,
    *,
    key,
):
    k = max(0, k)
    if k <= 0:
        return []
    if not isinstance(items, list):
        items = list(items)
    if not items:
        return []
    if k >= len(items):
        return sorted(items, key=key, reverse=True)
    return nlargest(k, items, key=key)


def _top_k_smallest_keys(
    items: List[KVBlockMetadata],
    k: int,
    *,
    key,
) -> List[str]:
    """Return keys for k smallest items by key, using heap select for small k.

    Most decode steps evict only a few blocks, so this avoids O(n log n)
    full sorts on the hot path and reduces ranking overhead.
    """
    k = max(0, k)
    if k <= 0 or not items:
        return []
    ranked = _smallest_items(items, k, key=key)
    return [b.key for b in ranked[:k]]


def _attention_signal(block: KVBlockMetadata) -> float:
    return float(max(block.attention_score, block.max_attention_score))


def _build_retained_and_protected_mask(
    retention_mask: List[bool],
    n_positions: int,
    n_protected: int = 0,
    n_suffix_protected: int = 0,
    *,
    static_suffix_mode: bool = False,
    orig_prompt_len: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    retained = np.array(retention_mask[:n_positions], dtype=bool)
    protected = np.zeros(n_positions, dtype=bool)
    protected[:n_protected] = True
    if n_suffix_protected > 0:
        if static_suffix_mode and orig_prompt_len > 0:
            suffix_start = max(0, orig_prompt_len - n_suffix_protected)
            suffix_end = min(n_positions, orig_prompt_len)
            if suffix_end > suffix_start:
                protected[suffix_start:suffix_end] = True
        else:
            retained_positions = np.where(retained)[0]
            if len(retained_positions) > 0:
                protected[retained_positions[-n_suffix_protected:]] = True
    return retained, protected & retained


def _allocate_adakv_head_budgets(
    head_attn: np.ndarray,
    budget_per_head: int,
    *,
    native_per_head: bool,
) -> np.ndarray:
    n_kv_heads = head_attn.shape[0]
    if budget_per_head <= 0 or n_kv_heads <= 0:
        return np.zeros(n_kv_heads, dtype=int)
    if native_per_head:
        return np.full(n_kv_heads, budget_per_head, dtype=int)

    evictable_budget = budget_per_head * n_kv_heads
    row_sums = head_attn.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    probs = head_attn / row_sums

    log_probs = np.log(np.maximum(probs, 1e-12))
    entropy = -np.sum(probs * log_probs, axis=1)

    inv_entropy = 1.0 / np.maximum(entropy, 1e-6)
    budget_fracs = inv_entropy / inv_entropy.sum()
    budgets = np.round(budget_fracs * evictable_budget).astype(int)

    budgets = np.maximum(budgets, 1)
    while budgets.sum() > evictable_budget:
        largest_h = np.argmax(budgets)
        budgets[largest_h] -= 1
    while budgets.sum() < evictable_budget:
        smallest_h = np.argmin(budgets)
        budgets[smallest_h] += 1
    return budgets


def _select_adakv_per_head_retention(
    per_head_cumul_attn: np.ndarray,
    retention_mask: List[bool],
    capacity: int,
    n_protected: int = 0,
    n_suffix_protected: int = 0,
    *,
    static_suffix_mode: bool = False,
    orig_prompt_len: int = 0,
    native_per_head: bool,
) -> np.ndarray:
    n_kv_heads, n_positions = per_head_cumul_attn.shape
    result = np.zeros((n_kv_heads, n_positions), dtype=bool)

    retained, protected = _build_retained_and_protected_mask(
        retention_mask,
        n_positions,
        n_protected=n_protected,
        n_suffix_protected=n_suffix_protected,
        static_suffix_mode=static_suffix_mode,
        orig_prompt_len=orig_prompt_len,
    )
    for h in range(n_kv_heads):
        result[h] = protected

    budget_per_head = max(0, int(capacity) - int(protected.sum()))
    if budget_per_head <= 0:
        return result

    evictable_indices = np.where(retained & ~protected)[0]
    if len(evictable_indices) == 0:
        return result

    head_attn = per_head_cumul_attn[:, evictable_indices]
    budgets = _allocate_adakv_head_budgets(
        head_attn,
        budget_per_head,
        native_per_head=native_per_head,
    )

    for h in range(n_kv_heads):
        scores = per_head_cumul_attn[h, evictable_indices]
        b_h = min(int(budgets[h]), len(evictable_indices))
        if b_h >= len(evictable_indices):
            result[h, evictable_indices] = True
            continue
        top_k_local = np.argpartition(scores, -b_h)[-b_h:]
        result[h, evictable_indices[top_k_local]] = True

    return result


@dataclass
class LRUPolicy:
    name: str = "lru"

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        del step
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (b.last_access_step, b.insert_step, b.key),
        )


@dataclass
class HybridRecencyFrequencyPolicy:
    """Lower score means higher eviction priority."""

    recency_weight: float = 0.7
    frequency_weight: float = 0.3
    name: str = "hybrid"

    def score(self, block: KVBlockMetadata, step: int) -> float:
        recency = float(step - block.last_access_step)
        frequency = float(block.access_count)
        return self.recency_weight * recency - self.frequency_weight * frequency

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        k = max(0, evict_count)
        if k <= 0 or not blocks:
            return []
        items = list(blocks.values())
        if k >= len(items):
            ranked = sorted(
                items,
                key=lambda b: (self.score(b, step), b.last_access_step, b.key),
                reverse=True,
            )
        else:
            # For descending top-k, invert numeric components and use nsmallest.
            ranked = nsmallest(
                k,
                items,
                key=lambda b: (-self.score(b, step), -b.last_access_step, b.key),
            )
        return [b.key for b in ranked[:k]]


@dataclass
class AttentionHeuristicPolicy:
    """Evict low-attention and stale blocks first."""

    attention_weight: float = 0.6
    staleness_weight: float = 0.4
    name: str = "attention_heuristic"

    def score(self, block: KVBlockMetadata, step: int) -> float:
        staleness = float(step - block.last_access_step)
        return self.attention_weight * _attention_signal(block) - self.staleness_weight * staleness

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self.score(b, step), b.last_access_step, b.key),
        )


@dataclass
class CausalCreditsPolicy:
    """Evict blocks with the lowest estimated retention utility."""

    uncertainty_penalty: float = 0.0
    singleton_attention_threshold: float = 0.70
    singleton_attention_bonus: float = 0.25
    on_missing_estimates: str = "error"
    name: str = "causal_credits"

    def score(self, block: KVBlockMetadata) -> float:
        base = float(block.estimated_credit) - self.uncertainty_penalty * float(block.uncertainty)
        if (
            block.access_count <= 1
            and _attention_signal(block) >= self.singleton_attention_threshold
        ):
            base += self.singleton_attention_bonus
        return base

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalCreditsPolicy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates == "fallback_lru":
                return _top_k_smallest_keys(
                    list(blocks.values()),
                    evict_count,
                    key=lambda b: (b.last_access_step, b.insert_step, b.key),
                )
            raise ValueError(
                "Invalid on_missing_estimates strategy: "
                f"{self.on_missing_estimates!r}"
            )

        del step
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self.score(b), b.last_access_step, b.key),
        )


@dataclass
class CausalNoUncertaintyPenaltyPolicy(CausalCreditsPolicy):
    """Ablation: disable uncertainty penalty in causal credits scoring."""

    uncertainty_penalty: float = 0.0
    name: str = "causal_no_uncertainty_penalty"


@dataclass
class CausalHighUncertaintyPenaltyPolicy(CausalCreditsPolicy):
    """Ablation: aggressively penalize uncertainty in causal credits scoring."""

    uncertainty_penalty: float = 1.0
    name: str = "causal_high_uncertainty_penalty"


@dataclass
class CausalRecoveryAllPolicy(CausalCreditsPolicy):
    """Causal credits with stability guardrails for long-context recovery."""

    rare_key_retention_fraction: float = 0.10
    max_evictions_per_window: int = 64
    eviction_window_size: int = 64
    retrieval_anchor_attention_threshold: float = 0.75
    retrieval_anchor_bonus: float = 0.35
    name: str = "causal_recovery_all"

    # Internal state for eviction-window budgeting.
    _window_start_step: int = field(default=1, init=False, repr=False)
    _evictions_used_in_window: int = field(default=0, init=False, repr=False)

    def score(self, block: KVBlockMetadata) -> float:
        base = super().score(block)
        if _attention_signal(block) >= self.retrieval_anchor_attention_threshold:
            return base + self.retrieval_anchor_bonus
        return base

    def _window_allowance(self, step: int) -> int:
        if self.eviction_window_size <= 0:
            return max(0, self.max_evictions_per_window)

        if step < self._window_start_step:
            self._window_start_step = step
            self._evictions_used_in_window = 0
        elif (step - self._window_start_step) >= self.eviction_window_size:
            window_index = (step - 1) // self.eviction_window_size
            self._window_start_step = window_index * self.eviction_window_size + 1
            self._evictions_used_in_window = 0

        return max(0, self.max_evictions_per_window - self._evictions_used_in_window)

    def _protected_rare_keys(self, blocks: Dict[str, KVBlockMetadata]) -> set[str]:
        if self.rare_key_retention_fraction <= 0.0:
            return set()

        quota = int(len(blocks) * self.rare_key_retention_fraction)
        if quota == 0 and blocks:
            quota = 1

        rare_first = _smallest_items(
            blocks.values(),
            quota,
            key=lambda b: (b.access_count, b.last_access_step, b.insert_step, b.key),
        )
        return {b.key for b in rare_first[:quota]}

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        allowed = self._window_allowance(step)
        effective_count = min(max(0, evict_count), allowed)
        if effective_count <= 0:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing and self.on_missing_estimates not in {"error", "fallback_lru"}:
            raise ValueError(
                "Invalid on_missing_estimates strategy: "
                f"{self.on_missing_estimates!r}"
            )
        if missing and self.on_missing_estimates == "error":
            sample = ", ".join(sorted(missing)[:3])
            raise ValueError(
                "CausalRecoveryAllPolicy requires estimates for all blocks. "
                f"Missing estimate annotations for: {sample}"
            )

        protected = self._protected_rare_keys(blocks)
        candidates = [b for b in blocks.values() if b.key not in protected]
        if not candidates:
            candidates = list(blocks.values())

        if missing:
            selected = _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
        else:
            selected = _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (self.score(b), b.last_access_step, b.key),
            )
        self._evictions_used_in_window += len(selected)
        return selected


@dataclass
class CausalAnchorFloorPolicy(CausalCreditsPolicy):
    """Causal credits with hard anchor pool and recency floor protection."""

    anchor_pool_fraction: float = 0.10
    anchor_attention_threshold: float = 0.75
    recency_floor_steps: int = 32
    name: str = "causal_anchor_floor"

    def _anchor_keys(self, blocks: Dict[str, KVBlockMetadata]) -> set[str]:
        if self.anchor_pool_fraction <= 0.0 or not blocks:
            return set()

        quota = int(len(blocks) * self.anchor_pool_fraction)
        if quota == 0:
            quota = 1

        ranked = _smallest_items(
            [b for b in blocks.values() if _attention_signal(b) >= self.anchor_attention_threshold],
            quota,
            key=lambda b: (-_attention_signal(b), -b.access_count, b.last_access_step, b.key),
        )
        return {b.key for b in ranked[:quota]}

    def _recency_floor_keys(self, blocks: Dict[str, KVBlockMetadata], step: int) -> set[str]:
        if self.recency_floor_steps < 0:
            return set()
        return {
            b.key
            for b in blocks.values()
            if (step - b.last_access_step) <= self.recency_floor_steps
        }

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalAnchorFloorPolicy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates != "fallback_lru":
                raise ValueError(
                    "Invalid on_missing_estimates strategy: "
                    f"{self.on_missing_estimates!r}"
                )
            return _top_k_smallest_keys(
                list(blocks.values()),
                evict_count,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )

        protected = self._anchor_keys(blocks) | self._recency_floor_keys(blocks, step)
        candidates = [b for b in blocks.values() if b.key not in protected]
        if not candidates:
            candidates = list(blocks.values())

        if missing:
            return _top_k_smallest_keys(
                candidates,
                evict_count,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
        else:
            return _top_k_smallest_keys(
                candidates,
                evict_count,
                key=lambda b: (self.score(b), b.last_access_step, b.key),
            )


@dataclass
class CausalQueryBonusPolicy(CausalCreditsPolicy):
    """Causal credits with a bonus for likely query-matching active context blocks."""

    query_recent_window: int = 24
    query_attention_threshold: float = 0.55
    query_match_bonus: float = 0.30
    name: str = "causal_query_bonus"

    def _score(self, block: KVBlockMetadata, step: int) -> float:
        base = self.score(block)
        if (
            (step - block.last_access_step) <= self.query_recent_window
            and _attention_signal(block) >= self.query_attention_threshold
        ):
            return base + self.query_match_bonus
        return base

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalQueryBonusPolicy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates != "fallback_lru":
                raise ValueError(
                    "Invalid on_missing_estimates strategy: "
                    f"{self.on_missing_estimates!r}"
                )

        if missing:
            return _top_k_smallest_keys(
                list(blocks.values()),
                evict_count,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
        else:
            return _top_k_smallest_keys(
                list(blocks.values()),
                evict_count,
                key=lambda b: (self._score(b, step), b.last_access_step, b.key),
            )


@dataclass
class CausalTwoStagePolicy(CausalCreditsPolicy):
    """Two-stage policy: evict non-anchors first, then causal-rank anchors if needed."""

    anchor_pool_fraction: float = 0.10
    anchor_attention_threshold: float = 0.75
    name: str = "causal_two_stage"

    def _anchor_keys(self, blocks: Dict[str, KVBlockMetadata]) -> set[str]:
        if self.anchor_pool_fraction <= 0.0 or not blocks:
            return set()

        quota = int(len(blocks) * self.anchor_pool_fraction)
        if quota == 0:
            quota = 1

        ranked = _smallest_items(
            [b for b in blocks.values() if _attention_signal(b) >= self.anchor_attention_threshold],
            quota,
            key=lambda b: (-_attention_signal(b), -b.access_count, b.last_access_step, b.key),
        )
        return {b.key for b in ranked[:quota]}

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        del step
        if evict_count <= 0 or not blocks:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalTwoStagePolicy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates != "fallback_lru":
                raise ValueError(
                    "Invalid on_missing_estimates strategy: "
                    f"{self.on_missing_estimates!r}"
                )

        anchor_keys = self._anchor_keys(blocks)
        non_anchor = [b for b in blocks.values() if b.key not in anchor_keys]
        anchors = [b for b in blocks.values() if b.key in anchor_keys]

        needed = max(0, evict_count)
        if missing:
            non_anchor_ranked = _smallest_items(
                non_anchor,
                needed,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
            if len(non_anchor) >= needed:
                return [b.key for b in non_anchor_ranked[:needed]]
            anchor_ranked = _smallest_items(
                anchors,
                needed - len(non_anchor_ranked),
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
        else:
            non_anchor_ranked = _smallest_items(
                non_anchor,
                needed,
                key=lambda b: (self.score(b), b.last_access_step, b.key),
            )
            if len(non_anchor) >= needed:
                return [b.key for b in non_anchor_ranked[:needed]]
            anchor_ranked = _smallest_items(
                anchors,
                needed - len(non_anchor_ranked),
                key=lambda b: (self.score(b), b.last_access_step, b.key),
            )

        ordered = non_anchor_ranked + anchor_ranked
        return [b.key for b in ordered[:needed]]


@dataclass
class CausalNIAHGuardedPolicy(CausalCreditsPolicy):
    """Retrieval-preserving causal policy with bounded per-step eviction work.

    Designed for NIAH-style exact-match robustness: preserve a small pool of
    salient anchors plus rare/high-attention candidates, while capping
    evictions per step to avoid latency spikes from aggressive churn.
    """

    anchor_pool_fraction: float = 0.10
    needle_attention_threshold: float = 0.85
    needle_access_ceiling: int = 2
    rare_key_attention_threshold: float = 0.70
    rare_key_access_ceiling: int = 3
    rare_key_recent_window: int = 128
    max_evictions_per_step: int = 8
    name: str = "causal_niah_guarded"

    def _anchor_keys(self, blocks: Dict[str, KVBlockMetadata]) -> set[str]:
        if self.anchor_pool_fraction <= 0.0 or not blocks:
            return set()

        quota = int(len(blocks) * self.anchor_pool_fraction)
        if quota == 0:
            quota = 1

        ranked = _smallest_items(
            blocks.values(),
            quota,
            key=lambda b: (-_attention_signal(b), b.last_access_step, b.key),
        )
        return {b.key for b in ranked[:quota]}

    def _needle_candidate_keys(self, blocks: Dict[str, KVBlockMetadata]) -> set[str]:
        return {
            b.key
            for b in blocks.values()
            if _attention_signal(b) >= self.needle_attention_threshold
            and b.access_count <= self.needle_access_ceiling
        }

    def _rare_high_attention_keys(
        self,
        blocks: Dict[str, KVBlockMetadata],
        step: int,
    ) -> set[str]:
        return {
            b.key
            for b in blocks.values()
            if _attention_signal(b) >= self.rare_key_attention_threshold
            and b.access_count <= self.rare_key_access_ceiling
            and (step - b.last_access_step) <= self.rare_key_recent_window
        }

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalNIAHGuardedPolicy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates != "fallback_lru":
                raise ValueError(
                    "Invalid on_missing_estimates strategy: "
                    f"{self.on_missing_estimates!r}"
                )

        effective_count = min(max(0, evict_count), max(1, self.max_evictions_per_step))
        protected = (
            self._anchor_keys(blocks)
            | self._needle_candidate_keys(blocks)
            | self._rare_high_attention_keys(blocks, step)
        )
        candidates = [b for b in blocks.values() if b.key not in protected]
        if not candidates:
            candidates = list(blocks.values())

        if missing:
            return _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
        else:
            del step
            return _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (self.score(b), b.last_access_step, b.key),
            )


@dataclass
class CausalNIAHGuardedV2Policy(CausalNIAHGuardedPolicy):
    """NIAH redesign variant that preserves bridge-context candidates.

    Extends CausalNIAHGuardedPolicy with an additional protected set for
    medium-attention, recently used bridge keys. This is intended to improve
    NIAH exact-match robustness while keeping per-step eviction work bounded.
    """

    bridge_attention_threshold: float = 0.60
    bridge_access_ceiling: int = 5
    bridge_recent_window: int = 192
    name: str = "causal_niah_guarded_v2"

    def _bridge_context_keys(
        self,
        blocks: Dict[str, KVBlockMetadata],
        step: int,
    ) -> set[str]:
        return {
            b.key
            for b in blocks.values()
            if _attention_signal(b) >= self.bridge_attention_threshold
            and b.access_count <= self.bridge_access_ceiling
            and (step - b.last_access_step) <= self.bridge_recent_window
        }

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalNIAHGuardedV2Policy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates != "fallback_lru":
                raise ValueError(
                    "Invalid on_missing_estimates strategy: "
                    f"{self.on_missing_estimates!r}"
                )

        effective_count = min(max(0, evict_count), max(1, self.max_evictions_per_step))
        protected = (
            self._anchor_keys(blocks)
            | self._needle_candidate_keys(blocks)
            | self._rare_high_attention_keys(blocks, step)
            | self._bridge_context_keys(blocks, step)
        )
        candidates = [b for b in blocks.values() if b.key not in protected]
        if not candidates:
            candidates = list(blocks.values())

        if missing:
            return _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
        else:
            return _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (self.score(b), b.last_access_step, b.key),
            )


@dataclass
class CausalNIAHGuardedV3Policy(CausalNIAHGuardedV2Policy):
    """V3 redesign: preserve a bounded rare-key pool across longer horizons.

    Adds a global low-access retention pool (bounded by fraction) to reduce
    accidental eviction of sparse-but-critical keys in hard NIAH regimes while
    retaining O(n) candidate filtering and bounded top-k selection.
    """

    rare_pool_fraction: float = 0.15
    rare_pool_access_ceiling: int = 4
    name: str = "causal_niah_guarded_v3"

    def _rare_pool_keys(self, blocks: Dict[str, KVBlockMetadata]) -> set[str]:
        if self.rare_pool_fraction <= 0.0 or not blocks:
            return set()

        quota = int(len(blocks) * self.rare_pool_fraction)
        if quota == 0:
            quota = 1

        ranked = _smallest_items(
            [b for b in blocks.values() if b.access_count <= self.rare_pool_access_ceiling],
            quota,
            key=lambda b: (b.access_count, -_attention_signal(b), b.last_access_step, b.key),
        )
        return {b.key for b in ranked[:quota]}

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalNIAHGuardedV3Policy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates != "fallback_lru":
                raise ValueError(
                    "Invalid on_missing_estimates strategy: "
                    f"{self.on_missing_estimates!r}"
                )

        effective_count = min(max(0, evict_count), max(1, self.max_evictions_per_step))
        protected = (
            self._anchor_keys(blocks)
            | self._needle_candidate_keys(blocks)
            | self._rare_high_attention_keys(blocks, step)
            | self._bridge_context_keys(blocks, step)
            | self._rare_pool_keys(blocks)
        )
        candidates = [b for b in blocks.values() if b.key not in protected]
        if not candidates:
            candidates = list(blocks.values())

        if missing:
            return _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
        return _top_k_smallest_keys(
            candidates,
            effective_count,
            key=lambda b: (self.score(b), b.last_access_step, b.key),
        )


@dataclass
class CausalNIAHRetrievalBalancedPolicy(CausalNIAHGuardedV3Policy):
    """Retrieval-focused guarded policy for hard NIAH/retrieval slices.

    Extends V3 with a retrieval-support protected set and a tighter per-step
    eviction cap to improve sparse key retention while limiting p99 spikes.
    """

    retrieval_pool_fraction: float = 0.12
    retrieval_attention_threshold: float = 0.52
    retrieval_min_access_count: int = 1
    retrieval_access_ceiling: int = 7
    retrieval_recent_window: int = 224
    retrieval_recent_bonus_window: int = 96
    retrieval_recent_bonus_threshold: float = 0.45
    retrieval_recent_bonus: float = 0.30
    max_evictions_per_step: int = 6
    name: str = "causal_niah_retrieval_balanced"

    def _score(self, block: KVBlockMetadata, step: int) -> float:
        base = self.score(block)
        if (
            (step - block.last_access_step) <= self.retrieval_recent_bonus_window
            and _attention_signal(block) >= self.retrieval_recent_bonus_threshold
        ):
            return base + self.retrieval_recent_bonus
        return base

    def _retrieval_support_keys(
        self,
        blocks: Dict[str, KVBlockMetadata],
        step: int,
    ) -> set[str]:
        if self.retrieval_pool_fraction <= 0.0 or not blocks:
            return set()

        quota = int(len(blocks) * self.retrieval_pool_fraction)
        if quota == 0:
            quota = 1

        ranked = _smallest_items(
            [
                b
                for b in blocks.values()
                if _attention_signal(b) >= self.retrieval_attention_threshold
                and b.access_count >= self.retrieval_min_access_count
                and b.access_count <= self.retrieval_access_ceiling
                and (step - b.last_access_step) <= self.retrieval_recent_window
            ],
            quota,
            key=lambda b: (-_attention_signal(b), b.access_count, b.last_access_step, b.key),
        )
        return {b.key for b in ranked[:quota]}

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalNIAHRetrievalBalancedPolicy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates != "fallback_lru":
                raise ValueError(
                    "Invalid on_missing_estimates strategy: "
                    f"{self.on_missing_estimates!r}"
                )

        effective_count = min(max(0, evict_count), max(1, self.max_evictions_per_step))
        protected = (
            self._anchor_keys(blocks)
            | self._needle_candidate_keys(blocks)
            | self._rare_high_attention_keys(blocks, step)
            | self._bridge_context_keys(blocks, step)
            | self._retrieval_support_keys(blocks, step)
        )
        candidates = [b for b in blocks.values() if b.key not in protected]
        if not candidates:
            candidates = list(blocks.values())

        if missing:
            return _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (b.last_access_step, b.insert_step, b.key),
            )
        return _top_k_smallest_keys(
            candidates,
            effective_count,
            key=lambda b: (self._score(b, step), b.last_access_step, b.key),
        )


@dataclass
class CausalConsensusGuardedPolicy(CausalNIAHRetrievalBalancedPolicy):
    """Consensus-guarded retrieval policy for rapid implementation sweeps.

    Blends causal keep utility with recency and heavy-hitter signals so
    eviction ranking is less brittle than pure causal scoring, while still
    retaining retrieval/anchor guardrails from the retrieval-balanced policy.
    Under high uncertainty, the policy automatically shifts weight toward
    recency to reduce risky evictions of currently active context.
    """

    causal_keep_weight: float = 0.55
    recency_keep_weight: float = 0.30
    heavy_hitter_keep_weight: float = 0.15
    high_uncertainty_threshold: float = 0.65
    uncertainty_recency_boost: float = 0.35
    recent_window: int = 48
    name: str = "causal_consensus_guarded"

    def _normalized_causal_keep(self, block: KVBlockMetadata, step: int) -> float:
        keep = float(self._score(block, step))
        if keep > 1.0:
            return 1.0
        if keep < -1.0:
            return -1.0
        return keep

    def _recency_keep(self, block: KVBlockMetadata, step: int) -> float:
        staleness = max(1.0, float(step - block.last_access_step + 1))
        return 1.0 / staleness

    def _heavy_hitter_keep(self, block: KVBlockMetadata, step: int) -> float:
        access_keep = float(block.access_count) / float(block.access_count + 4)
        attn_keep = 0.5 * _attention_signal(block)
        recent_keep = 0.25 if (step - block.last_access_step) <= self.recent_window else 0.0
        return access_keep + attn_keep + recent_keep

    def _weights(self, blocks: Dict[str, KVBlockMetadata]) -> tuple[float, float, float]:
        if not blocks:
            return self.causal_keep_weight, self.recency_keep_weight, self.heavy_hitter_keep_weight

        avg_uncertainty = sum(float(b.uncertainty) for b in blocks.values()) / float(len(blocks))
        if avg_uncertainty <= self.high_uncertainty_threshold:
            return self.causal_keep_weight, self.recency_keep_weight, self.heavy_hitter_keep_weight

        recency = self.recency_keep_weight + self.uncertainty_recency_boost
        causal = max(0.10, self.causal_keep_weight - self.uncertainty_recency_boost)
        heavy = self.heavy_hitter_keep_weight
        total = max(1e-8, causal + recency + heavy)
        return causal / total, recency / total, heavy / total

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        missing = [block.key for block in blocks.values() if not block.has_estimate]
        effective_count = min(max(0, evict_count), max(1, self.max_evictions_per_step))
        protected = (
            self._anchor_keys(blocks)
            | self._needle_candidate_keys(blocks)
            | self._rare_high_attention_keys(blocks, step)
            | self._bridge_context_keys(blocks, step)
            | self._retrieval_support_keys(blocks, step)
            | self._rare_pool_keys(blocks)
        )
        candidates = [b for b in blocks.values() if b.key not in protected]
        if not candidates:
            candidates = list(blocks.values())

        if missing:
            if self.on_missing_estimates == "error":
                sample = ", ".join(sorted(missing)[:3])
                raise ValueError(
                    "CausalConsensusGuardedPolicy requires estimates for all blocks. "
                    f"Missing estimate annotations for: {sample}"
                )
            if self.on_missing_estimates != "fallback_lru":
                raise ValueError(
                    "Invalid on_missing_estimates strategy: "
                    f"{self.on_missing_estimates!r}"
                )
        if missing:
            # Non-causal fallback: use recency + heavy_hitter + attention
            # protection guardrails instead of pure LRU.  This preserves the
            # advantage of attention-aware eviction even without causal
            # credit estimates.
            recency_w_fb = self.recency_keep_weight / max(
                1e-8, self.recency_keep_weight + self.heavy_hitter_keep_weight
            )
            heavy_w_fb = 1.0 - recency_w_fb

            def noncausal_keep(block: KVBlockMetadata) -> float:
                return (
                    recency_w_fb * self._recency_keep(block, step)
                    + heavy_w_fb * self._heavy_hitter_keep(block, step)
                )

            return _top_k_smallest_keys(
                candidates,
                effective_count,
                key=lambda b: (noncausal_keep(b), b.last_access_step, b.key),
            )

        causal_w, recency_w, heavy_w = self._weights(blocks)

        def keep_score(block: KVBlockMetadata) -> float:
            return (
                causal_w * self._normalized_causal_keep(block, step)
                + recency_w * self._recency_keep(block, step)
                + heavy_w * self._heavy_hitter_keep(block, step)
            )

        return _top_k_smallest_keys(
            candidates,
            effective_count,
            key=lambda b: (keep_score(b), b.last_access_step, b.key),
        )


@dataclass
class CausalLRUGuardedLightPolicy(CausalCreditsPolicy):
    """Near-LRU policy with minimal high-signal needle protection.

    Keeps behavior close to LRU for retrieval stability while reserving a small
    pool of sparse high-attention blocks to reduce accidental needle loss.
    """

    anchor_pool_fraction: float = 0.05
    anchor_attention_threshold: float = 0.82
    sparse_high_attention_threshold: float = 0.78
    sparse_access_ceiling: int = 2
    sparse_pool_quota: int = 2
    name: str = "causal_lru_guarded_light"

    def _anchor_keys(self, blocks: Dict[str, KVBlockMetadata]) -> set[str]:
        if self.anchor_pool_fraction <= 0.0 or not blocks:
            return set()
        quota = int(len(blocks) * self.anchor_pool_fraction)
        if quota == 0:
            quota = 1
        ranked = _smallest_items(
            [b for b in blocks.values() if _attention_signal(b) >= self.anchor_attention_threshold],
            quota,
            key=lambda b: (-_attention_signal(b), b.last_access_step, b.key),
        )
        return {b.key for b in ranked[:quota]}

    def _sparse_keys(self, blocks: Dict[str, KVBlockMetadata]) -> set[str]:
        quota = max(0, self.sparse_pool_quota)
        ranked = _smallest_items(
            [
                b
                for b in blocks.values()
                if _attention_signal(b) >= self.sparse_high_attention_threshold
                and b.access_count <= self.sparse_access_ceiling
            ],
            quota,
            key=lambda b: (-_attention_signal(b), b.access_count, b.last_access_step, b.key),
        )
        return {b.key for b in ranked[:quota]}

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        del step
        if evict_count <= 0 or not blocks:
            return []

        protected = self._anchor_keys(blocks) | self._sparse_keys(blocks)
        candidates = [b for b in blocks.values() if b.key not in protected]
        if not candidates:
            candidates = list(blocks.values())

        return _top_k_smallest_keys(
            candidates,
            evict_count,
            key=lambda b: (b.last_access_step, b.insert_step, b.key),
        )


@dataclass
class HeavyHitterH2OLikePolicy:
    """Prior-art-inspired baseline: keep frequent and recently used blocks."""

    heavy_hitter_weight: float = 0.7
    attention_weight: float = 0.1
    recent_window: int = 64
    recent_bonus: float = 1.0
    name: str = "h2o_like"

    def keep_score(self, block: KVBlockMetadata, step: int) -> float:
        is_recent = 1.0 if (step - block.last_access_step) <= self.recent_window else 0.0
        return (
            self.heavy_hitter_weight * float(block.cumulative_attention_score)
            + self.attention_weight * _attention_signal(block)
            + self.recent_bonus * is_recent
        )

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self.keep_score(b, step), b.last_access_step, b.insert_step, b.key),
        )


@dataclass
class StreamingSinkLikePolicy:
    """Prior-art-inspired baseline with sink-token and recency retention."""

    sink_size: int = 4
    recent_window_size: int = 8
    name: str = "streaming_sink_like"

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if evict_count <= 0 or not blocks:
            return []

        all_blocks = list(blocks.values())
        by_insert = _smallest_items(
            all_blocks,
            max(0, self.sink_size),
            key=lambda b: (b.insert_step, b.key),
        )
        by_recent = _largest_items(
            all_blocks,
            max(0, self.recent_window_size),
            key=lambda b: (b.last_access_step, b.insert_step, b.key),
        )

        protected = {
            b.key for b in by_insert[: max(0, self.sink_size)]
        } | {
            b.key for b in by_recent[: max(0, self.recent_window_size)]
        }

        candidates = [b for b in all_blocks if b.key not in protected]
        if not candidates:
            candidates = all_blocks

        return _top_k_smallest_keys(
            candidates,
            evict_count,
            key=lambda b: (b.last_access_step, b.insert_step, b.key),
        )


@dataclass
class SnapKVPolicy:
    """SnapKV-inspired baseline: retain positions with highest cumulative
    attention scores, following the observation-window approach of
    Ge et al. (2024).

    In our KV-coupled framework the attention metadata (max_attention_score,
    access_count) is maintained per-position during decode.  SnapKV keeps
    the positions with the highest "importance" measured by accumulated
    attention mass.  We approximate this as the product of access count
    and peak attention weight, which captures both frequency and magnitude
    of attention a position receives.
    """

    name: str = "snapkv"

    def _keep_score(self, block: KVBlockMetadata) -> float:
        """Higher score → more important → less likely evicted."""
        attn = _attention_signal(block)
        return float(block.cumulative_attention_score) + attn

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        del step
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self._keep_score(b), b.last_access_step, b.key),
        )


@dataclass
class SnapKVFaithfulPolicy:
    """Faithful SnapKV baseline: retain positions with highest prefill-time
    attention scores, following the observation-window approach of
    Ge et al. (2024).

    Unlike the simplified SnapKV (which continuously accumulates attention
    during decode), this variant uses only the attention scores computed
    during prefill and does not update them during generation. This
    faithfully captures the original SnapKV's core design: score positions
    once during prefill, then freeze the importance ranking.

    Positions inserted during decode (generated tokens) have
    prefill_attention_score=0 and are ranked by recency as a tiebreaker.
    """

    name: str = "snapkv_faithful"

    def _keep_score(self, block: KVBlockMetadata) -> float:
        """Higher score → more important → less likely evicted."""
        return float(block.prefill_attention_score)

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        del step
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self._keep_score(b), b.last_access_step, b.key),
        )


@dataclass
class H2OFaithfulPolicy:
    """More faithful H2O baseline: retain positions with highest cumulative
    attention mass, following Zhang et al. (2023).

    Unlike the simplified H2O (which blends cumulative attention with
    max_attention and a recency bonus), this variant uses only cumulative
    attention mass as the retention score, closer to the original H2O
    design.  The original H2O uses per-head, per-layer budgets; we still
    use a single global ranking (a remaining simplification), but the
    scoring function itself is faithful: pure cumulative attention mass.
    """

    name: str = "h2o_faithful"

    def keep_score(self, block: KVBlockMetadata, step: int) -> float:
        del step
        return float(block.cumulative_attention_score)

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self.keep_score(b, step), b.last_access_step, b.insert_step, b.key),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Recent baselines (2024+)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AdaKVPolicy:
    """Ada-KV-inspired baseline (Feng et al. 2024), simplified to global ranking.

    Ada-KV allocates per-head KV cache budgets based on each head's attention
    distribution entropy: heads with concentrated distributions keep more tokens.
    In our position-level (not head-level) framework we cannot perform per-head
    budget allocation.  Instead we approximate the core principle: positions that
    consistently attract concentrated attention are more important.

    Scoring:
        keep_score = cumulative_attention + alpha * peak_intensity
    where peak_intensity = max_attention_score captures how strongly any single
    decode step attended to this position (analogous to being in a concentrated
    head's top-k).  Alpha controls the weight of concentration vs. mass.
    """

    alpha: float = 0.5
    name: str = "adakv"

    def _keep_score(self, block: KVBlockMetadata) -> float:
        """Higher → more important → less likely evicted."""
        return float(block.cumulative_attention_score) + self.alpha * float(
            block.max_attention_score
        )

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        del step
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self._keep_score(b), b.last_access_step, b.key),
        )


@dataclass
class QUESTPolicy:
    """QUEST-inspired baseline (Tang et al. 2024), simplified.

    QUEST keeps cache entries whose key vectors are most relevant to the
    current query, determined by query-key dot products at decode time.
    In our framework, the per-position ``attention_score`` field already
    captures the most recent decode step's softmax attention weight (which
    is a monotone function of the query-key dot product).  We use this as
    the retention score: evict positions the model currently attends to
    least.

    This is fundamentally query-aware: the eviction ordering changes every
    step based on what the model is currently generating, unlike H2O/SnapKV
    which accumulate historical attention statistics.
    """

    name: str = "quest"

    def _keep_score(self, block: KVBlockMetadata) -> float:
        """Higher → more important → less likely evicted."""
        return float(block.attention_score)

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        del step
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self._keep_score(b), b.last_access_step, b.key),
        )


@dataclass
class QUESTFaithfulPolicy:
    """Faithful QUEST baseline using per-head current-query attention.

    QUEST ranks keys against the current decode query. In the JAX generator we
    approximate that by using the most recent per-head attention scores emitted
    by the model and giving each KV head its own top-k retention set. The
    model-level cache budget semantics stay the same as other faithful per-head
    variants: each head may retain up to `capacity` positions, with optional
    prefix/suffix protection applied independently to every head.
    """

    name: str = "quest_faithful"
    static_suffix_mode: bool = False  # If True, pin suffix to original prompt boundary

    def _keep_score(self, block: KVBlockMetadata) -> float:
        return float(block.attention_score)

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        """Fallback global ranking for non per-head callers."""
        del step
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (self._keep_score(b), b.last_access_step, b.key),
        )

    def select_per_head_retention(
        self,
        per_head_attn: np.ndarray,
        retention_mask: List[bool],
        capacity: int,
        n_protected: int = 0,
        n_suffix_protected: int = 0,
        orig_prompt_len: int = 0,
    ) -> np.ndarray:
        """Return per-head retention mask from the latest per-head attention.

        Suffix protection modes:
        - Dynamic (default): protects highest-numbered retained positions (may migrate during generation)
        - Static: protects fixed positions in range [orig_prompt_len - n_suffix_protected, orig_prompt_len)
        """
        n_kv_heads, n_positions = per_head_attn.shape
        result = np.zeros((n_kv_heads, n_positions), dtype=bool)

        retained = np.array(retention_mask[:n_positions], dtype=bool)
        protected = np.zeros(n_positions, dtype=bool)
        protected[:n_protected] = True
        if n_suffix_protected > 0:
            if self.static_suffix_mode and orig_prompt_len > 0:
                # Static suffix: pin to original prompt boundary
                suffix_start = max(0, orig_prompt_len - n_suffix_protected)
                suffix_end = min(n_positions, orig_prompt_len)
                if suffix_end > suffix_start:
                    protected[suffix_start:suffix_end] = True
            else:
                # Dynamic suffix: highest-numbered retained positions (original behavior)
                retained_positions = np.where(retained)[0]
                if len(retained_positions) > 0:
                    for p in retained_positions[-n_suffix_protected:]:
                        protected[p] = True

        for h in range(n_kv_heads):
            result[h] = protected & retained

        budget_per_head = max(0, int(capacity) - int(protected.sum()))
        if budget_per_head <= 0:
            return result

        evictable_indices = np.where(retained & ~protected)[0]
        if len(evictable_indices) == 0:
            return result

        for h in range(n_kv_heads):
            scores = per_head_attn[h, evictable_indices]
            keep_count = min(budget_per_head, len(evictable_indices))
            if keep_count >= len(evictable_indices):
                result[h, evictable_indices] = True
                continue
            top_k_local = np.argpartition(scores, -keep_count)[-keep_count:]
            result[h, evictable_indices[top_k_local]] = True

        return result


@dataclass
class QUESTNativePolicy(QUESTFaithfulPolicy):
    """Explicit native per-head QUEST variant for side-by-side experiments.

    The faithful QUEST selector already applies a full per-head
    `capacity - protected_count` budget, so this class is a named alias with
    identical retention semantics.
    """

    name: str = "quest_native"


@dataclass
class RandomEvictionPolicy:
    """Random eviction baseline: evict uniformly at random.

    Serves as the ultimate control: if random eviction + protection matches
    sophisticated policies + protection, it demonstrates that the eviction
    scoring criterion is irrelevant under structural protection.
    """

    seed: int = 42
    name: str = "random"

    def __post_init__(self) -> None:
        import random as _random

        self._rng = _random.Random(self.seed)

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        del step
        k = min(max(0, evict_count), len(blocks))
        if k <= 0:
            return []
        keys = list(blocks.keys())
        return self._rng.sample(keys, k)


@dataclass
class AdaKVFaithfulPolicy:
    """Faithful Ada-KV with per-head budget allocation (Feng et al. 2024).

    Each KV head gets a budget proportional to its attention concentration.
    Each head independently retains its top-B_h positions.
    """

    name: str = "adakv_faithful"
    static_suffix_mode: bool = False  # If True, pin suffix to original prompt boundary

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        """Fallback: global ranking using cumulative attention (unused in per-head mode)."""
        del step
        return _top_k_smallest_keys(
            list(blocks.values()),
            evict_count,
            key=lambda b: (float(b.cumulative_attention_score), b.last_access_step, b.key),
        )

    def select_per_head_retention(
        self,
        per_head_cumul_attn: np.ndarray,
        retention_mask: List[bool],
        capacity: int,
        n_protected: int = 0,
        n_suffix_protected: int = 0,
        orig_prompt_len: int = 0,
    ) -> np.ndarray:
        """Return per-head retention mask: (n_kv_heads, n_positions) bool array.

        Algorithm:
        1. Compute per-head entropy from cumulative attention
        2. Allocate budgets: B_h proportional to 1/entropy_h (concentrated heads get more)
        3. Each head keeps its top-B_h by cumul_attn among non-protected retained positions
        4. Protected positions (prefix and suffix) are always retained in all heads

        Suffix protection modes:
        - Dynamic (default): protects highest-numbered retained positions (may migrate during generation)
        - Static: protects fixed positions in range [orig_prompt_len - n_suffix_protected, orig_prompt_len)
        """
        return _select_adakv_per_head_retention(
            per_head_cumul_attn,
            retention_mask,
            capacity,
            n_protected=n_protected,
            n_suffix_protected=n_suffix_protected,
            static_suffix_mode=self.static_suffix_mode,
            orig_prompt_len=orig_prompt_len,
            native_per_head=False,
        )


@dataclass
class AdaKVNativePolicy(AdaKVFaithfulPolicy):
    """Closer-to-native per-head Ada-KV comparison variant.

    Uses the same cumulative-attention signal and shared protection as the
    faithful variant, but gives every head the full unprotected per-head budget
    instead of redistributing a shared cross-head budget by entropy.
    """

    name: str = "adakv_native"

    def select_per_head_retention(
        self,
        per_head_cumul_attn: np.ndarray,
        retention_mask: List[bool],
        capacity: int,
        n_protected: int = 0,
        n_suffix_protected: int = 0,
        orig_prompt_len: int = 0,
    ) -> np.ndarray:
        return _select_adakv_per_head_retention(
            per_head_cumul_attn,
            retention_mask,
            capacity,
            n_protected=n_protected,
            n_suffix_protected=n_suffix_protected,
            static_suffix_mode=self.static_suffix_mode,
            orig_prompt_len=orig_prompt_len,
            native_per_head=True,
        )
