from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Tuple


try:
    import jax
except ImportError:  # pragma: no cover - optional dependency in minimal test envs.
    jax = None


@dataclass
class KVBlockMetadata:
    """State tracked for each KV block key."""

    key: str
    last_access_step: int
    insert_step: int
    access_count: int = 0
    attention_score: float = 0.0
    max_attention_score: float = 0.0
    cumulative_attention_score: float = 0.0
    prefill_attention_score: float = 0.0
    estimated_credit: float = 0.0
    uncertainty: float = 1.0
    has_estimate: bool = False


class KVPolicy(Protocol):
    """Policy interface for selecting keys to evict."""

    name: str

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        """Return keys to evict, ordered from first to last eviction."""


class CreditEstimator(Protocol):
    """Estimator protocol for causal utility of retaining a block."""

    def predict_credit(self, features: Iterable[float]) -> Tuple[float, float]:
        """Returns (estimated_credit, uncertainty)."""

    def update(self, features: Iterable[float], observed_delta: float) -> None:
        """Online update from observed quality delta after eviction events."""


def build_feature_vector(block: KVBlockMetadata, step: int, capacity: int = 0) -> List[float]:
    """Build a small feature vector for utility models.

    When capacity > 0, uses capacity-relative normalization (preferred).
    Otherwise falls back to step-relative normalization.

    Uses log-scale encoding for recency/age to prevent feature collapse:
    blocks 1x, 2x, and 4x capacity old remain distinguishable.
    Uses current attention_score (not lifetime max) as the attention feature
    to avoid stale early spikes dominating signal.
    """
    import math
    denom = float(max(1, capacity if capacity > 0 else step))
    recency_raw = float(max(0, step - block.last_access_step))
    age_raw = float(max(0, step - block.insert_step))
    # Log-scale: log1p(raw) / log1p(4*denom) caps at ~1.0 for 4x-stale blocks
    # while preserving discrimination beyond 1x capacity
    log_denom = math.log1p(4.0 * denom)
    recency_norm = min(1.0, math.log1p(recency_raw) / log_denom) if log_denom > 0 else 0.0
    age_norm = min(1.0, math.log1p(age_raw) / log_denom) if log_denom > 0 else 0.0
    access_norm = float(block.access_count) / float(block.access_count + 4)
    # Use current attention score (not lifetime max) to avoid stale early spikes
    attention_signal = min(1.0, float(block.attention_score))
    return [recency_norm, age_norm, access_norm, attention_signal]


def apply_access(
    block: KVBlockMetadata,
    step: int,
    attention_score: Optional[float] = None,
) -> None:
    """Update metadata after a cache hit/access."""

    block.last_access_step = step
    block.access_count += 1
    if attention_score is not None:
        block.attention_score = attention_score
        block.max_attention_score = max(float(block.max_attention_score), float(attention_score))
