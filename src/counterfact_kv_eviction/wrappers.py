from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .interfaces import KVBlockMetadata, KVPolicy, build_feature_vector


@dataclass
class UncertaintyGatedPolicy:
    """Use candidate policy when confidence is acceptable, else fallback policy."""

    candidate_policy: KVPolicy
    fallback_policy: KVPolicy
    uncertainty_threshold: float = 0.35
    min_samples: int = 3
    name: str = "uncertainty_gated"

    def select_evictions(
        self,
        blocks: Dict[str, KVBlockMetadata],
        evict_count: int,
        step: int,
    ) -> List[str]:
        if len(blocks) < self.min_samples:
            return self.fallback_policy.select_evictions(blocks, evict_count, step)

        avg_uncertainty = (
            sum(block.uncertainty for block in blocks.values()) / float(len(blocks))
        )
        if avg_uncertainty > self.uncertainty_threshold:
            return self.fallback_policy.select_evictions(blocks, evict_count, step)

        return self.candidate_policy.select_evictions(blocks, evict_count, step)


def annotate_blocks_with_estimates(blocks, estimator, step: int, capacity: int = 0) -> None:
    """Populate estimated_credit and uncertainty from an estimator."""

    for block in blocks.values():
        features = build_feature_vector(block, step, capacity=capacity)
        credit, uncertainty = estimator.predict_credit(features)
        block.estimated_credit = credit
        block.uncertainty = uncertainty
        block.has_estimate = True
