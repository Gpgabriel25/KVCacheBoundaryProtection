from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple


try:
    import jax.numpy as jnp
    from flax import linen as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency in minimal test envs.
    jnp = None
    nn = None


if nn is not None:

    class CausalCreditMLP(nn.Module):
        """Tiny Flax MLP for causal utility estimates."""

        hidden_dim: int = 16

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            credit = nn.Dense(1)(x)
            log_var = nn.Dense(1)(x)
            return credit, log_var


@dataclass
class OnlineCausalCreditEstimator:
    """Lightweight online estimator with EMA-based uncertainty tracking.

    Key improvements over v1:
    - EMA residual tracking so uncertainty can INCREASE when predictions worsen
    - Decaying learning rate schedule (base_lr / sqrt(t+1))
    - Gradient clipping to prevent single-event destabilization
    """

    base_learning_rate: float = 0.08
    weights: List[float] = field(default_factory=lambda: [-0.5, -0.1, 0.3, 0.5])
    bias: float = 0.0
    update_count: int = 0
    warmup_min_updates: int = 5
    warmup_uncertainty: float = 1.0

    # EMA-based uncertainty (replaces cumulative residual_m2)
    ema_squared_error: float = 1.0
    ema_decay: float = 0.92
    max_grad: float = 1.0

    # Feature-specific uncertainty: track running stats to detect OOD inputs
    feature_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    feature_var: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    feature_ema_decay: float = 0.95
    ood_uncertainty_weight: float = 0.5

    # Backward compat: legacy field kept so old code that checks it doesn't break
    residual_m2: float = field(default=1e-6, repr=False)

    # Batch-level accumulators (call finish_batch() once per eviction event)
    _batch_squared_errors: List[float] = field(default_factory=list, repr=False)
    _batch_features: List[List[float]] = field(default_factory=list, repr=False)

    # v1 compat alias
    @property
    def learning_rate(self) -> float:
        return self._effective_lr()

    def _effective_lr(self) -> float:
        return self.base_learning_rate / ((self.update_count + 1) ** 0.5)

    def _validate_feature_dim(self, feats: Sequence[float]) -> None:
        expected_dim = len(self.weights)
        actual_dim = len(feats)
        if actual_dim != expected_dim:
            raise ValueError(
                "Feature dimension mismatch: "
                f"expected {expected_dim}, got {actual_dim}"
            )

    def predict_credit(self, features: Iterable[float]) -> Tuple[float, float]:
        feats = list(features)
        self._validate_feature_dim(feats)
        credit = self.bias + sum(w * f for w, f in zip(self.weights, feats))
        if self.update_count < self.warmup_min_updates:
            return credit, self.warmup_uncertainty
        base_uncertainty = self.ema_squared_error ** 0.5
        # Feature-specific OOD detection: novel feature combinations get higher uncertainty
        leverage = sum(
            ((f - mu) ** 2) / (var + 1e-4)
            for f, mu, var in zip(feats, self.feature_mean, self.feature_var)
        ) / max(1, len(feats))
        uncertainty = base_uncertainty * (1.0 + self.ood_uncertainty_weight * leverage)
        return credit, uncertainty

    def update(self, features: Iterable[float], observed_delta: float) -> None:
        """Update weights for a single observation.

        NOTE: This only updates weights/bias (the gradient step). Call
        ``finish_batch()`` once per eviction *event* (not per-token) to
        increment ``update_count``, decay EMA, and update feature stats.
        """
        feats = list(features)
        self._validate_feature_dim(feats)
        pred, _ = self.predict_credit(feats)
        err = observed_delta - pred

        # Clip gradient to prevent destabilization from outlier events
        clipped_err = max(-self.max_grad, min(self.max_grad, err))

        lr = self._effective_lr()
        for i, feat in enumerate(feats):
            self.weights[i] += lr * clipped_err * feat
        self.bias += lr * clipped_err

        # Accumulate per-token errors; batch-level EMA done in finish_batch()
        self._batch_squared_errors.append(clipped_err * clipped_err)
        self._batch_features.append(feats)
        # Keep legacy field in sync for any external inspection
        self.residual_m2 += err * err

    # ── Batch-level tracking (call once per eviction event) ──────────────

    def finish_batch(self) -> None:
        """Commit batch-level statistics after all per-token updates.

        Must be called exactly once per eviction event to keep update_count,
        EMA uncertainty, and feature statistics at the correct cadence.
        """
        if not self._batch_squared_errors:
            return

        self.update_count += 1

        # EMA of squared error: use mean of batch, not per-token
        mean_sq = sum(self._batch_squared_errors) / len(self._batch_squared_errors)
        self.ema_squared_error = (
            self.ema_decay * self.ema_squared_error
            + (1 - self.ema_decay) * mean_sq
        )

        # Track running feature statistics for OOD detection:
        # use mean features from the batch for a single EMA step
        n = len(self._batch_features)
        d = self.feature_ema_decay
        for i in range(len(self.weights)):
            feat_mean = sum(f[i] for f in self._batch_features) / n
            old_mean = self.feature_mean[i]
            self.feature_mean[i] = d * old_mean + (1 - d) * feat_mean
            diff = feat_mean - old_mean
            self.feature_var[i] = d * self.feature_var[i] + (1 - d) * diff * diff

        self._batch_squared_errors.clear()
        self._batch_features.clear()


def batch_predict(
    estimator: OnlineCausalCreditEstimator,
    feature_batch: Sequence[Iterable[float]],
) -> List[Tuple[float, float]]:
    return [estimator.predict_credit(features) for features in feature_batch]
