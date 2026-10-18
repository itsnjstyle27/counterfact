"""Scoring estimators: error summaries over repeated runs and bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from counterfact.estimators import Estimator, OPEInput


def relative_error(estimate: float | np.ndarray, truth: float) -> float | np.ndarray:
    """|V_hat - V| / V, the metric of the Open Bandit Dataset paper."""
    return np.abs(np.asarray(estimate) - truth) / truth


def summarize(estimates: Sequence[float], truth: float) -> dict[str, float]:
    """Bias, spread and error of one estimator's estimates across independent runs.

    MSE = bias^2 + variance exactly when the variance uses ddof=0, so that is what is
    reported, letting the plots stack the two terms.
    """
    e = np.asarray(estimates, dtype=np.float64)
    rel = relative_error(e, truth)
    bias = float(e.mean() - truth)
    var = float(e.var())
    return {
        "n_runs": int(e.size),
        "mean": float(e.mean()),
        "bias": bias,
        "variance": var,
        "mse": bias**2 + var,
        "rmse": float(np.sqrt(bias**2 + var)),
        "relative_error_mean": float(rel.mean()),
        "relative_error_std": float(rel.std(ddof=1)) if e.size > 1 else 0.0,
        "relative_error_median": float(np.median(rel)),
    }


def bootstrap_ci(
    estimator: Estimator, data: OPEInput, n_boot: int = 1000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling logged rounds with replacement.

    This holds the reward model fixed, so it reflects sampling noise in the logged data
    but not the reward model's own estimation noise.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = estimator.estimate(data.take(rng.integers(0, data.n, data.n)))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def evaluate_all(estimators: Sequence[Estimator], data: OPEInput) -> dict[str, float]:
    """Each estimator's point estimate, skipping those that need a reward model when there is none."""
    return {e.name: e.estimate(data) for e in estimators if data.has_reward_model or not e.needs_reward_model}
