"""Off-policy estimators of an evaluation policy's value from logged bandit data.

Notation for logged round i (x_i context, a_i action taken by the behaviour policy pi_b):

    r_i         observed reward
    w_i         importance weight pi_e(a_i | x_i) / pi_b(a_i | x_i)
    q_hat_i     reward-model prediction for the logged action, q_hat(x_i, a_i)
    q_pi_i      reward-model prediction averaged over the evaluation policy,
                sum_a pi_e(a | x_i) q_hat(x_i, a)

Every estimator below needs only these four vectors, so the (n, n_actions) policy and
reward-model matrices are reduced once in `OPEInput.build` and everything after that,
including bootstrap resampling, is O(n).

Each estimator returns per-round terms whose mean is the estimate. Self-normalised
estimators divide by the mean weight of the sample they are given, so they must be
recomputed (not averaged) on every resample, which `estimate` does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OPEInput:
    """Logged data reduced to what the estimators consume."""

    reward: np.ndarray
    weight: np.ndarray
    q_factual: np.ndarray | None = None
    q_policy: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.reward.shape[0]
        if self.reward.ndim != 1 or self.weight.shape != (n,):
            raise ValueError("reward and weight must be 1-D arrays of the same length")
        if np.any(self.weight < 0) or not np.all(np.isfinite(self.weight)):
            raise ValueError("importance weights must be finite and non-negative")
        if (self.q_factual is None) != (self.q_policy is None):
            raise ValueError("q_factual and q_policy must be given together")
        if self.q_factual is not None and (self.q_factual.shape != (n,) or self.q_policy.shape != (n,)):
            raise ValueError("q_factual and q_policy must have one entry per round")

    @classmethod
    def build(
        cls,
        *,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        pi_e: np.ndarray,
        q_hat: np.ndarray | None = None,
    ) -> OPEInput:
        """Reduce full policy and reward-model matrices to per-round vectors.

        pi_e and q_hat have shape (n, n_actions): row i holds every action's probability
        under the evaluation policy (and predicted reward) in round i's context and slot.
        """
        n = action.shape[0]
        if pi_e.shape[0] != n:
            raise ValueError(f"pi_e has {pi_e.shape[0]} rows for {n} rounds")
        if np.any(pscore <= 0):
            raise ValueError("behaviour propensities must be positive")
        rows = np.arange(n)
        weight = pi_e[rows, action].astype(np.float64) / pscore
        q_factual = q_policy = None
        if q_hat is not None:
            if q_hat.shape != pi_e.shape:
                raise ValueError(f"q_hat shape {q_hat.shape} does not match pi_e shape {pi_e.shape}")
            q_factual = q_hat[rows, action].astype(np.float64)
            q_policy = np.einsum("ij,ij->i", pi_e, q_hat, dtype=np.float64)
        return cls(reward=reward.astype(np.float64), weight=weight, q_factual=q_factual, q_policy=q_policy)

    @property
    def n(self) -> int:
        return self.reward.shape[0]

    @property
    def has_reward_model(self) -> bool:
        return self.q_factual is not None

    def take(self, idx: np.ndarray) -> OPEInput:
        """Rows `idx` of this input (used for bootstrap resampling)."""
        return OPEInput(
            reward=self.reward[idx],
            weight=self.weight[idx],
            q_factual=None if self.q_factual is None else self.q_factual[idx],
            q_policy=None if self.q_policy is None else self.q_policy[idx],
        )


class Estimator(ABC):
    """An off-policy estimator: per-round terms whose mean estimates V(pi_e)."""

    name: str
    needs_reward_model: bool = False

    @abstractmethod
    def round_terms(self, data: OPEInput) -> np.ndarray: ...

    def estimate(self, data: OPEInput) -> float:
        if self.needs_reward_model and not data.has_reward_model:
            raise ValueError(f"{self.name} needs reward-model predictions")
        return float(self.round_terms(data).mean())

    def __repr__(self) -> str:
        return self.name


def _clip(weight: np.ndarray, clip: float) -> np.ndarray:
    return weight if np.isinf(clip) else np.minimum(weight, clip)


def _clip_label(base: str, clip: float) -> str:
    return base if np.isinf(clip) else f"{base} (clip {clip:g})"


class DirectMethod(Estimator):
    """DM: average the reward model over the evaluation policy.

    V_DM = 1/n sum_i q_pi_i. Low variance, but biased by exactly the reward model's error
    averaged under pi_e; no amount of data fixes a misspecified model.
    """

    name = "DM"
    needs_reward_model = True

    def round_terms(self, data: OPEInput) -> np.ndarray:
        return data.q_policy


class IPW(Estimator):
    """Inverse propensity weighting (Horvitz-Thompson).

    V_IPW = 1/n sum_i w_i r_i. Unbiased when pi_b(a|x) > 0 wherever pi_e(a|x) > 0, with
    variance that grows with E[w^2], i.e. with how far pi_e is from pi_b. Clipping the
    weight at `clip` trades a downward bias for lower variance.
    """

    needs_reward_model = False

    def __init__(self, clip: float = np.inf) -> None:
        if not clip > 0:
            raise ValueError("clip must be positive")
        self.clip = clip
        self.name = _clip_label("IPW", clip)

    def round_terms(self, data: OPEInput) -> np.ndarray:
        return _clip(data.weight, self.clip) * data.reward


class SNIPW(Estimator):
    """Self-normalised IPW: sum_i w_i r_i / sum_i w_i.

    Biased for finite n (ratio estimator) but consistent, bounded by the reward range,
    and invariant to rescaling the weights, which matters when propensities are tiny.
    """

    name = "SNIPW"
    needs_reward_model = False

    def round_terms(self, data: OPEInput) -> np.ndarray:
        return data.weight * data.reward / data.weight.mean()


class DoublyRobust(Estimator):
    """DR: the reward model as a baseline, IPW to correct its residual on the logged action.

    V_DR = 1/n sum_i [ q_pi_i + w_i (r_i - q_hat_i) ].
    Unbiased if either the propensities or the reward model are correct, and its
    variance shrinks as the residual r - q_hat shrinks. Clipping the residual's weight
    gives the "pessimistic shrinkage" variant of Su et al. (2020).
    """

    needs_reward_model = True

    def __init__(self, clip: float = np.inf) -> None:
        if not clip > 0:
            raise ValueError("clip must be positive")
        self.clip = clip
        self.name = _clip_label("DR", clip)

    def round_terms(self, data: OPEInput) -> np.ndarray:
        return data.q_policy + _clip(data.weight, self.clip) * (data.reward - data.q_factual)


class SNDR(Estimator):
    """Self-normalised DR: the IPW correction term divided by the mean weight."""

    name = "SNDR"
    needs_reward_model = True

    def round_terms(self, data: OPEInput) -> np.ndarray:
        return data.q_policy + data.weight * (data.reward - data.q_factual) / data.weight.mean()


class SwitchDR(Estimator):
    """Switch-DR (Wang et al., 2017): use DR where w <= tau, fall back to DM elsewhere.

    Rounds with extreme weights are exactly where IPW's variance comes from, so they get
    the reward model alone. tau = inf is DR; tau = 0 is DM.
    """

    needs_reward_model = True

    def __init__(self, tau: float) -> None:
        if tau < 0:
            raise ValueError("tau must be non-negative")
        self.tau = tau
        self.name = f"Switch-DR (tau {tau:g})"

    def round_terms(self, data: OPEInput) -> np.ndarray:
        keep = data.weight <= self.tau
        return data.q_policy + keep * data.weight * (data.reward - data.q_factual)


class DRos(Estimator):
    """DR with optimistic shrinkage (Su et al., 2020).

    Replaces w with w_hat = lam * w / (w^2 + lam), the weight that minimises a bound on
    the estimator's MSE. lam -> inf recovers DR; lam = 0 is DM.
    """

    needs_reward_model = True

    def __init__(self, lam: float) -> None:
        if lam < 0:
            raise ValueError("lam must be non-negative")
        self.lam = lam
        self.name = f"DRos (lambda {lam:g})"

    def round_terms(self, data: OPEInput) -> np.ndarray:
        w = data.weight
        shrunk = w if np.isinf(self.lam) else self.lam * w / (w**2 + self.lam)
        return data.q_policy + shrunk * (data.reward - data.q_factual)


def default_estimators(switch_tau: float = 100.0, dros_lambda: float = 100.0) -> list[Estimator]:
    """The estimator set compared in every experiment."""
    return [DirectMethod(), IPW(), SNIPW(), DoublyRobust(), SNDR(), SwitchDR(switch_tau), DRos(dros_lambda)]
