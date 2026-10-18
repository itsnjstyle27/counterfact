"""Evaluation policies for the Open Bandit Dataset.

Both logging policies in the dataset are context-free and fill a slate of three
positions, so a policy is fully described by a (n_actions, len_list) matrix whose
column k is the distribution of the item shown at position k.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import yaml


def uniform_random(n_actions: int, len_list: int = 3) -> np.ndarray:
    """Uniformly random slate without replacement: every item is equally likely at every position."""
    return np.full((n_actions, len_list), 1.0 / n_actions)


def zozotown_prior(campaign: str) -> tuple[np.ndarray, np.ndarray]:
    """Beta(alpha, beta) parameters of ZOZOTOWN's production Thompson-sampling policy."""
    text = resources.files("counterfact.resources").joinpath("zozotown_bts_prior.yaml").read_text(encoding="utf-8")
    prior = yaml.safe_load(text)[campaign]
    return np.asarray(prior["alpha"], dtype=np.float64), np.asarray(prior["beta"], dtype=np.float64)


def bernoulli_thompson_sampling(
    alpha: np.ndarray,
    beta: np.ndarray,
    len_list: int = 3,
    n_sim: int = 1_000_000,
    seed: int = 0,
    chunk: int = 50_000,
) -> np.ndarray:
    """Position-wise action distribution of Bernoulli Thompson sampling, by Monte Carlo.

    Each impression draws theta_a ~ Beta(alpha_a, beta_a) for every item and shows the
    items with the `len_list` largest draws, best first. The probability that item a
    lands at position k has no closed form, so it is estimated from `n_sim` simulated
    slates; its standard error is at most 0.5 / sqrt(n_sim) (5e-4 at the default).
    """
    rng = np.random.default_rng(seed)
    n_actions = alpha.shape[0]
    counts = np.zeros((n_actions, len_list))
    done = 0
    while done < n_sim:
        m = min(chunk, n_sim - done)
        theta = rng.beta(alpha, beta, size=(m, n_actions))
        top = np.argpartition(-theta, len_list - 1, axis=1)[:, :len_list]
        ranked = np.take_along_axis(top, np.argsort(-np.take_along_axis(theta, top, axis=1), axis=1), axis=1)
        for k in range(len_list):
            counts[:, k] += np.bincount(ranked[:, k], minlength=n_actions)
        done += m
    return counts / n_sim


def at_logged_position(action_dist: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Expand a (n_actions, len_list) slate policy to the (n, n_actions) matrix the estimators use.

    Row i is the policy's distribution over items at round i's logged position.
    """
    return np.ascontiguousarray(action_dist[:, position].T)


class LoggedSlatePolicy:
    """A context-free slate policy that changes over time, rebuilt from the propensities it logged.

    ZOZOTOWN's Thompson-sampling policy re-estimated its Beta posteriors throughout the
    week, so the probability of showing item a at position k drifts. Every BTS log row
    records that probability for the (item, position) it showed, at the time it showed
    it. The policy at time t is rebuilt by taking, for every (item, position), the
    propensity logged closest to t at or before it (the next one if there is none
    before), then renormalising each position's distribution to sum to one.
    """

    def __init__(
        self,
        timestamp: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        n_actions: int,
        len_list: int = 3,
    ):
        self.n_actions = n_actions
        self.len_list = len_list
        key = action * len_list + position
        order = np.lexsort((timestamp, key))
        key, self._time, self._pscore = key[order], timestamp[order], pscore[order]
        n_keys = n_actions * len_list
        self._start = np.searchsorted(key, np.arange(n_keys), side="left")
        self._stop = np.searchsorted(key, np.arange(n_keys), side="right")
        missing = np.flatnonzero(self._stop == self._start)
        if missing.size:
            raise ValueError(f"{missing.size} (item, position) pairs never appear in the log")

    def raw_propensity(self, timestamp: np.ndarray, action: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Last logged propensity of each (action, position) at or before each timestamp."""
        out = np.empty(timestamp.shape[0], dtype=np.float64)
        key = action * self.len_list + position
        for k in np.unique(key):
            rows = np.flatnonzero(key == k)
            lo, hi = self._start[k], self._stop[k]
            idx = np.searchsorted(self._time[lo:hi], timestamp[rows], side="right") - 1
            out[rows] = self._pscore[lo:hi][np.clip(idx, 0, hi - lo - 1)]
        return out

    def at(self, timestamp: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, n_actions) distribution over items at each round's time and position.

        Also returns each row's sum before renormalisation, a direct measure of how
        consistent the rebuilt snapshot is (exactly 1 if every propensity were current).
        """
        n = timestamp.shape[0]
        dist = np.empty((n, self.n_actions), dtype=np.float64)
        for a in range(self.n_actions):
            dist[:, a] = self.raw_propensity(timestamp, np.full(n, a), position)
        raw_sum = dist.sum(axis=1)
        return dist / raw_sum[:, None], raw_sum
