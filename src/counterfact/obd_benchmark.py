"""The Open Bandit Dataset OPE benchmark (Saito et al., 2021, Section 5).

Both arms of ZOZOTOWN's A/B test ran at the same time on the same campaigns, so each
arm's mean click rate is a near-exact on-policy value (12M BTS rows, 1.4M random rows on
the "all" campaign). The benchmark estimates one arm's value from the other arm's logs
and scores each estimator by its relative error against that on-policy value:

    random_to_bts   logs from the uniform-random policy, evaluate Thompson sampling
    bts_to_random   logs from Thompson sampling, evaluate the uniform-random policy

Each run draws a bootstrap sample of the logged arm, cross-fits the reward model on it,
and applies every estimator, so the spread across runs includes reward-model noise.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from counterfact.data import LoggedBandit, load
from counterfact.estimators import Estimator, OPEInput
from counterfact.evaluation import summarize
from counterfact.policies import (
    LoggedSlatePolicy,
    at_logged_position,
    bernoulli_thompson_sampling,
    uniform_random,
    zozotown_prior,
)
from counterfact.reward_model import BanditFeatures, RewardModelConfig, cross_fit

DIRECTIONS = ("random_to_bts", "bts_to_random")
# How the BTS evaluation policy is specified in random_to_bts:
#   snapshot  obp's production-prior snapshot, simulated (the protocol the paper and obp use)
#   replayed  the time-varying policy rebuilt from the propensities BTS actually logged
TARGETS = ("snapshot", "replayed")


@dataclass(frozen=True)
class Task:
    campaign: str
    direction: str
    target: str = "snapshot"

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")

    @property
    def label(self) -> str:
        suffix = f" [{self.target}]" if self.direction == "random_to_bts" else ""
        return f"{self.campaign}: {self.direction}{suffix}"


@dataclass
class Prepared:
    task: Task
    logged: LoggedBandit
    truth: float
    truth_se: float
    policy: Callable[[LoggedBandit], np.ndarray]  # rows of `logged` -> (n, n_actions) pi_e


def bandit_features(log: LoggedBandit, item_ids: bool = False) -> BanditFeatures:
    """Reward-model inputs. `item_ids` appends a one-hot of the item id to its features,
    letting the model learn a click rate per item instead of only per feature value."""
    items = log.item_features
    if item_ids:
        items = np.hstack([items, np.eye(log.n_actions, dtype=np.float32)])
    return BanditFeatures(
        context=log.user_onehot(),
        item_features=items,
        action=log.action,
        position=log.position,
        reward=log.reward,
        n_positions=log.len_list,
    )


def prepare(task: Task, root: Path | None = None, n_sim: int = 1_000_000) -> Prepared:
    behavior, evaluation = ("random", "bts") if task.direction == "random_to_bts" else ("bts", "random")
    logged = load(behavior, task.campaign, root)
    target_log = load(evaluation, task.campaign, root)
    truth = target_log.on_policy_value()
    truth_se = float(np.sqrt(truth * (1 - truth) / target_log.n_rounds))

    if evaluation == "random":
        dist = uniform_random(logged.n_actions, logged.len_list)
        policy = lambda log: at_logged_position(dist, log.position)  # noqa: E731
    elif task.target == "snapshot":
        alpha, beta = zozotown_prior(task.campaign)
        dist = bernoulli_thompson_sampling(alpha, beta, logged.len_list, n_sim=n_sim)
        policy = lambda log: at_logged_position(dist, log.position)  # noqa: E731
    else:
        replay = LoggedSlatePolicy(
            target_log.timestamp, target_log.action, target_log.position, target_log.pscore, target_log.n_actions
        )
        policy = lambda log: replay.at(log.timestamp, log.position)[0]  # noqa: E731
    return Prepared(task=task, logged=logged, truth=truth, truth_se=truth_se, policy=policy)


def run(
    prepared: Prepared,
    estimators: Sequence[Estimator],
    reward_config: RewardModelConfig,
    n_runs: int = 20,
    sample_size: int | None = None,
    n_folds: int = 2,
    seed: int = 0,
    item_ids: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Bootstrap the logged arm `n_runs` times and score every estimator on each sample."""
    logged = prepared.logged
    size = min(sample_size or logged.n_rounds, logged.n_rounds)
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {e.name: [] for e in estimators}
    diagnostics: list[dict] = []
    max_weight: list[float] = []
    for r in range(n_runs):
        start = time.perf_counter()
        sample = logged.take(rng.integers(0, logged.n_rounds, size))
        q_hat, diag = cross_fit(bandit_features(sample, item_ids), reward_config, n_folds=n_folds, seed=seed + 1000 * r)
        data = OPEInput.build(
            action=sample.action, reward=sample.reward, pscore=sample.pscore, pi_e=prepared.policy(sample), q_hat=q_hat
        )
        for e in estimators:
            estimates[e.name].append(e.estimate(data))
        diagnostics.append(diag.as_dict())
        max_weight.append(float(data.weight.max()))
        log(f"  {prepared.task.label} run {r + 1}/{n_runs} ({time.perf_counter() - start:.1f}s) auc {diag.auc:.3f}")

    return {
        "task": asdict(prepared.task),
        "truth": prepared.truth,
        "truth_se": prepared.truth_se,
        "behavior_rounds": logged.n_rounds,
        "sample_size": size,
        "n_runs": n_runs,
        "n_folds": n_folds,
        "reward_model": {"label": reward_config.label, "item_ids": item_ids, **asdict(reward_config)},
        "max_weight_median": float(np.median(max_weight)),
        "reward_model_diagnostics": {k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]},
        "estimates": estimates,
        "summary": {name: summarize(v, prepared.truth) for name, v in estimates.items()},
    }
