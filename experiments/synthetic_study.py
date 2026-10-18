"""Controlled experiments where the true policy value is known exactly.

    python experiments/synthetic_study.py --study sample_size   # error vs n, two reward models
    python experiments/synthetic_study.py --study overlap       # error vs how far pi_e is from pi_b
    python experiments/synthetic_study.py --study clipping      # bias-variance trade-off of weight clipping

Every run draws a fresh logged dataset, cross-fits the reward models on it, and applies
the estimators. Logged data does not depend on the evaluation policy, so the overlap and
clipping studies fit the reward model once per run and reuse it across settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counterfact.estimators import IPW, DirectMethod, DoublyRobust, DRos, OPEInput, SwitchDR, default_estimators
from counterfact.evaluation import summarize
from counterfact.reward_model import RewardModelConfig, cross_fit
from counterfact.synthetic import SyntheticBandit

MODELS = {"logistic": RewardModelConfig(hidden=()), "mlp": RewardModelConfig(hidden=(64, 64))}


def _collect(rows: dict, key: tuple, name: str, value: float) -> None:
    rows.setdefault(key, {}).setdefault(name, []).append(value)


def _summaries(rows: dict, truths: dict) -> list[dict]:
    out = []
    for key, by_estimator in rows.items():
        for name, values in by_estimator.items():
            out.append({"setting": list(key), "estimator": name, **summarize(values, truths[key[0]])})
    return out


def sample_size_study(env: SyntheticBandit, sizes: list[int], n_runs: int, seed: int) -> dict:
    truth = env.policy_value()
    rows: dict = {}
    rng = np.random.default_rng(seed)
    for n in sizes:
        start = time.perf_counter()
        for r in range(n_runs):
            log = env.sample(n, rng)
            no_model = log.ope_input()
            for e in default_estimators():
                if not e.needs_reward_model:
                    _collect(rows, (truth, n, "none"), e.name, e.estimate(no_model))
            for label, config in MODELS.items():
                q_hat, _ = cross_fit(log.features(), config, n_folds=2, seed=seed + r)
                data = log.ope_input(q_hat)
                for e in default_estimators():
                    if e.needs_reward_model:
                        _collect(rows, (truth, n, label), e.name, e.estimate(data))
        print(f"n={n}: {n_runs} runs in {time.perf_counter() - start:.0f}s", flush=True)
    return {"truth": truth, "results": _summaries(rows, {truth: truth})}


def overlap_study(env: SyntheticBandit, epsilons: list[float], n: int, n_runs: int, seed: int) -> dict:
    envs = {eps: env.with_epsilon(eps) for eps in epsilons}
    truths = {eps: e.policy_value() for eps, e in envs.items()}
    rows: dict = {}
    rng = np.random.default_rng(seed)
    for r in range(n_runs):
        log = env.sample(n, rng)
        q_hat, _ = cross_fit(log.features(), MODELS["mlp"], n_folds=2, seed=seed + r)
        for eps, e_env in envs.items():
            data = OPEInput.build(
                action=log.action,
                reward=log.reward,
                pscore=log.pscore,
                pi_e=e_env.evaluation_policy(log.context),
                q_hat=q_hat,
            )
            for est in default_estimators():
                _collect(rows, (eps,), est.name, est.estimate(data))
            _collect(rows, (eps,), "_max_weight", float(data.weight.max()))
        if (r + 1) % 10 == 0:
            print(f"overlap: {r + 1}/{n_runs} runs", flush=True)
    results = []
    for (eps,), by_estimator in rows.items():
        for name, values in by_estimator.items():
            if name == "_max_weight":
                results.append({"setting": [eps], "estimator": name, "median": float(np.median(values))})
            else:
                results.append({"setting": [eps], "estimator": name, **summarize(values, truths[eps])})
    return {"truths": {str(k): v for k, v in truths.items()}, "n": n, "results": results}


def clipping_study(env: SyntheticBandit, grid: list[float], n: int, n_runs: int, seed: int) -> dict:
    truth = env.policy_value()
    rows: dict = {}
    rng = np.random.default_rng(seed)
    for r in range(n_runs):
        log = env.sample(n, rng)
        q_hat, _ = cross_fit(log.features(), MODELS["mlp"], n_folds=2, seed=seed + r)
        data = log.ope_input(q_hat)
        _collect(rows, ("DM", 0.0), "DM", DirectMethod().estimate(data))
        for g in grid:
            _collect(rows, ("IPW", g), "IPW", IPW(clip=g).estimate(data))
            _collect(rows, ("DR", g), "DR", DoublyRobust(clip=g).estimate(data))
            _collect(rows, ("Switch-DR", g), "Switch-DR", SwitchDR(tau=g).estimate(data))
            _collect(rows, ("DRos", g), "DRos", DRos(lam=g).estimate(data))
        if (r + 1) % 10 == 0:
            print(f"clipping: {r + 1}/{n_runs} runs", flush=True)
    results = [
        {"setting": [family, "inf" if np.isinf(g) else g], "estimator": family, **summarize(v[family], truth)}
        for (family, g), v in rows.items()
    ]
    return {"truth": truth, "n": n, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True, choices=["sample_size", "overlap", "clipping"])
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (small models gain little from more)")
    parser.add_argument("--quick", action="store_true", help="tiny settings to smoke-test the pipeline")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    env = SyntheticBandit(seed=0, epsilon=0.3)
    config = {
        "environment": {k: v for k, v in vars(env).items() if not isinstance(v, np.ndarray)},
        "n_runs": args.n_runs,
    }
    start = time.perf_counter()
    if args.study == "sample_size":
        sizes = [250, 1000] if args.quick else [250, 500, 1000, 2000, 4000, 8000, 16000]
        out = sample_size_study(env, sizes, args.n_runs, args.seed)
    elif args.study == "overlap":
        eps = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
        out = overlap_study(env, eps, n=2000, n_runs=args.n_runs, seed=args.seed)
    else:
        grid = [1, 10, np.inf] if args.quick else [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, np.inf]
        out = clipping_study(env.with_epsilon(0.0), grid, n=2000, n_runs=args.n_runs, seed=args.seed)
    out |= {"study": args.study, "config": config, "seconds": round(time.perf_counter() - start, 1)}

    out_dir = ROOT / "results" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.study}{'_quick' if args.quick else ''}.json"
    (out_dir / name).write_text(json.dumps(out, indent=2))
    print(f"wrote results/synthetic/{name} in {out['seconds']}s")


if __name__ == "__main__":
    main()
