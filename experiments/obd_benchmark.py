"""Run the Open Bandit Dataset OPE benchmark and write one JSON per task to results/obd/.

# main table: every campaign and direction, MLP reward model
python experiments/obd_benchmark.py --tag main

# ablation: evaluate BTS as it actually ran (rebuilt from its logs) instead of the snapshot
python experiments/obd_benchmark.py --tag target --directions random_to_bts --target replayed

# ablation: logistic reward model
python experiments/obd_benchmark.py --tag logistic --campaigns all --hidden
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counterfact.data import CAMPAIGNS
from counterfact.estimators import default_estimators
from counterfact.obd_benchmark import DIRECTIONS, TARGETS, Task, prepare, run
from counterfact.reward_model import RewardModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="name for this configuration in results/obd/")
    parser.add_argument("--campaigns", nargs="+", default=list(CAMPAIGNS), choices=CAMPAIGNS)
    parser.add_argument("--directions", nargs="+", default=list(DIRECTIONS), choices=DIRECTIONS)
    parser.add_argument("--target", default="snapshot", choices=TARGETS)
    parser.add_argument(
        "--hidden", nargs="*", type=int, default=[64], help="reward-model hidden widths; none = logistic"
    )
    parser.add_argument("--item-ids", action="store_true", help="give the reward model a one-hot item id")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--max-sample", type=int, default=1_000_000, help="cap on rows per bootstrap sample")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = RewardModelConfig(hidden=tuple(args.hidden))
    out_dir = ROOT / "results" / "obd"
    out_dir.mkdir(parents=True, exist_ok=True)
    for campaign in args.campaigns:
        for direction in args.directions:
            task = Task(campaign, direction, args.target)
            start = time.perf_counter()
            prepared = prepare(task, args.data_root)
            result = run(
                prepared,
                default_estimators(),
                config,
                n_runs=args.n_runs,
                sample_size=args.max_sample,
                n_folds=args.n_folds,
                seed=args.seed,
                item_ids=args.item_ids,
                log=lambda msg: print(msg, flush=True),
            )
            result["tag"] = args.tag
            result["seconds"] = round(time.perf_counter() - start, 1)
            name = f"{args.tag}_{campaign}_{direction}" + (f"_{args.target}" if direction == "random_to_bts" else "")
            (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
            best = min(result["summary"].items(), key=lambda kv: kv[1]["relative_error_mean"])
            print(f"{task.label}: best {best[0]} rel. error {best[1]['relative_error_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
