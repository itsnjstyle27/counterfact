"""End-to-end smoke test of the benchmark harness on a tiny fake release (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

from counterfact.data import convert_one
from counterfact.estimators import default_estimators
from counterfact.obd_benchmark import Task, prepare, run
from counterfact.reward_model import RewardModelConfig

N_ITEMS = 34  # the "men" campaign, so obp's production prior applies


def _fake_arm(root, policy: str, n: int, rng: np.random.Generator) -> None:
    base = root / policy / "men"
    base.mkdir(parents=True)
    pd.DataFrame(
        {
            "item_id": range(N_ITEMS),
            "item_feature_0": rng.normal(size=N_ITEMS),
            "item_feature_1": rng.choice(["a", "b", "c"], N_ITEMS),
            "item_feature_2": rng.choice(["x", "y"], N_ITEMS),
            "item_feature_3": rng.choice(["p", "q"], N_ITEMS),
        }
    ).to_csv(base / "item_context.csv")
    times = pd.Timestamp("2019-11-24", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 3600, n)), unit="s")
    pscore = np.full(n, 1 / N_ITEMS) if policy == "random" else rng.uniform(0.005, 0.3, n)
    rows = pd.DataFrame(
        {
            "timestamp": times.astype(str),
            "item_id": np.tile(np.arange(N_ITEMS), n // N_ITEMS + 1)[:n],  # every (item, slot) appears
            "position": np.tile([1, 2, 3], n // 3 + 1)[:n],
            "click": (rng.random(n) < 0.1).astype(int),
            "propensity_score": pscore,
        }
    )
    for j in range(4):
        rows[f"user_feature_{j}"] = rng.choice(["u", "v", "w"], n)
    for j in range(N_ITEMS):
        rows[f"user-item_affinity_{j}"] = 0.0
    rows.to_csv(base / "men.csv")


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("obd")
    rng = np.random.default_rng(0)
    for policy in ("random", "bts"):
        _fake_arm(tmp / "raw", policy, 34 * 3 * 4, rng)
        convert_one(tmp / "raw", policy, "men", tmp / "processed")
    return tmp


@pytest.mark.parametrize(
    "task", [Task("men", "random_to_bts"), Task("men", "random_to_bts", "replayed"), Task("men", "bts_to_random")]
)
def test_benchmark_runs_end_to_end(data_root, task):
    prepared = prepare(task, root=data_root, n_sim=2_000)
    result = run(
        prepared,
        default_estimators(),
        RewardModelConfig(hidden=(8,), epochs=2),
        n_runs=2,
        item_ids=True,
        log=lambda _: None,
    )
    assert result["n_runs"] == 2
    assert set(result["summary"]) == {e.name for e in default_estimators()}
    for name, values in result["estimates"].items():
        assert len(values) == 2 and all(np.isfinite(values)), name
    assert result["truth"] == pytest.approx(prepared.truth)
