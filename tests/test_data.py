import zipfile

import numpy as np
import pandas as pd
import pytest

from counterfact.data import convert_one, load


def _write_fixture(root, policy="bts", campaign="men", n_items=3):
    base = root / policy / campaign
    base.mkdir(parents=True)
    pd.DataFrame(
        {
            "item_id": range(n_items),
            "item_feature_0": [0.1, -0.2, 0.3],
            "item_feature_1": ["a", "b", "a"],
            "item_feature_2": ["x", "x", "y"],
            "item_feature_3": ["p", "q", "r"],
        }
    ).to_csv(base / "item_context.csv")
    rows = pd.DataFrame(
        {
            "timestamp": ["2019-11-24 00:00:03+00:00", "2019-11-24 00:00:01+00:00", "2019-11-24 00:00:02+00:00"],
            "item_id": [2, 0, 1],
            "position": [3, 1, 2],
            "click": [1, 0, 0],
            "propensity_score": [0.5, 0.25, 0.125],
            "user_feature_0": ["u", "v", "u"],
            "user_feature_1": ["k", "k", "k"],
            "user_feature_2": ["m", "n", "o"],
            "user_feature_3": ["z", "z", "w"],
        }
    )
    for j in range(n_items):
        rows[f"user-item_affinity_{j}"] = [0.0, 0.0, 0.0]
    rows.loc[0, "user-item_affinity_2"] = 1.0
    rows.to_csv(base / f"{campaign}.csv")


@pytest.mark.parametrize("as_zip", [False, True])
def test_convert_and_load(tmp_path, as_zip):
    raw = tmp_path / "raw" / "open_bandit_dataset"
    _write_fixture(raw)
    source = raw
    if as_zip:
        source = tmp_path / "release.zip"
        with zipfile.ZipFile(source, "w") as zf:
            for f in raw.rglob("*.csv"):
                zf.write(f, f.relative_to(raw.parent).as_posix())

    card = convert_one(source, "bts", "men", tmp_path / "out" / "processed")
    assert card["n_rounds"] == 3
    assert card["clicks"] == 1
    assert card["own_affinity_nonzero_fraction"] == pytest.approx(1 / 3)
    assert card["click_rate_when_own_affinity_nonzero"] == 1.0

    data = load("bts", "men", root=tmp_path / "out")
    # sorted by timestamp, positions shifted to 0-based
    np.testing.assert_array_equal(data.action, [0, 1, 2])
    np.testing.assert_array_equal(data.position, [0, 1, 2])
    np.testing.assert_array_equal(data.reward, [0, 0, 1])
    np.testing.assert_allclose(data.pscore, [0.25, 0.125, 0.5])
    assert data.n_actions == 3
    assert data.user_cardinality == (2, 1, 3, 2)
    onehot = data.user_onehot()
    assert onehot.shape == (3, 8)
    np.testing.assert_allclose(onehot.sum(axis=1), 4.0)
    # 3 one-hot item categoricals (2 + 2 + 3 levels) + 1 numeric feature
    assert data.item_features.shape == (3, 8)
    assert data.on_policy_value() == pytest.approx(1 / 3)

    train, test = data.time_split(test_fraction=1 / 3)
    assert train.n_rounds == 2 and test.n_rounds == 1
