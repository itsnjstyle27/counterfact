"""Open Bandit Dataset (Saito et al., 2021): conversion from the released CSVs and loading.

Raw layout, shared by the full release and the sample bundled with obp:

    {root}/{behavior_policy}/{campaign}/{campaign}.csv
    {root}/{behavior_policy}/{campaign}/item_context.csv

The full release is 11.7 GB of CSV. `convert` streams each file in chunks, from a
directory or straight out of the downloaded zip, keeps the columns the experiments use,
and writes one compact .npz per (policy, campaign) plus a JSON data card with summary
statistics.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

POLICIES = ("random", "bts")
CAMPAIGNS = ("all", "men", "women")
USER_COLUMNS = tuple(f"user_feature_{i}" for i in range(4))
ITEM_CATEGORICAL = ("item_feature_1", "item_feature_2", "item_feature_3")


def data_dir() -> Path:
    """Root for downloaded and processed data (override with COUNTERFACT_DATA)."""
    env = os.environ.get("COUNTERFACT_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class LoggedBandit:
    """Logged bandit feedback from one behaviour policy on one campaign, sorted by time."""

    behavior_policy: str
    campaign: str
    timestamp: np.ndarray  # int64 nanoseconds since epoch (UTC)
    action: np.ndarray  # item id
    position: np.ndarray  # 0 = left, 1 = centre, 2 = right
    reward: np.ndarray  # click indicator
    pscore: np.ndarray  # P(item shown at this position) under the behaviour policy
    user_codes: np.ndarray  # (n, 4) integer codes of the categorical user features
    user_cardinality: tuple[int, ...]
    item_features: np.ndarray  # (n_actions, d_item): one-hot categoricals + 1 numeric

    @property
    def n_rounds(self) -> int:
        return self.action.shape[0]

    @property
    def n_actions(self) -> int:
        return self.item_features.shape[0]

    @property
    def len_list(self) -> int:
        return 3

    def on_policy_value(self) -> float:
        """Mean click rate: the behaviour policy's own value, the ground truth in OPE benchmarks."""
        return float(self.reward.mean())

    def user_onehot(self) -> np.ndarray:
        """One-hot encoding of the four categorical user features, shape (n, sum(cardinality))."""
        offsets = np.concatenate([[0], np.cumsum(self.user_cardinality)[:-1]])
        out = np.zeros((self.n_rounds, sum(self.user_cardinality)), dtype=np.float32)
        rows = np.arange(self.n_rounds)
        for j, offset in enumerate(offsets):
            out[rows, offset + self.user_codes[:, j]] = 1.0
        return out

    def take(self, idx: np.ndarray) -> LoggedBandit:
        return replace(
            self,
            timestamp=self.timestamp[idx],
            action=self.action[idx],
            position=self.position[idx],
            reward=self.reward[idx],
            pscore=self.pscore[idx],
            user_codes=self.user_codes[idx],
        )

    def time_split(self, test_fraction: float) -> tuple[LoggedBandit, LoggedBandit]:
        """Earlier rounds for training, the last `test_fraction` for testing."""
        cut = int(self.n_rounds * (1.0 - test_fraction))
        return self.take(np.arange(cut)), self.take(np.arange(cut, self.n_rounds))


def processed_path(behavior_policy: str, campaign: str, root: Path | None = None) -> Path:
    return (root or data_dir()) / "processed" / f"{behavior_policy}_{campaign}.npz"


def load(behavior_policy: str, campaign: str, root: Path | None = None) -> LoggedBandit:
    """Load a converted (policy, campaign) file. Run `scripts/prepare_data.py` first."""
    path = processed_path(behavior_policy, campaign, root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_data.py first")
    with np.load(path) as z:
        return LoggedBandit(
            behavior_policy=behavior_policy,
            campaign=campaign,
            timestamp=z["timestamp"],
            action=z["action"].astype(np.int64),
            position=z["position"].astype(np.int64),
            reward=z["reward"].astype(np.float64),
            pscore=z["pscore"],
            user_codes=z["user_codes"].astype(np.int64),
            user_cardinality=tuple(int(c) for c in z["user_cardinality"]),
            item_features=z["item_features"],
        )


# --- conversion -------------------------------------------------------------------------


@contextmanager
def _open_member(source: Path, relative: str) -> Iterator[io.BufferedIOBase]:
    """Open `relative` (e.g. 'bts/all/all.csv') inside a directory or a release zip."""
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as zf:
            matches = [n for n in zf.namelist() if n.endswith("/" + relative) or n == relative]
            if len(matches) != 1:
                raise FileNotFoundError(f"{relative} not found (or ambiguous) in {source}")
            with zf.open(matches[0]) as fh:
                yield fh
    else:
        with open(source / relative, "rb") as fh:
            yield fh


def _encode_items(item_context: pd.DataFrame) -> np.ndarray:
    item_context = item_context.sort_values("item_id")
    if not np.array_equal(item_context["item_id"].to_numpy(), np.arange(len(item_context))):
        raise ValueError("item ids are not 0..n_actions-1")
    onehot = pd.get_dummies(item_context[list(ITEM_CATEGORICAL)].astype(str), dtype=np.float32)
    numeric = item_context[["item_feature_0"]].to_numpy(np.float32)
    return np.hstack([onehot.to_numpy(np.float32), numeric])


def convert_one(source: Path, behavior_policy: str, campaign: str, out_dir: Path, chunksize: int = 500_000) -> dict:
    """Stream one (policy, campaign) CSV into a compact .npz and return its data card."""
    base = f"{behavior_policy}/{campaign}"
    with _open_member(source, f"{base}/item_context.csv") as fh:
        item_features = _encode_items(pd.read_csv(fh, index_col=0))
    n_actions = item_features.shape[0]

    vocab: list[dict[str, int]] = [{} for _ in USER_COLUMNS]
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("timestamp", "action", "position", "reward", "pscore", "user")
    }
    own_affinity_nonzero = 0
    clicks_with_affinity = 0
    affinity_cols = [f"user-item_affinity_{j}" for j in range(n_actions)]

    with _open_member(source, f"{base}/{campaign}.csv") as fh:
        dtypes = {"item_id": np.int16, "position": np.int8, "click": np.int8, "propensity_score": np.float64}
        dtypes |= dict.fromkeys(USER_COLUMNS, str) | dict.fromkeys(affinity_cols, np.float32)
        for chunk in pd.read_csv(fh, index_col=0, dtype=dtypes, chunksize=chunksize):
            ts = pd.to_datetime(chunk["timestamp"], format="ISO8601", utc=True)
            parts["timestamp"].append(ts.dt.tz_convert(None).to_numpy("datetime64[ns]").astype(np.int64))
            action = chunk["item_id"].to_numpy(np.int16)
            parts["action"].append(action)
            parts["position"].append((chunk["position"].to_numpy(np.int8) - 1).astype(np.int8))
            click = chunk["click"].to_numpy(np.int8)
            parts["reward"].append(click)
            parts["pscore"].append(chunk["propensity_score"].to_numpy(np.float64))
            codes = np.empty((len(chunk), len(USER_COLUMNS)), dtype=np.int16)
            for j, col in enumerate(USER_COLUMNS):
                values = chunk[col].astype(str)
                for v in pd.unique(values):
                    vocab[j].setdefault(v, len(vocab[j]))
                codes[:, j] = values.map(vocab[j]).to_numpy(np.int16)
            parts["user"].append(codes)
            own = chunk[affinity_cols].to_numpy(np.float32)[np.arange(len(chunk)), action]
            own_affinity_nonzero += int((own != 0).sum())
            clicks_with_affinity += int(click[own != 0].sum())

    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(arrays["timestamp"], kind="stable")
    arrays = {k: v[order] for k, v in arrays.items()}
    if arrays["action"].max() >= n_actions or arrays["position"].min() < 0 or arrays["position"].max() > 2:
        raise ValueError(f"{base}: item ids or positions out of range")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{behavior_policy}_{campaign}.npz"
    np.savez(
        out,
        timestamp=arrays["timestamp"],
        action=arrays["action"],
        position=arrays["position"],
        reward=arrays["reward"],
        pscore=arrays["pscore"],
        user_codes=arrays["user"],
        user_cardinality=np.array([len(v) for v in vocab]),
        item_features=item_features,
    )

    n = len(arrays["action"])
    reward = arrays["reward"]
    card = {
        "behavior_policy": behavior_policy,
        "campaign": campaign,
        "n_rounds": n,
        "n_actions": n_actions,
        "clicks": int(reward.sum()),
        "click_rate": float(reward.mean()),
        "pscore_min": float(arrays["pscore"].min()),
        "pscore_max": float(arrays["pscore"].max()),
        "user_cardinality": [len(v) for v in vocab],
        "item_feature_dim": int(item_features.shape[1]),
        "first_timestamp": str(np.datetime64(int(arrays["timestamp"][0]), "ns")),
        "last_timestamp": str(np.datetime64(int(arrays["timestamp"][-1]), "ns")),
        "own_affinity_nonzero_fraction": own_affinity_nonzero / n,
        "click_rate_when_own_affinity_nonzero": clicks_with_affinity / own_affinity_nonzero
        if own_affinity_nonzero
        else None,
    }
    out.with_suffix(".json").write_text(json.dumps(card, indent=2))
    return card
