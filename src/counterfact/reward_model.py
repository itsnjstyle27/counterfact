"""Reward models for DM and DR: a PyTorch click model, its training loop, and cross-fitting.

The model predicts P(click | context, item, slot) from the concatenation of the context
vector, the item's feature vector and a one-hot of the slot. `hidden=()` makes it a
logistic regression; any hidden layers make it an MLP that can learn context x item
interactions, which the linear model cannot represent.

Cross-fitting (Chernozhukov et al., 2018; used for DR by Narita et al., 2021) splits the
logged data into K folds and predicts each fold with a model trained on the others, so
the residuals r - q_hat that DR reweights are never computed on the model's own training
rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class BanditFeatures:
    """Reward-model inputs for n logged rounds."""

    context: np.ndarray  # (n, d_context)
    item_features: np.ndarray  # (n_actions, d_item)
    action: np.ndarray  # (n,)
    position: np.ndarray  # (n,) slot index in [0, n_positions)
    reward: np.ndarray  # (n,)
    n_positions: int = 1

    @property
    def n(self) -> int:
        return self.action.shape[0]

    @property
    def n_actions(self) -> int:
        return self.item_features.shape[0]

    def take(self, idx: np.ndarray) -> BanditFeatures:
        return replace(
            self,
            context=self.context[idx],
            action=self.action[idx],
            position=self.position[idx],
            reward=self.reward[idx],
        )


@dataclass(frozen=True)
class RewardModelConfig:
    hidden: tuple[int, ...] = ()
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 3
    val_fraction: float = 0.1

    @property
    def label(self) -> str:
        return "logistic" if not self.hidden else "mlp-" + "x".join(map(str, self.hidden))


class ClickModel(nn.Module):
    def __init__(self, d_context: int, d_item: int, n_positions: int, hidden: tuple[int, ...], base_rate: float):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = d_context + d_item + n_positions
        for width in hidden:
            layers += [nn.Linear(d_in, width), nn.ReLU()]
            d_in = width
        out = nn.Linear(d_in, 1)
        # Start at the base click rate: with ~0.4% positives, a zero-initialised bias spends
        # the first epochs just learning that almost nothing is clicked.
        p = min(max(base_rate, 1e-6), 1 - 1e-6)
        nn.init.constant_(out.bias, float(np.log(p / (1 - p))))
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.n_positions = n_positions

    def forward(self, context: torch.Tensor, item: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        slot = nn.functional.one_hot(position, self.n_positions).to(context.dtype)
        return self.net(torch.cat([context, item, slot], dim=1)).squeeze(1)

    def logits_all_actions(self, context: torch.Tensor, items: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Logits for every item in every round, shape (m, n_actions).

        The first layer is linear in the concatenated input, so W [x; e_a; s] splits into
        W_x x + W_a e_a + W_s s: each part is computed once and broadcast to (m, n_actions,
        width) instead of materialising m * n_actions concatenated input rows.
        """
        first = self.net[0]
        d_x, d_a = context.shape[1], items.shape[1]
        w = first.weight
        h = (
            (context @ w[:, :d_x].T)[:, None, :]
            + (items @ w[:, d_x : d_x + d_a].T)[None, :, :]
            + w[:, d_x + d_a :].T[position][:, None, :]
            + first.bias
        )
        for layer in self.net[1:]:
            h = layer(h)
        return h.squeeze(-1)


@dataclass
class FitResult:
    model: ClickModel
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0


def _tensors(features: BanditFeatures) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(features.context, dtype=torch.float32),
        torch.as_tensor(features.item_features, dtype=torch.float32),
        torch.as_tensor(features.action, dtype=torch.long),
        torch.as_tensor(features.position, dtype=torch.long),
        torch.as_tensor(features.reward, dtype=torch.float32),
    )


def fit(features: BanditFeatures, config: RewardModelConfig, seed: int = 0) -> FitResult:
    """Train a click model with minibatch AdamW and early stopping on a held-out split."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    context, items, action, position, reward = _tensors(features)

    perm = torch.randperm(features.n, generator=generator)
    n_val = int(features.n * config.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = ClickModel(
        context.shape[1], items.shape[1], features.n_positions, config.hidden, float(reward[train_idx].mean())
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def loss_on(idx: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            total = 0.0
            for start in range(0, len(idx), 65_536):
                b = idx[start : start + 65_536]
                logits = model(context[b], items[action[b]], position[b])
                total += nn.functional.binary_cross_entropy_with_logits(logits, reward[b], reduction="sum").item()
        return total / max(len(idx), 1)

    result = FitResult(model=model)
    best_state, best_val, stale = None, float("inf"), 0
    for epoch in range(config.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        running, seen = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            b = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(context[b], items[action[b]], position[b]), reward[b])
            loss.backward()
            optimiser.step()
            running += loss.item() * len(b)
            seen += len(b)
        result.train_loss.append(running / seen)
        if n_val == 0:
            continue
        val = loss_on(val_idx)
        result.val_loss.append(val)
        if val < best_val - 1e-7:
            best_val, stale, result.best_epoch = val, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return result


def predict_all_actions(model: ClickModel, features: BanditFeatures, chunk_rounds: int = 8192) -> np.ndarray:
    """P(click) for every item at each round's logged slot, shape (n, n_actions), float32."""
    context, items, _, position, _ = _tensors(features)
    out = np.empty((features.n, features.n_actions), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, features.n, chunk_rounds):
            stop = start + chunk_rounds
            logits = model.logits_all_actions(context[start:stop], items, position[start:stop])
            out[start:stop] = torch.sigmoid(logits).numpy()
    return out


@dataclass(frozen=True)
class Diagnostics:
    """Out-of-fold quality of the reward model on the logged (factual) actions."""

    log_loss: float
    brier: float
    auc: float
    calibration: float  # mean prediction / mean reward; 1.0 is calibrated in the large

    def as_dict(self) -> dict[str, float]:
        return {"log_loss": self.log_loss, "brier": self.brier, "auc": self.auc, "calibration": self.calibration}


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def diagnostics(reward: np.ndarray, q_factual: np.ndarray) -> Diagnostics:
    p = np.clip(q_factual.astype(np.float64), 1e-7, 1 - 1e-7)
    return Diagnostics(
        log_loss=float(-np.mean(reward * np.log(p) + (1 - reward) * np.log(1 - p))),
        brier=float(np.mean((p - reward) ** 2)),
        auc=auc_score(reward, p),
        calibration=float(p.mean() / reward.mean()) if reward.mean() > 0 else float("nan"),
    )


def cross_fit(
    features: BanditFeatures, config: RewardModelConfig, n_folds: int = 2, seed: int = 0
) -> tuple[np.ndarray, Diagnostics]:
    """Out-of-fold q_hat for every round and action, plus out-of-fold diagnostics.

    n_folds=1 trains on all rounds and predicts them in-sample (no cross-fitting); it
    exists for the ablation that measures what cross-fitting buys.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    q_hat = np.empty((features.n, features.n_actions), dtype=np.float32)
    if n_folds == 1:
        model = fit(features, config, seed=seed).model
        q_hat[:] = predict_all_actions(model, features)
    else:
        folds = np.array_split(np.random.default_rng(seed).permutation(features.n), n_folds)
        for k, held_out in enumerate(folds):
            train = np.concatenate([f for j, f in enumerate(folds) if j != k])
            model = fit(features.take(train), config, seed=seed + k).model
            q_hat[held_out] = predict_all_actions(model, features.take(held_out))
    q_factual = q_hat[np.arange(features.n), features.action]
    return q_hat, diagnostics(features.reward, q_factual)
