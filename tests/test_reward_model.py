import numpy as np
import pytest

from counterfact.reward_model import RewardModelConfig, auc_score, cross_fit
from counterfact.synthetic import SyntheticBandit


def test_auc_score():
    y = np.array([0, 0, 1, 1])
    assert auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc_score(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(auc_score(np.zeros(3), np.arange(3.0)))


@pytest.mark.slow
def test_mlp_learns_interactions_the_logistic_model_cannot():
    env = SyntheticBandit(seed=0, interaction_scale=2.0)
    log = env.sample(20_000, np.random.default_rng(0))
    features = log.features()
    errors = {}
    for config in (RewardModelConfig(hidden=()), RewardModelConfig(hidden=(64, 64))):
        q_hat, diag = cross_fit(features, config, n_folds=2, seed=0)
        assert q_hat.shape == log.q_true.shape
        assert 0.9 < diag.calibration < 1.1
        errors[config.label] = float(np.sqrt(np.mean((q_hat - log.q_true) ** 2)))
    assert errors["mlp-64x64"] < 0.75 * errors["logistic"]


@pytest.mark.parametrize("hidden", [(), (16, 8)])
def test_factorised_all_action_logits_match_the_plain_forward_pass(hidden):
    import torch

    from counterfact.reward_model import ClickModel

    torch.manual_seed(0)
    model = ClickModel(d_context=5, d_item=4, n_positions=3, hidden=hidden, base_rate=0.1)
    context, items = torch.randn(7, 5), torch.randn(6, 4)
    position = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    fast = model.logits_all_actions(context, items, position)
    slow = model(context.repeat_interleave(6, 0), items.repeat(7, 1), position.repeat_interleave(6)).reshape(7, 6)
    torch.testing.assert_close(fast, slow)
