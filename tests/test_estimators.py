import numpy as np
import pytest

from counterfact.estimators import (
    IPW,
    SNDR,
    SNIPW,
    DirectMethod,
    DoublyRobust,
    DRos,
    OPEInput,
    SwitchDR,
    default_estimators,
)
from counterfact.synthetic import SyntheticBandit


def _toy() -> OPEInput:
    # 3 rounds, 2 actions. pi_e and q_hat are per-round rows.
    action = np.array([0, 1, 1])
    reward = np.array([1.0, 0.0, 1.0])
    pscore = np.array([0.5, 0.25, 0.5])
    pi_e = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    q_hat = np.array([[0.8, 0.2], [0.4, 0.4], [0.3, 0.6]])
    return OPEInput.build(action=action, reward=reward, pscore=pscore, pi_e=pi_e, q_hat=q_hat)


def test_build_reduces_to_weights_and_model_terms():
    d = _toy()
    np.testing.assert_allclose(d.weight, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(d.q_factual, [0.8, 0.4, 0.6])
    np.testing.assert_allclose(d.q_policy, [0.8, 0.4, 0.6])


def test_hand_computed_values():
    d = _toy()
    assert IPW().estimate(d) == pytest.approx((2 * 1 + 2 * 0 + 2 * 1) / 3)
    assert SNIPW().estimate(d) == pytest.approx(4 / 6)
    assert DirectMethod().estimate(d) == pytest.approx((0.8 + 0.4 + 0.6) / 3)
    dr = (0.8 + 2 * 0.2) + (0.4 + 2 * -0.4) + (0.6 + 2 * 0.4)
    assert DoublyRobust().estimate(d) == pytest.approx(dr / 3)
    assert IPW(clip=1.5).estimate(d) == pytest.approx((1.5 + 0 + 1.5) / 3)


@pytest.fixture(scope="module")
def logged():
    env = SyntheticBandit(seed=3, epsilon=0.2)
    log = env.sample(2000, np.random.default_rng(0))
    q_hat = np.clip(log.q_true + np.random.default_rng(1).normal(scale=0.1, size=log.q_true.shape), 0, 1)
    return log, q_hat


def test_limits_of_tuned_estimators(logged):
    log, q_hat = logged
    d = log.ope_input(q_hat)
    dm, dr = DirectMethod().estimate(d), DoublyRobust().estimate(d)
    assert SwitchDR(tau=np.inf).estimate(d) == pytest.approx(dr)
    assert SwitchDR(tau=0.0).estimate(d) == pytest.approx(dm)
    assert DRos(lam=np.inf).estimate(d) == pytest.approx(dr)
    assert DRos(lam=0.0).estimate(d) == pytest.approx(dm)
    assert DoublyRobust(clip=np.inf).estimate(d) == pytest.approx(dr)


def test_dr_with_zero_model_is_ipw(logged):
    log, _ = logged
    d = log.ope_input(np.zeros_like(log.q_true))
    assert DoublyRobust().estimate(d) == pytest.approx(IPW().estimate(d))
    assert SNDR().estimate(d) == pytest.approx(SNIPW().estimate(d))


def test_self_normalised_estimators_ignore_weight_scale(logged):
    log, q_hat = logged
    d = log.ope_input(q_hat)
    scaled = OPEInput(reward=d.reward, weight=d.weight * 7.0, q_factual=d.q_factual, q_policy=d.q_policy)
    assert SNIPW().estimate(scaled) == pytest.approx(SNIPW().estimate(d))
    assert SNDR().estimate(scaled) == pytest.approx(SNDR().estimate(d))


def test_clipping_never_increases_ipw_on_nonnegative_rewards(logged):
    log, _ = logged
    d = log.ope_input()
    values = [IPW(clip=c).estimate(d) for c in (np.inf, 50, 10, 2, 0.5)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:], strict=False))


def test_validation():
    with pytest.raises(ValueError):
        OPEInput.build(action=np.array([0]), reward=np.array([1.0]), pscore=np.array([0.0]), pi_e=np.ones((1, 1)))
    with pytest.raises(ValueError):
        IPW(clip=0)
    d = OPEInput(reward=np.ones(3), weight=np.ones(3))
    with pytest.raises(ValueError, match="reward-model"):
        DirectMethod().estimate(d)


@pytest.mark.slow
def test_unbiasedness_and_double_robustness():
    """Monte Carlo over 400 logged datasets with an exactly known policy value.

    IPW and DR are unbiased; DR stays unbiased with a badly wrong reward model (double
    robustness) while DM with the same model does not; a better model lowers DR's variance.
    """
    env = SyntheticBandit(seed=7, epsilon=0.3)
    truth = env.policy_value(n_contexts=1_000_000)
    rng = np.random.default_rng(0)
    runs = {name: [] for name in ("ipw", "dr_bad", "dm_bad", "dr_oracle")}
    for _ in range(400):
        log = env.sample(1000, rng)
        bad = np.full_like(log.q_true, 0.05)  # constant, far below the true value (~0.43)
        runs["ipw"].append(IPW().estimate(log.ope_input()))
        runs["dr_bad"].append(DoublyRobust().estimate(log.ope_input(bad)))
        runs["dm_bad"].append(DirectMethod().estimate(log.ope_input(bad)))
        runs["dr_oracle"].append(DoublyRobust().estimate(log.ope_input(log.q_true)))
    for name in ("ipw", "dr_bad", "dr_oracle"):
        e = np.array(runs[name])
        se = e.std(ddof=1) / np.sqrt(len(e))
        assert abs(e.mean() - truth) < 4 * se, name
    assert abs(np.mean(runs["dm_bad"]) - truth) > 0.1
    assert np.var(runs["dr_oracle"]) < np.var(runs["ipw"])


def test_default_estimator_names_are_unique():
    names = [e.name for e in default_estimators()]
    assert len(names) == len(set(names))
