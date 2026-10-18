import numpy as np

from counterfact.policies import at_logged_position, bernoulli_thompson_sampling, uniform_random, zozotown_prior


def test_uniform_random():
    dist = uniform_random(4, 3)
    np.testing.assert_allclose(dist.sum(axis=0), 1.0)
    np.testing.assert_allclose(dist, 0.25)


def test_bts_is_a_distribution_per_position():
    alpha, beta = zozotown_prior("men")
    dist = bernoulli_thompson_sampling(alpha, beta, n_sim=20_000)
    assert dist.shape == (34, 3)
    np.testing.assert_allclose(dist.sum(axis=0), 1.0)
    assert np.all(dist.sum(axis=1) <= 1.0 + 1e-12)  # an item fills at most one slot per slate


def test_bts_ranks_a_dominant_item_first():
    alpha = np.array([900.0, 10.0, 10.0, 10.0])
    beta = np.array([100.0, 990.0, 990.0, 990.0])
    dist = bernoulli_thompson_sampling(alpha, beta, n_sim=5_000)
    assert dist[0, 0] > 0.999
    assert dist[0, 1:].sum() < 0.001


def test_bts_is_reproducible():
    alpha, beta = zozotown_prior("women")
    a = bernoulli_thompson_sampling(alpha, beta, n_sim=5_000, seed=1)
    b = bernoulli_thompson_sampling(alpha, beta, n_sim=5_000, seed=1)
    np.testing.assert_array_equal(a, b)


def test_at_logged_position():
    dist = np.arange(12, dtype=float).reshape(4, 3)
    out = at_logged_position(dist, np.array([0, 2, 1]))
    np.testing.assert_array_equal(out, [[0, 3, 6, 9], [2, 5, 8, 11], [1, 4, 7, 10]])


def test_logged_slate_policy_follows_the_latest_logged_propensity():
    from counterfact.policies import LoggedSlatePolicy

    # 2 items, 1 position. Item 0's propensity moves 0.5 -> 0.8 at t=10; item 1's 0.5 -> 0.2 at t=12.
    t = np.array([0, 10, 1, 12])
    action = np.array([0, 0, 1, 1])
    pos = np.zeros(4, dtype=int)
    p = np.array([0.5, 0.8, 0.5, 0.2])
    policy = LoggedSlatePolicy(t, action, pos, p, n_actions=2, len_list=1)
    np.testing.assert_allclose(
        policy.raw_propensity(np.array([5, 11, 13, -3]), np.zeros(4, int), np.zeros(4, int)), [0.5, 0.8, 0.8, 0.5]
    )
    dist, raw = policy.at(np.array([5, 11, 13]), np.zeros(3, int))
    np.testing.assert_allclose(raw, [1.0, 1.3, 1.0])  # at t=11 item 1's 0.5 is stale
    np.testing.assert_allclose(dist, [[0.5, 0.5], [0.8 / 1.3, 0.5 / 1.3], [0.8, 0.2]])
    np.testing.assert_allclose(dist.sum(axis=1), 1.0)
