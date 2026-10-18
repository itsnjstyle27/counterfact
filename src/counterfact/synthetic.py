"""A synthetic contextual bandit with a known policy value.

Real logged data only has a noisy on-policy estimate as ground truth. Here the expected
reward q(x, a) is known in closed form, so bias, variance and MSE of every estimator can
be measured exactly, and the two quantities that drive OPE error can be dialled
independently:

* overlap: the evaluation policy is epsilon-greedy on the true reward, so epsilon = 1 is
  uniform (close to the logging policy) and epsilon = 0 is deterministic (large weights);
* reward-model misspecification: q depends on context x action interactions, which a
  logistic model on [x, one-hot(a)] cannot represent but an MLP can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from counterfact.estimators import OPEInput
from counterfact.reward_model import BanditFeatures


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class SyntheticLog:
    context: np.ndarray  # (n, d)
    action: np.ndarray  # (n,)
    reward: np.ndarray  # (n,)
    pscore: np.ndarray  # (n,) pi_b(a_i | x_i)
    pi_e: np.ndarray  # (n, n_actions)
    q_true: np.ndarray  # (n, n_actions) true expected reward

    @property
    def n(self) -> int:
        return self.action.shape[0]

    def ope_input(self, q_hat: np.ndarray | None = None) -> OPEInput:
        return OPEInput.build(action=self.action, reward=self.reward, pscore=self.pscore, pi_e=self.pi_e, q_hat=q_hat)

    def features(self) -> BanditFeatures:
        """Reward-model inputs: the context, with items identified only by a one-hot id."""
        n_actions = self.pi_e.shape[1]
        return BanditFeatures(
            context=self.context.astype(np.float32),
            item_features=np.eye(n_actions, dtype=np.float32),
            action=self.action,
            position=np.zeros(self.n, dtype=np.int64),
            reward=self.reward,
        )


class SyntheticBandit:
    """q(x, a) = sigmoid(base + b_a + scale * <x, theta_a>), x ~ N(0, I).

    The logging policy is a softmax over a noisy copy of the reward parameters (it is
    informed but imperfect, like a production policy); the evaluation policy is
    epsilon-greedy on the true expected reward.
    """

    def __init__(
        self,
        n_actions: int = 10,
        dim_context: int = 5,
        base_logit: float = -1.5,
        interaction_scale: float = 1.0,
        behavior_correlation: float = 0.3,
        behavior_inverse_temperature: float = 1.0,
        epsilon: float = 0.3,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.base_logit = base_logit
        self.interaction_scale = interaction_scale
        self.epsilon = epsilon
        self.behavior_inverse_temperature = behavior_inverse_temperature
        self.theta = rng.normal(size=(dim_context, n_actions)) / np.sqrt(dim_context)
        self.action_bias = rng.normal(scale=0.5, size=n_actions)
        noise = rng.normal(size=(dim_context, n_actions)) / np.sqrt(dim_context)
        rho = behavior_correlation
        self.behavior_theta = rho * self.theta + np.sqrt(1 - rho**2) * noise

    def with_epsilon(self, epsilon: float) -> SyntheticBandit:
        """Same environment and logging policy, different evaluation policy."""
        clone = object.__new__(SyntheticBandit)
        clone.__dict__.update(self.__dict__)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        clone.epsilon = epsilon
        return clone

    def expected_reward(self, context: np.ndarray) -> np.ndarray:
        logits = self.base_logit + self.action_bias + self.interaction_scale * context @ self.theta
        return 1.0 / (1.0 + np.exp(-logits))

    def behavior_policy(self, context: np.ndarray) -> np.ndarray:
        return _softmax(self.behavior_inverse_temperature * (context @ self.behavior_theta + self.action_bias))

    def evaluation_policy(self, context: np.ndarray) -> np.ndarray:
        q = self.expected_reward(context)
        greedy = np.zeros_like(q)
        greedy[np.arange(len(q)), q.argmax(axis=1)] = 1.0
        return (1.0 - self.epsilon) * greedy + self.epsilon / self.n_actions

    def policy_value(self, n_contexts: int = 2_000_000, seed: int = 12345, chunk: int = 200_000) -> float:
        """V(pi_e) = E_x[sum_a pi_e(a|x) q(x, a)], exact over actions, Monte Carlo over x.

        At the default 2M contexts the Monte Carlo standard error is ~1e-4, an order of
        magnitude below the smallest estimator RMSE reported in the experiments.
        """
        rng = np.random.default_rng(seed)
        total, done = 0.0, 0
        while done < n_contexts:
            m = min(chunk, n_contexts - done)
            x = rng.normal(size=(m, self.dim_context))
            total += float((self.evaluation_policy(x) * self.expected_reward(x)).sum())
            done += m
        return total / n_contexts

    def sample(self, n: int, rng: np.random.Generator) -> SyntheticLog:
        x = rng.normal(size=(n, self.dim_context))
        pi_b = self.behavior_policy(x)
        cumulative = pi_b.cumsum(axis=1)
        action = (rng.random((n, 1)) > cumulative).sum(axis=1)
        action = np.minimum(action, self.n_actions - 1)
        q = self.expected_reward(x)
        rows = np.arange(n)
        reward = (rng.random(n) < q[rows, action]).astype(np.float64)
        return SyntheticLog(
            context=x,
            action=action,
            reward=reward,
            pscore=pi_b[rows, action],
            pi_e=self.evaluation_policy(x),
            q_true=q,
        )
