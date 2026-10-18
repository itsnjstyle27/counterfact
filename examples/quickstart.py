"""Estimate a policy's value from logs it never generated, in ~30 lines.

python examples/quickstart.py
"""

import numpy as np

from counterfact.estimators import default_estimators
from counterfact.evaluation import bootstrap_ci
from counterfact.reward_model import RewardModelConfig, cross_fit
from counterfact.synthetic import SyntheticBandit

# A logging policy ran in production; we want the value of a greedier evaluation policy.
env = SyntheticBandit(seed=0, epsilon=0.1)
log = env.sample(5_000, np.random.default_rng(42))
truth = env.policy_value()

# Reward model for DM / DR, cross-fitted so DR never reweights in-sample residuals.
q_hat, diagnostics = cross_fit(log.features(), RewardModelConfig(hidden=(64, 64)), n_folds=2)
data = log.ope_input(q_hat)

print(f"true value {truth:.4f}   (logging policy's own value {log.reward.mean():.4f})")
print(f"reward model: AUC {diagnostics.auc:.3f}, calibration {diagnostics.calibration:.3f}\n")
for estimator in default_estimators():
    lo, hi = bootstrap_ci(estimator, data, n_boot=300)
    est = estimator.estimate(data)
    print(f"{estimator.name:22s} {est:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  error {est - truth:+.4f}")
