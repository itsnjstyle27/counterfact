# Method

This document derives the estimators implemented in
[`src/counterfact/estimators.py`](../src/counterfact/estimators.py), states the assumptions each one
needs, and describes the evaluation protocol. Notation follows the code.

## Problem

A logging (behaviour) policy $\pi_b$ ran in production. For each round $i = 1..n$ we observe a
context $x_i$, the action it chose $a_i \sim \pi_b(\cdot \mid x_i)$, the propensity
$p_i = \pi_b(a_i \mid x_i)$ and a reward $r_i$ with $\mathbb{E}[r \mid x, a] = q(x, a)$.

We want the value of a different policy $\pi_e$ that never ran:

$$V(\pi_e) = \mathbb{E}_{x}\Big[\sum_a \pi_e(a \mid x)\, q(x, a)\Big].$$

Everything below needs **overlap**: $\pi_b(a \mid x) > 0$ wherever $\pi_e(a \mid x) > 0$.

### Slates

ZOZOTOWN shows three items (left, centre, right). Following the Open Bandit Dataset paper, each
(round, slot) pair is treated as one observation: the action is the item, $\pi(a \mid x, k)$ is the
probability that item $a$ fills slot $k$, and the click on that item is assumed to depend only on
$(x, a, k)$, not on the other two items. The policy value is then the expected click rate per
impression. This position-wise independence is a modelling assumption. If neighbouring items
change click rates, the estimators below are biased in ways no sample size fixes. That is one
candidate explanation for the residual errors on the real data.

## Estimators

Write $w_i = \pi_e(a_i \mid x_i) / p_i$ for the importance weight, $\hat q$ for a reward model,
$\hat q_i = \hat q(x_i, a_i)$ and $\hat q^{\pi}_i = \sum_a \pi_e(a \mid x_i)\, \hat q(x_i, a)$.
All estimators consume only these four per-round vectors, which `OPEInput.build` computes once.

### Direct method (DM)

$$\hat V_{DM} = \frac1n \sum_i \hat q^{\pi}_i$$

Its bias is $\mathbb{E}_x\big[\sum_a \pi_e(a \mid x)\,(\hat q(x,a) - q(x,a))\big]$: the reward model's
error, averaged *under the evaluation policy*. Logged data mostly covers actions $\pi_b$ likes, so
the model is least accurate exactly where $\pi_e$ differs from $\pi_b$. Variance is low, but the
bias does not shrink with $n$ for a misspecified model (see the synthetic sample-size study).

### Inverse propensity weighting (IPW)

$$\hat V_{IPW} = \frac1n \sum_i w_i\, r_i$$

**Unbiased.** For one round,

$$\mathbb{E}[w\, r] = \mathbb{E}_x \sum_a \pi_b(a \mid x)\,\frac{\pi_e(a \mid x)}{\pi_b(a \mid x)}\, q(x, a) = V(\pi_e).$$

**Variance.** $\operatorname{Var}(w r)/n$, and $\mathbb{E}[w^2 r^2] \le \mathbb{E}[w^2]$ for
$r \in [0, 1]$. $\mathbb{E}_{\pi_b}[w^2] = 1 + \chi^2(\pi_e \,\|\, \pi_b)$ grows with the divergence
between the policies. In the BTS → random direction of the Open Bandit Dataset, weights reach
$0.0125 / 10^{-6} = 12{,}500$.

**Clipping** at $\lambda$ uses $\min(w_i, \lambda)$. For non-negative rewards this can only lower
the estimate, so it adds a downward bias of $\mathbb{E}[(w - \lambda)_+\, r]$ in exchange for
variance at most $\lambda^2/n$.

### Self-normalised IPW (SNIPW)

$$\hat V_{SNIPW} = \frac{\sum_i w_i r_i}{\sum_i w_i}$$

Since $\mathbb{E}[w] = 1$, the denominator estimates 1, so SNIPW is a ratio estimator. It is
biased at $O(1/n)$ but consistent, always within the reward range, and invariant to rescaling
all weights. That makes it robust when a few huge weights dominate the sum.

### Doubly robust (DR)

$$\hat V_{DR} = \frac1n \sum_i \Big[\hat q^{\pi}_i + w_i\,(r_i - \hat q_i)\Big]$$

**Double robustness.** Take expectations over $a \sim \pi_b$ and $r$ given $x$:

$$\mathbb{E}\big[\hat q^{\pi} + w\,(r - \hat q)\,\big|\, x\big] = \sum_a \pi_e(a \mid x)\,\hat q(x,a) + \sum_a \pi_e(a \mid x)\,\big(q(x,a) - \hat q(x,a)\big) = \sum_a \pi_e(a \mid x)\, q(x,a).$$

The reward model cancels, so DR is unbiased with *any* fixed $\hat q$ as long as the weights are
correct. If the weights are wrong but $\hat q = q$, the correction term has mean zero and DR is
again unbiased. The test `test_unbiasedness_and_double_robustness` checks the first case by Monte
Carlo, using a constant reward model that is far below the true value: DM is badly biased and DR
is not.

**Variance.** The IPW term now weights the residual $r - \hat q$ instead of $r$, so a good model
shrinks the variance. With $\hat q = 0$, DR is exactly IPW (also tested).

**Cross-fitting.** The argument above needs $\hat q$ to be independent of the round it corrects.
If the model is trained on the same rows, it partly fits their noise, the residuals shrink
artificially, and the correction under-corrects. `reward_model.cross_fit` splits the data into $K$
folds and predicts each fold with a model trained on the others.

### Self-normalised DR (SNDR)

Divides the correction term by $\frac1n \sum_i w_i$, which brings SNIPW's stability to the
correction term.

### Switch-DR (Wang et al., 2017)

$$\hat V_{Switch} = \frac1n \sum_i \Big[\hat q^{\pi}_i + \mathbb{1}[w_i \le \tau]\, w_i\,(r_i - \hat q_i)\Big]$$

The rare rounds with extreme weights carry most of the variance, so they get the reward model
alone. $\tau = \infty$ gives DR and $\tau = 0$ gives DM. The bias is limited to the $\pi_e$-mass on
the switched-off rounds times the model's error there.

### DR with optimistic shrinkage, DRos (Su et al., 2020)

Replaces $w$ with $\tilde w = \lambda w / (w^2 + \lambda)$, the per-round weight that minimises a
bound on DR's MSE. $\tilde w \approx w$ for $w \ll \sqrt\lambda$, and $\tilde w \to 0$ as $w \to
\infty$. $\lambda = \infty$ gives DR and $\lambda = 0$ gives DM.

## Reward model

`reward_model.ClickModel` predicts the click probability from the concatenation of the context
vector, the item's feature vector and a one-hot of the slot. There are two variants:

- **logistic** (`hidden=()`): additive in context, item and slot. It cannot represent
  "this user type prefers that item", so it is misspecified whenever the reward has
  context × item interactions (the synthetic environment has them by construction).
- **MLP** (`hidden=(64,)` on the real data, `(64, 64)` in the synthetic studies): learns
  interactions.

Training is a hand-written loop: AdamW, binary cross-entropy, minibatches, early stopping on a
held-out 10% split. Clicks are rare (0.35% to 0.67%), so the output bias starts at the base-rate
logit. Calibration matters more than ranking here, because DM and DR use the predicted
*probabilities* as values. Class reweighting would wreck that, so none is used. Out-of-fold
calibration (mean prediction / mean click) is logged for every run.

For every round the estimators need predictions for *all* items, which is $n \times |\mathcal A|$
forward passes. The first layer is linear in the concatenated input, so it decomposes as
$W[x; e_a; s] = W_x x + W_a e_a + W_s s$. `logits_all_actions` computes each part once and
broadcasts it to $(n, |\mathcal A|, \text{width})$ instead of materialising $n|\mathcal A|$ input
rows. A test checks that the result matches the plain forward pass.

## Evaluation protocol on the Open Bandit Dataset

Both arms of ZOZOTOWN's A/B test (uniform random and Bernoulli Thompson sampling) ran in the same
week on the same campaigns. Each arm's mean click rate is therefore an on-policy estimate of its
own value with a small standard error (12.4M BTS rows and 1.37M random rows on "all"). For each
campaign:

- **random → BTS:** estimate BTS's value from the random arm's logs. Behaviour propensities are
  exactly $1/|\mathcal A|$.
- **BTS → random:** estimate the random policy's value from BTS's logs, using the logged BTS
  propensities.

Each run draws a bootstrap sample of the logged arm (capped at 1M rows), cross-fits the reward
model on it, and applies every estimator. The reported metric is the paper's relative error
$|\hat V - V| / V$, as mean ± standard deviation over runs. Two standard errors of the ground truth
are shown as a noise floor: no estimator can be credited with beating it.

## Which BTS policy is being evaluated?

obp replicates the production BTS policy from one set of Beta parameters
(`BernoulliTS(is_zozotown_prior=True)`, whose README describes it as replicating "the policy in
the ZOZOTOWN production"). Simulating it gives a fixed $(|\mathcal A|, 3)$ distribution. The BTS
log tells a different story. The propensity it recorded for each (item, slot) changes throughout
the week, with a median of hundreds of distinct values per pair. The ground truth (the BTS arm's
click rate) is the value of this drifting policy, averaged over the week.

`policies.LoggedSlatePolicy` rebuilds the policy at any time $t$. For every (item, slot) it takes
the propensity logged closest to $t$ at or before it, then renormalises each slot to sum to 1.
Before renormalising, the rebuilt rows sum to more than 1, because items whose probability fell
are rarely shown afterwards, so their last logged value is stale. The rebuild is therefore an
approximation. `experiments/bts_policy_audit.py` quantifies both the gap between the snapshot and
the logged policy and how much it moves the estimates. The benchmark runs under both targets.
