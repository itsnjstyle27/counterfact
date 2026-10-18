"""Audit: is the BTS policy obp ships the policy that generated the BTS logs?

obp replicates ZOZOTOWN's Thompson-sampling policy from one set of Beta parameters
(`BernoulliTS(is_zozotown_prior=True)`), and the benchmark compares estimates of that
policy's value with the BTS arm's on-policy click rate. This script measures how far the
snapshot is from the propensities BTS actually logged, and what that does to the
cleanest estimator, IPW, which needs only those propensities.

    python experiments/bts_policy_audit.py      # writes results/bts_policy_audit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counterfact.data import CAMPAIGNS, load
from counterfact.estimators import IPW, OPEInput
from counterfact.policies import LoggedSlatePolicy, at_logged_position, bernoulli_thompson_sampling, zozotown_prior

HOUR = 3_600_000_000_000


def audit(campaign: str) -> dict:
    bts, rnd = load("bts", campaign), load("random", campaign)
    k = bts.len_list
    key = bts.action * k + bts.position
    order = np.lexsort((bts.timestamp, key))
    sorted_key, sorted_p = key[order], bts.pscore[order]
    distinct = [len(np.unique(sorted_p[sorted_key == j])) for j in range(bts.n_actions * k)]
    changes = int(np.sum((sorted_key[1:] == sorted_key[:-1]) & (sorted_p[1:] != sorted_p[:-1])))

    alpha, beta = zozotown_prior(campaign)
    snapshot = bernoulli_thompson_sampling(alpha, beta, k)  # (n_actions, k)
    replay = LoggedSlatePolicy(bts.timestamp, bts.action, bts.position, bts.pscore, bts.n_actions, k)

    # Total-variation distance between the snapshot and the rebuilt policy, every hour, averaged over positions.
    hours = np.arange(bts.timestamp[0] + HOUR, bts.timestamp[-1], HOUR)
    tv = np.zeros(len(hours))
    for pos in range(k):
        dist, _ = replay.at(hours, np.full(len(hours), pos))
        tv += 0.5 * np.abs(dist - snapshot[:, pos][None, :]).sum(axis=1) / k

    truth = bts.on_policy_value()
    ipw = {}
    for name, pi_e in (
        ("snapshot", at_logged_position(snapshot, rnd.position)),
        ("replayed", replay.at(rnd.timestamp, rnd.position)[0]),
    ):
        data = OPEInput.build(action=rnd.action, reward=rnd.reward, pscore=rnd.pscore, pi_e=pi_e)
        terms = IPW().round_terms(data)
        est = float(terms.mean())
        ipw[name] = {
            "estimate": est,
            "standard_error": float(terms.std(ddof=1) / np.sqrt(data.n)),
            "relative_error": abs(est - truth) / truth,
        }

    # Hourly propensity of the three items BTS showed most at the left slot, for the drift figure.
    left = bts.position == 0
    top = np.argsort(-np.bincount(bts.action[left], minlength=bts.n_actions))[:3]
    hour_index = (bts.timestamp - bts.timestamp[0]) // HOUR
    trace = {}
    for a in top:
        rows = left & (bts.action == a)
        h, p = hour_index[rows], bts.pscore[rows]
        trace[int(a)] = {
            "hour": np.unique(h).tolist(),
            "median_pscore": [float(np.median(p[h == x])) for x in np.unique(h)],
            "snapshot": float(snapshot[a, 0]),
        }

    return {
        "campaign": campaign,
        "bts_rounds": bts.n_rounds,
        "distinct_propensities_per_item_position": {"median": float(np.median(distinct)), "max": int(max(distinct))},
        "propensity_changes": changes,
        "tv_snapshot_vs_logged_hourly": {
            "min": float(tv.min()),
            "median": float(np.median(tv)),
            "max": float(tv.max()),
        },
        "truth": truth,
        "truth_standard_error": float(np.sqrt(truth * (1 - truth) / bts.n_rounds)),
        "ipw_on_full_random_log": ipw,
        "left_slot_trace": trace,
    }


def main() -> None:
    out = {c: audit(c) for c in CAMPAIGNS}
    for c, r in out.items():
        tv = r["tv_snapshot_vs_logged_hourly"]
        ipw = r["ipw_on_full_random_log"]
        print(
            f"{c}: TV median {tv['median']:.3f} (min {tv['min']:.3f}); IPW rel. error snapshot "
            f"{ipw['snapshot']['relative_error']:.3f} vs replayed {ipw['replayed']['relative_error']:.3f} "
            f"(IPW s.e. {ipw['snapshot']['standard_error'] / r['truth']:.3f} of truth)"
        )
    path = ROOT / "results" / "bts_policy_audit.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
