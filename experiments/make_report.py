"""Build figures (results/figures/*.svg) and tables (results/RESULTS.md) from the result JSONs.

python experiments/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Reference data-viz palette (validated: first three slots pass all-pairs CVD checks on this surface).
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FAMILY_STYLE = {"IPW": (BLUE, "o"), "DM": (ORANGE, "s"), "DR": (AQUA, "^")}
ESTIMATOR_ORDER = ["DM", "IPW", "SNIPW", "DR", "SNDR", "Switch-DR (tau 100)", "DRos (lambda 100)"]
CAMPAIGNS = ["all", "men", "women"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": MUTED,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "svg.fonttype": "none",
    }
)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _plain_log_axes(ax, x: bool = True, y: bool = True) -> None:
    """Label log axes with plain numbers (0.02, 4k) instead of powers of ten."""
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    for on, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if on:
            axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


# --- Open Bandit Dataset ---------------------------------------------------------------


def obd_results(tag: str) -> dict[tuple[str, str], dict]:
    out = {}
    for path in sorted((RESULTS / "obd").glob(f"{tag}_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        out[(r["task"]["campaign"], r["task"]["direction"])] = r
    return out


def obd_table(results: dict[tuple[str, str], dict], title: str) -> list[str]:
    tasks = [(c, d) for d in ("random_to_bts", "bts_to_random") for c in CAMPAIGNS if (c, d) in results]
    if not tasks:
        return []
    header = (
        "| Estimator | " + " | ".join(f"{c}: {d.replace('_to_', ' → ').replace('bts', 'BTS')}" for c, d in tasks) + " |"
    )
    lines = [
        f"**{title}** — mean relative error ± std over bootstrap runs (lower is better; **bold** = best per column)",
        "",
        header,
    ]
    lines.append("|---|" + "---|" * len(tasks))
    best = {
        t: min(results[t]["summary"].values(), key=lambda s: s["relative_error_mean"])["relative_error_mean"]
        for t in tasks
    }
    for name in ESTIMATOR_ORDER:
        cells = []
        for t in tasks:
            s = results[t]["summary"][name]
            cell = f"{_pct(s['relative_error_mean'])} ± {_pct(s['relative_error_std'])}"
            cells.append(f"**{cell}**" if s["relative_error_mean"] == best[t] else cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    noise = [_pct(2 * results[t]["truth_se"] / results[t]["truth"]) for t in tasks]
    lines.append("| *ground-truth noise (2 s.e.)* | " + " | ".join(noise) + " |")
    lines.append("")
    r0 = results[tasks[0]]
    lines.append(
        f"{r0['n_runs']} bootstrap runs per column, reward model `{r0['reward_model']['label']}` with {r0['n_folds']}-fold cross-fitting; "
        f"rows per run: "
        + ", ".join(f"{c}/{d.split('_')[0]} {results[(c, d)]['sample_size']:,}" for c, d in tasks)
        + "."
    )
    lines.append("")
    return lines


def obd_figure(results: dict[tuple[str, str], dict]) -> None:
    directions = [d for d in ("random_to_bts", "bts_to_random") if any((c, d) in results for c in CAMPAIGNS)]
    if not directions:
        return
    fig, axes = plt.subplots(len(directions), 3, figsize=(11, 2.9 * len(directions)), sharey=True, squeeze=False)
    for i, d in enumerate(directions):
        xmax = max(
            s["relative_error_mean"] + s["relative_error_std"]
            for c in CAMPAIGNS
            if (c, d) in results
            for s in results[(c, d)]["summary"].values()
        )
        for j, c in enumerate(CAMPAIGNS):
            ax = axes[i][j]
            if (c, d) not in results:
                ax.set_visible(False)
                continue
            r = results[(c, d)]
            noise = 2 * r["truth_se"] / r["truth"]
            ax.axvspan(0, noise, color=GRID, lw=0)
            ys = range(len(ESTIMATOR_ORDER))
            means = [r["summary"][n]["relative_error_mean"] for n in ESTIMATOR_ORDER]
            stds = [r["summary"][n]["relative_error_std"] for n in ESTIMATOR_ORDER]
            ax.errorbar(means, ys, xerr=stds, fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.5, capsize=0, markersize=6)
            best = min(range(len(means)), key=lambda k: means[k])
            ax.annotate(
                _pct(means[best]),
                (means[best], best),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
            ax.set_xlim(0, xmax * 1.05)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{c} · {d.replace('_to_', ' → ').replace('bts', 'BTS')}", fontsize=10, loc="left", color=INK)
            if j == 0:
                ax.set_yticks(
                    list(ys),
                    [n.replace(" (tau 100)", " τ=100").replace(" (lambda 100)", " λ=100") for n in ESTIMATOR_ORDER],
                )
    axes[0][0].invert_yaxis()  # y is shared by every panel, so one call flips them all
    fig.supxlabel("relative error |V̂ − V| / V   (shaded: ground-truth noise, 2 s.e.)", color=INK_2, fontsize=9)
    fig.tight_layout()
    _save(fig, "obd_relative_error.svg")


def audit_section(audit: dict) -> list[str]:
    lines = [
        "| Campaign | distinct logged propensities per (item, slot), median / max | TV(snapshot, logged policy) hourly, min / median / max | IPW rel. error, snapshot target | IPW rel. error, replayed target | IPW s.e. / V |",
        "|---|---|---|---|---|---|",
    ]
    for c in CAMPAIGNS:
        r = audit[c]
        d, tv, ipw = (
            r["distinct_propensities_per_item_position"],
            r["tv_snapshot_vs_logged_hourly"],
            r["ipw_on_full_random_log"],
        )
        lines.append(
            f"| {c} | {d['median']:.0f} / {d['max']} | {tv['min']:.2f} / {tv['median']:.2f} / {tv['max']:.2f} | "
            f"{_pct(ipw['snapshot']['relative_error'])} | {_pct(ipw['replayed']['relative_error'])} | "
            f"{_pct(ipw['snapshot']['standard_error'] / r['truth'])} |"
        )
    return lines + [""]


def audit_figure(audit: dict, campaign: str = "men") -> None:
    trace = audit[campaign]["left_slot_trace"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for (item, t), color in zip(trace.items(), (BLUE, ORANGE, AQUA), strict=False):
        days = [h / 24 for h in t["hour"]]
        ax.plot(days, t["median_pscore"], color=color, lw=1.6)
        ax.axhline(t["snapshot"], color=color, lw=1.2, ls=(0, (4, 3)))
        ax.annotate(
            f"item {item}",
            (days[-1], t["median_pscore"][-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlabel("day of the logging week")
    ax.set_ylabel("P(item in left slot)")
    ax.set_xlim(0, 7)
    ax.set_title(
        f"BTS propensities as logged (solid, hourly median) vs obp's production snapshot (dashed) — '{campaign}' campaign",
        fontsize=10,
        loc="left",
    )
    _save(fig, "bts_drift.svg")


# --- Synthetic --------------------------------------------------------------------------


def _family(name: str) -> str | None:
    return {"IPW": "IPW", "DM": "DM", "DR": "DR"}.get(name)


def synthetic_sample_size(data: dict) -> list[str]:
    rows = data["results"]
    sizes = sorted({r["setting"][1] for r in rows})

    def get(estimator: str, model: str, n: int) -> dict:
        return next(
            r for r in rows if r["estimator"] == estimator and r["setting"][1] == n and r["setting"][2] == model
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, model, title in (
        (axes[0], "logistic", "RMSE · logistic reward model (misspecified)"),
        (axes[1], "mlp", "RMSE · MLP reward model"),
    ):
        for est in ("IPW", "DM", "DR"):
            color, marker = FAMILY_STYLE[est]
            m = "none" if est == "IPW" else model
            y = [get(est, m, n)["rmse"] for n in sizes]
            ax.plot(sizes, y, color=color, marker=marker, label=est)
            ax.annotate(
                est, (sizes[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log_axes(ax)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("logged rounds n")
    axes[0].legend(frameon=False, loc="lower left")
    ax = axes[2]
    for est in ("DM", "DR"):
        color, marker = FAMILY_STYLE[est]
        y = [abs(get(est, "logistic", n)["bias"]) for n in sizes]
        ax.plot(sizes, y, color=color, marker=marker, label=est)
        ax.annotate(
            est, (sizes[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK
        )
    ax.set_xscale("log")
    _plain_log_axes(ax, y=False)
    ax.set_title("|bias| · logistic reward model", fontsize=10, loc="left")
    ax.set_xlabel("logged rounds n")
    fig.tight_layout()
    _save(fig, "synthetic_sample_size.svg")

    names = ["IPW", "SNIPW", "DM", "DR", "SNDR", "Switch-DR (tau 100)", "DRos (lambda 100)"]
    picks = [n for n in (250, 1000, 4000, 16000) if n in sizes]
    lines = [
        "| Estimator (reward model) | " + " | ".join(f"n = {n:,}" for n in picks) + " |",
        "|---|" + "---|" * len(picks),
    ]
    for est in names:
        models = ["none"] if est in ("IPW", "SNIPW") else ["logistic", "mlp"]
        for m in models:
            label = est if m == "none" else f"{est} ({m})"
            cells = [f"{get(est, m, n)['rmse']:.4f} ({get(est, m, n)['bias']:+.4f})" for n in picks]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"RMSE (bias) over {data['config']['n_runs']} independent datasets; true value V = {data['truth']:.4f}.",
        "",
    ]
    return lines


def synthetic_overlap(data: dict) -> list[str]:
    rows = data["results"]
    eps = sorted({r["setting"][0] for r in rows})

    def get(estimator: str, e: float) -> dict:
        return next(r for r in rows if r["estimator"] == estimator and r["setting"][0] == e)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for est in ("IPW", "DM", "DR"):
        color, marker = FAMILY_STYLE[est]
        y = [get(est, e)["rmse"] for e in eps]
        ax.plot(eps, y, color=color, marker=marker, label=est)
    ax.set_xlabel("ε of the ε-greedy evaluation policy (0 = deterministic, 1 = uniform)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Error vs. overlap · n = {data['n']:,}, MLP reward model", fontsize=10, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "synthetic_overlap.svg")

    names = ["IPW", "SNIPW", "DM", "DR", "SNDR", "Switch-DR (tau 100)", "DRos (lambda 100)"]
    lines = ["| ε | median max weight | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for e in eps:
        best = min(get(n, e)["rmse"] for n in names)
        cells = [
            (f"**{get(n, e)['rmse']:.4f}**" if get(n, e)["rmse"] == best else f"{get(n, e)['rmse']:.4f}") for n in names
        ]
        lines.append(f"| {e:g} | {get('_max_weight', e)['median']:.0f} | " + " | ".join(cells) + " |")
    return lines + ["", "RMSE; **bold** = best per row.", ""]


def synthetic_clipping(data: dict) -> list[str]:
    rows = data["results"]

    def series(family: str) -> tuple[list[float], list[dict]]:
        pts = sorted(
            (r for r in rows if r["setting"][0] == family and r["setting"][1] != "inf"), key=lambda r: r["setting"][1]
        )
        return [r["setting"][1] for r in pts], pts

    unclipped = {f: next(r for r in rows if r["setting"][0] == f and r["setting"][1] == "inf") for f in ("IPW", "DR")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, family in zip(axes, ("IPW", "DR"), strict=True):
        xs, pts = series(family)
        for key, color, marker, label in (
            ("bias", ORANGE, "s", "bias²"),
            ("variance", BLUE, "o", "variance"),
            ("mse", AQUA, "^", "MSE"),
        ):
            y = [p["bias"] ** 2 if key == "bias" else p[key] for p in pts]
            ax.plot(xs, y, color=color, marker=marker, label=label)
        ax.axhline(unclipped[family]["mse"], color=MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.annotate(
            "MSE without clipping",
            (xs[0], unclipped[family]["mse"]),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log_axes(ax)
        ax.set_xlabel("clipping threshold λ (weights capped at λ)")
        ax.set_title(f"{family} with clipped weights · ε = 0, n = {data['n']:,}", fontsize=10, loc="left")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "synthetic_clipping.svg")

    lines = ["| Estimator | best λ | MSE at best λ | MSE unclipped | reduction |", "|---|---|---|---|---|"]
    for family in ("IPW", "DR", "Switch-DR", "DRos"):
        inf = next(r for r in rows if r["setting"][0] == family and r["setting"][1] == "inf")
        best = min([*series(family)[1], inf], key=lambda p: p["mse"])
        lam = "∞ (none)" if best is inf else f"{best['setting'][1]:g}"
        lines.append(
            f"| {family} | {lam} | {best['mse']:.2e} | {inf['mse']:.2e} | {1 - best['mse'] / inf['mse']:.0%} |"
        )
    dm = next(r for r in rows if r["setting"][0] == "DM")
    lines += ["", f"DM (no weights) MSE {dm['mse']:.2e}; true value V = {data['truth']:.4f}.", ""]
    return lines


def main() -> None:
    md = [
        "# Results",
        "",
        "Generated by `experiments/make_report.py` from the JSON files in this directory. Do not edit by hand.",
        "",
    ]

    main_results = obd_results("main")
    if main_results:
        md += ["## Open Bandit Dataset", ""] + obd_table(
            main_results, "Main benchmark (snapshot BTS policy, as in the paper)"
        )
        obd_figure(main_results)
    target = obd_results("target")
    if target:
        md += obd_table(
            target, "Ablation: evaluate BTS as it actually ran (policy replayed from its logged propensities)"
        )
    logistic = obd_results("logistic")
    if logistic:
        md += obd_table(logistic, "Ablation: logistic reward model instead of the MLP")
    itemids = obd_results("itemids")
    if itemids:
        md += obd_table(itemids, "Ablation: reward model also sees a one-hot item id")

    audit = _load(RESULTS / "bts_policy_audit.json")
    if audit:
        md += ["## Audit of the BTS evaluation policy", ""] + audit_section(audit)
        audit_figure(audit)

    for study, fn, title in (
        ("sample_size", synthetic_sample_size, "Synthetic: error vs. sample size (ε = 0.3)"),
        ("overlap", synthetic_overlap, "Synthetic: error vs. overlap"),
        ("clipping", synthetic_clipping, "Synthetic: weight clipping"),
    ):
        data = _load(RESULTS / "synthetic" / f"{study}.json")
        if data:
            md += [f"## {title}", ""] + fn(data)

    (RESULTS / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(
        "wrote results/RESULTS.md and",
        ", ".join(sorted(p.name for p in FIGURES.glob("*.svg"))) if FIGURES.exists() else "",
    )


if __name__ == "__main__":
    main()
