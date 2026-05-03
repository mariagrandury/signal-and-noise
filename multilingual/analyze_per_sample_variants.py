"""Compare per-sample subset-selection strategies (A/B/C/D/E).

Reads ``results/smooth_subtasks/per_sample_variants/summary_all.csv``
(produced by ``run_per_sample_variants.py``) and emits:

- ``by_strategy_summary.csv``  — overall mean/median best SNR per strategy.
- ``by_strategy_per_size.csv`` — same broken out by ckpt size.
- ``win_rates.csv``            — for each (size, strategy) pair, the
                                 fraction of (lang, task) where the
                                 strategy strictly beats Option D.
- ``best_snr_box.png``         — distribution of best SNR per strategy
                                 across all (lang, task, size).
- ``delta_vs_d_hist.png``      — histogram of (best_snr_strat - best_snr_D)
                                 per non-D strategy.
- ``best_frac_violin.png``     — distribution of best_n / n_total per
                                 strategy (smaller = more aggressive
                                 prune).
- ``per_language_winners.csv`` — for each language, the strategy with
                                 the highest mean best_snr.

Run after ``run_per_sample_variants.py`` finishes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from snr.constants import PLOT_DIR

ROOT = PLOT_DIR / "smooth_subtasks" / "per_sample_variants"
SUMMARY_PATH = ROOT / "summary_all.csv"
STRATEGY_ORDER = ["A", "B", "C", "D", "E"]
STRATEGY_COLOURS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green",
                    "D": "tab:red", "E": "tab:purple"}


def _load_ok(summary_path: Path = SUMMARY_PATH) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} missing — run multilingual.run_per_sample_variants first."
        )
    df = pd.read_csv(summary_path)
    df = df[df["status"] == "ok"].copy()
    for c in ("best_snr", "full_set_snr", "snr_gain", "best_frac",
              "n_total", "n_candidates", "best_n"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def by_strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("strategy")
              .agg(mean_best_snr=("best_snr", "mean"),
                   median_best_snr=("best_snr", "median"),
                   mean_full_set_snr=("full_set_snr", "mean"),
                   mean_snr_gain=("snr_gain", "mean"),
                   median_snr_gain=("snr_gain", "median"),
                   mean_best_frac=("best_frac", "mean"),
                   median_best_frac=("best_frac", "median"),
                   n_runs=("best_snr", "size"))
              .reindex(STRATEGY_ORDER)
              .reset_index())


def by_strategy_per_size(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby(["size", "strategy"])
             .agg(mean_best_snr=("best_snr", "mean"),
                  median_best_snr=("best_snr", "median"),
                  mean_snr_gain=("snr_gain", "mean"),
                  mean_best_frac=("best_frac", "mean"),
                  n_runs=("best_snr", "size"))
             .reset_index())
    return out.sort_values(["size", "strategy"])


def win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """For each (size, strategy ≠ D), share of (lang, task) where strat > D."""
    keys = ["language", "task", "size"]
    d = df[df["strategy"] == "D"][keys + ["best_snr"]].rename(
        columns={"best_snr": "best_snr_D"})
    j = df.merge(d, on=keys, how="left")
    j["wins_D"] = j["best_snr"] > j["best_snr_D"]
    j["ties_D"] = np.isclose(j["best_snr"], j["best_snr_D"], equal_nan=False)

    rows = []
    for (size, strat), g in j[j["strategy"] != "D"].groupby(["size", "strategy"]):
        rows.append({
            "size": size,
            "strategy": strat,
            "n": len(g),
            "win_rate_vs_D": float(g["wins_D"].mean()),
            "tie_rate_vs_D": float(g["ties_D"].mean()),
            "mean_delta_vs_D": float((g["best_snr"] - g["best_snr_D"]).mean()),
            "median_delta_vs_D": float((g["best_snr"] - g["best_snr_D"]).median()),
        })
    return pd.DataFrame(rows).sort_values(["size", "strategy"])


def per_language_winners(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["language", "strategy"])
           .agg(mean_best_snr=("best_snr", "mean"),
                n_runs=("best_snr", "size"))
           .reset_index())
    rows = []
    for lang, sub in g.groupby("language"):
        sub = sub.sort_values("mean_best_snr", ascending=False)
        winner = sub.iloc[0]
        rows.append({
            "language": lang,
            "winner_strategy": winner["strategy"],
            "winner_mean_best_snr": float(winner["mean_best_snr"]),
            "runner_up": (sub.iloc[1]["strategy"] if len(sub) >= 2 else ""),
            "runner_up_mean_best_snr": (float(sub.iloc[1]["mean_best_snr"])
                                        if len(sub) >= 2 else float("nan")),
            "n_tasks": int(winner["n_runs"]),
        })
    return pd.DataFrame(rows).sort_values("language")


def plot_best_snr_box(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [df.loc[df["strategy"] == s, "best_snr"].dropna().values
            for s in STRATEGY_ORDER]
    bp = ax.boxplot(data, labels=STRATEGY_ORDER, showmeans=True, widths=0.55)
    for i, s in enumerate(STRATEGY_ORDER):
        for patch in bp["boxes"]:
            pass  # default style is fine
    ax.set_ylabel("Best SNR")
    ax.set_xlabel("Strategy")
    ax.set_title("Distribution of best SNR achieved per strategy")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_delta_vs_d_hist(df: pd.DataFrame, save_path: Path):
    keys = ["language", "task", "size"]
    d = df[df["strategy"] == "D"][keys + ["best_snr"]].rename(
        columns={"best_snr": "best_snr_D"})
    j = df.merge(d, on=keys, how="left")
    j["delta"] = j["best_snr"] - j["best_snr_D"]

    others = [s for s in STRATEGY_ORDER if s != "D"]
    fig, axes = plt.subplots(1, len(others), figsize=(3.2 * len(others), 3.4),
                             sharey=True)
    if len(others) == 1:
        axes = [axes]
    for ax, s in zip(axes, others):
        deltas = j.loc[j["strategy"] == s, "delta"].dropna().values
        if deltas.size == 0:
            ax.set_title(f"{s}\n(no data)")
            continue
        ax.hist(deltas, bins=40, color=STRATEGY_COLOURS[s], alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        med = float(np.median(deltas))
        mean = float(np.mean(deltas))
        ax.set_title(f"{s}  median Δ {med:+.3f}\n(mean {mean:+.3f}, "
                     f"n={deltas.size})", fontsize=9)
        ax.set_xlabel("best_snr - best_snr_D")
    axes[0].set_ylabel("count")
    fig.suptitle("Per-strategy uplift over Option D", fontsize=11)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_best_frac_violin(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [df.loc[df["strategy"] == s, "best_frac"].dropna().values
            for s in STRATEGY_ORDER]
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.7)
    ax.set_xticks(range(1, len(STRATEGY_ORDER) + 1))
    ax.set_xticklabels(STRATEGY_ORDER)
    ax.set_ylabel("best_n / n_total")
    ax.set_xlabel("Strategy")
    ax.set_title("Subset size at the SNR peak (smaller = more aggressive prune)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(root: Path = ROOT):
    summary_path = root / "summary_all.csv"
    df = _load_ok(summary_path)
    if df.empty:
        print(f"No 'ok' rows in {summary_path} — nothing to analyse.")
        return

    summary = by_strategy_summary(df)
    per_size = by_strategy_per_size(df)
    wins = win_rates(df)
    lang_winners = per_language_winners(df)

    summary.to_csv(root / "by_strategy_summary.csv", index=False)
    per_size.to_csv(root / "by_strategy_per_size.csv", index=False)
    wins.to_csv(root / "win_rates.csv", index=False)
    lang_winners.to_csv(root / "per_language_winners.csv", index=False)

    plot_best_snr_box(df, root / "best_snr_box.png")
    plot_delta_vs_d_hist(df, root / "delta_vs_d_hist.png")
    plot_best_frac_violin(df, root / "best_frac_violin.png")

    print("=== Per-strategy summary ===")
    print(summary.to_string(index=False))
    print("\n=== Win rate vs Option D ===")
    print(wins.to_string(index=False))
    print("\n=== Per-language winning strategy ===")
    print(lang_winners.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Directory containing summary_all.csv to analyse.")
    args = parser.parse_args()
    main(root=args.root)
