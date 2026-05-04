"""Postprocessing for `snr_definition` — Q1.3 and Q1.4 from the
INSTRUCTIONS.md plan.

Q1.3: For each language, rank SNR variants by Pearson r against DA-size
      (pooled across the 3 small sizes). Pick the best variant.
      Save → best_variant_per_language.csv
Q1.4: Take the global-best variant; for each language, list the top
      benchmarks by SNR at the largest size (1B).
      Save → top_benchmarks_per_language.{csv,png}
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

from multilingual.analyze_snr_variants import (  # noqa: E402
    _per_language_pearson_table, assign_language, da_size_pairs,
    list_variants,
)
from snr.constants import PLOT_DIR  # noqa: E402

OUT_DIR = PLOT_DIR / "snr_definition"
TARGET_SIZE = "1B"
TOP_K = 5


def best_variant_per_language(df: pd.DataFrame) -> pd.DataFrame:
    """variant × language Pearson r table → per-language argmax."""
    variants = list_variants(df)
    pairs = list(da_size_pairs())
    table = _per_language_pearson_table(df, variants, pairs)
    rows = []
    for lang in table.columns:
        col = table[lang].dropna()
        if col.empty:
            continue
        col = col.sort_values(ascending=False)
        best = col.index[0]
        runner_up = col.index[1] if len(col) > 1 else ""
        rows.append({
            "language": lang,
            "best_variant": best,
            "best_pearson_r": float(col.iloc[0]),
            "runner_up_variant": runner_up,
            "runner_up_pearson_r": float(col.iloc[1]) if len(col) > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("language").reset_index(drop=True)


def global_best_variant(df: pd.DataFrame) -> str:
    """Best mean-Pearson-r variant across all languages (DA-size)."""
    variants = list_variants(df)
    pairs = list(da_size_pairs())
    table = _per_language_pearson_table(df, variants, pairs)
    means = table.mean(axis=1, skipna=True).sort_values(ascending=False)
    return str(means.index[0])


def top_benchmarks_per_language(df: pd.DataFrame, variant: str,
                                size: str = TARGET_SIZE,
                                top_k: int = TOP_K) -> pd.DataFrame:
    snr_col = f"snr_{variant}_{size}"
    if snr_col not in df.columns:
        raise KeyError(snr_col)
    df = df.copy()
    df["language"] = [assign_language(t) for t in df.index]
    df = df[df["language"] != "??"]
    rows = []
    for lang, sub in df.groupby("language"):
        sub_sorted = sub.sort_values(snr_col, ascending=False)
        sub_sorted = sub_sorted[sub_sorted[snr_col].notna()]
        for rank, (task, row) in enumerate(sub_sorted.head(top_k).iterrows(), 1):
            rows.append({
                "language": lang,
                "rank": rank,
                "task": task,
                "snr": float(row[snr_col]),
                "decision_acc_size_600M": float(row.get(
                    "decision_acc_size_600M", np.nan)),
            })
    return pd.DataFrame(rows)


def render_top_benchmarks_grid(top_df: pd.DataFrame, variant: str,
                               save_path: Path):
    langs = sorted(top_df["language"].unique())
    n = len(langs)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows),
                             squeeze=False)
    for i, lang in enumerate(langs):
        ax = axes[i // ncols][i % ncols]
        sub = top_df[top_df["language"] == lang].sort_values("snr",
                                                             ascending=True)
        ax.barh(range(len(sub)), sub["snr"], color="#1f77b4", alpha=0.85)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["task"], fontsize=8)
        ax.set_xlabel(f"SNR ({variant} @ {TARGET_SIZE})", fontsize=8)
        ax.set_title(f"{lang}  (top {len(sub)})", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(f"Top-{TOP_K} benchmarks per language by SNR — variant `{variant}` @ {TARGET_SIZE}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    csv_path = OUT_DIR / "snr_variants_per_task.csv"
    df = pd.read_csv(csv_path, index_col="task")

    best_df = best_variant_per_language(df)
    best_path = OUT_DIR / "best_variant_per_language.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Wrote → {best_path}")
    print(best_df.to_string(index=False))

    g_best = global_best_variant(df)
    print(f"\nGlobal best variant (mean Pearson r across languages, DA-size): {g_best}")

    top_df = top_benchmarks_per_language(df, g_best)
    top_path = OUT_DIR / "top_benchmarks_per_language.csv"
    top_df.to_csv(top_path, index=False)
    print(f"\nWrote → {top_path}  ({len(top_df)} rows)")

    plot_path = OUT_DIR / "top_benchmarks_per_language.png"
    render_top_benchmarks_grid(top_df, g_best, plot_path)
    print(f"Wrote → {plot_path}")


if __name__ == "__main__":
    main()
