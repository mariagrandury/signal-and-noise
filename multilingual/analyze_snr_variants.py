"""Render SNR-vs-decision-accuracy scatter grids from snr_variants_per_task.csv.

For every SNR variant in `snr.snr_variants.AGGREGATION_FUNCTIONS`, render a
row of scatter panels (one column per small size, 175M/350M/600M → 1B) and
stack the rows top-to-bottom in descending order of R² with decision accuracy.

Outputs:
  results/snr_definition/snr_vs_decision_accuracy.png      — all benchmarks
  results/snr_definition/snr_vs_decision_accuracy_<lang>.png  — per-language
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from snr.constants import PLOT_DIR
from snr.plot import config_snr_ax, plot_snr_scatter

OUT_DIR = PLOT_DIR / "snr_definition"
CSV_PATH = OUT_DIR / "snr_variants_per_task.csv"
SMALL_SIZES = ["175M", "350M", "600M"]
SIZES = SMALL_SIZES + ["1B"]


# --- language assignment -----------------------------------------------------

_LANG_MAP = {
    "ar": "ar", "arb": "ar",
    "de": "de",
    "es": "es", "spa": "es",
    "eu": "eu", "eus": "eu",
    "fr": "fr",
    "hi": "hi", "hin": "hi",
    "ru": "ru", "rus": "ru",
    "vi": "vi", "vie": "vi",
    "zh": "zh", "zho": "zh", "cmn": "zh",
    "ja": "ja", "jp": "ja", "jpn": "ja",
    "sw": "sw", "swh": "sw",
    "th": "th", "tha": "th",
    "tr": "tr", "tur": "tr",
    "en": "en", "eng": "en",
}

_ENGLISH_ONLY_TASKS = {
    "arc_challenge", "arc_easy", "commonsense_qa", "hellaswag", "mmlu",
    "openbookqa", "piqa", "truthfulqa_mc1",
}

# Capture last lang-code-looking token before any trailing decoration.
_LANG_TOKEN_RE = re.compile(r"_([a-z]{2,4})(?:_|$)")


def assign_language(task: str) -> str:
    if task in _ENGLISH_ONLY_TASKS:
        return "en"
    # Walk tokens left-to-right, take the first that matches a known code.
    for tok in task.split("_"):
        if tok in _LANG_MAP:
            return _LANG_MAP[tok]
    return "??"


def benchmark_family(task: str) -> str:
    """Strip language/script suffix to get the benchmark family (arc, belebele, ...)."""
    if task in _ENGLISH_ONLY_TASKS:
        # arc_challenge, arc_easy → "arc"; mmlu → "mmlu"; etc.
        return task.split("_")[0] if task != "commonsense_qa" else "csqa"
    parts = task.split("_")
    out = []
    for p in parts:
        if p in _LANG_MAP:
            break
        out.append(p)
    fam = "_".join(out) if out else parts[0]
    # Trim trailing modifiers we don't want as part of the family name.
    fam = re.sub(r"_(mc1|completions|full)$", "", fam)
    return fam


# --- variant / column helpers -----------------------------------------------

def list_variants(df: pd.DataFrame) -> list[str]:
    variants = set()
    for col in df.columns:
        m = re.match(r"^snr_(.+)_([0-9]+[MB])$", col)
        if m:
            variants.add(m.group(1))
    return sorted(variants)


def snr_col(variant: str, size: str) -> str:
    return f"snr_{variant}_{size}"


# --- analyses ---------------------------------------------------------------

def variant_decision_acc_correlation(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """For each variant, compute R^2 between log10(SNR) and decision_acc,
    pooled across (task, small_size) pairs (matches the analysis in
    analysis/snr_variants.ipynb)."""
    rows = []
    for variant in variants:
        xs, ys = [], []
        for size in SMALL_SIZES:
            snr = df.get(snr_col(variant, size))
            da = df.get(f"decision_acc_{size}")
            if snr is None or da is None:
                continue
            both = pd.concat([snr, da], axis=1).dropna()
            both = both[both.iloc[:, 0] > 0]  # log10 needs positive
            xs.extend(np.log10(both.iloc[:, 0].to_numpy()))
            ys.extend(both.iloc[:, 1].to_numpy())
        if len(xs) < 3:
            rows.append({"variant": variant, "n": len(xs),
                         "pearson_r": np.nan, "r2": np.nan, "spearman_r": np.nan})
            continue
        xs_a = np.asarray(xs); ys_a = np.asarray(ys)
        if np.std(xs_a) == 0 or np.std(ys_a) == 0:
            r = np.nan; rho = np.nan
        else:
            r = float(np.corrcoef(xs_a, ys_a)[0, 1])
            # Spearman: corr of ranks.
            rho = float(np.corrcoef(pd.Series(xs_a).rank(), pd.Series(ys_a).rank())[0, 1])
        rows.append({
            "variant": variant, "n": len(xs),
            "pearson_r": r,
            "r2": (r * r) if r is not None and np.isfinite(r) else np.nan,
            "spearman_r": rho,
        })
    return pd.DataFrame(rows).sort_values("r2", ascending=False)


# --- plotting ---------------------------------------------------------------

def _ranked_variants(df: pd.DataFrame, variants: list[str]) -> list[tuple[str, float]]:
    corr = variant_decision_acc_correlation(df, variants)
    return [(row["variant"], row["r2"]) for _, row in corr.iterrows()]


def render_grid(df: pd.DataFrame, variants_ranked: list[tuple[str, float]],
                save_path: Path, title: str) -> bool:
    """Stack one row of (size) scatter panels per SNR variant; rows ordered by R²."""
    n_rows = len(variants_ranked)
    n_cols = len(SMALL_SIZES)
    if n_rows == 0:
        return False
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows), squeeze=False,
    )
    for r, (variant, r2) in enumerate(variants_ranked):
        for c, size in enumerate(SMALL_SIZES):
            ax = axes[r][c]
            snr_c = snr_col(variant, size)
            da_c = f"decision_acc_{size}"
            if snr_c not in df.columns or da_c not in df.columns:
                ax.set_visible(False)
                continue
            sub = df[[snr_c, da_c]].dropna()
            sub = sub[sub[snr_c] > 0]
            if sub.empty:
                ax.set_visible(False)
                continue
            x = sub[snr_c].to_numpy()
            y = sub[da_c].to_numpy()
            texts = plot_snr_scatter(ax, x, y, sub.index.tolist(),
                                     size=size, task_names={})
            plot_fit = len(sub) >= 3
            config_snr_ax(ax, x, y, texts, xlabel=f"SNR {variant} ({size})",
                          plot_fit=plot_fit, log_scale=True)
            r2_label = "" if not np.isfinite(r2) else f"  (overall R²={r2:.3f})"
            ax.set_title(f"{variant} — {size}{r2_label} (n={len(sub)})", fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return True


# --- driver -----------------------------------------------------------------

def main():
    df = pd.read_csv(CSV_PATH, index_col="task")
    variants = list_variants(df)
    print(f"Loaded {len(df)} tasks × {df.shape[1]} columns "
          f"({len(variants)} variants × {len(SIZES)} sizes "
          f"+ {len(SMALL_SIZES)} decision-acc columns)\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Global figure: all benchmarks, variants ordered by overall R².
    ranked = _ranked_variants(df, variants)
    render_grid(df, ranked, OUT_DIR / "snr_vs_decision_accuracy.png",
                title="SNR vs decision accuracy — all benchmarks "
                      "(variants ordered by R²)")
    print(f"Wrote → {OUT_DIR / 'snr_vs_decision_accuracy.png'}")

    # Per-language figures.
    df_lang = df.copy()
    df_lang["language"] = [assign_language(t) for t in df_lang.index]
    for lang, sub in sorted(df_lang.groupby("language")):
        sub = sub.drop(columns=["language"])
        if len(sub) < 2:
            continue
        ranked_lang = _ranked_variants(sub, variants)
        path = OUT_DIR / f"snr_vs_decision_accuracy_{lang}.png"
        ok = render_grid(sub, ranked_lang, path,
                         title=f"SNR vs decision accuracy — {lang} "
                               f"({len(sub)} benchmarks; variants ordered by R²)")
        if ok:
            print(f"Wrote → {path}")


if __name__ == "__main__":
    main()
