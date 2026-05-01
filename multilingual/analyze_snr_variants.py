"""Analyze the per-task SNR-variant CSV produced by run_apertus_snr_variants.py.

Answers:
  1. Per language: which benchmark has the highest SNR, under which variant?
  2. Across the whole table: which SNR variant correlates best with
     decision accuracy (R^2 of log10(SNR) vs decision_acc, pooled across
     small sizes 175M/350M/600M and tasks)?
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from snr.constants import PLOT_DIR

OUT_DIR = PLOT_DIR
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

def per_language_best(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """For each (language, size): the (task, variant, snr) with the max SNR."""
    df = df.copy()
    df["language"] = [assign_language(t) for t in df.index]
    df["family"] = [benchmark_family(t) for t in df.index]

    rows = []
    for lang, sub in df.groupby("language"):
        for size in SIZES:
            best = {"language": lang, "size": size, "task": None,
                    "family": None, "variant": None, "snr": np.nan}
            for variant in variants:
                col = snr_col(variant, size)
                if col not in sub.columns:
                    continue
                vals = sub[col]
                vals = vals[np.isfinite(vals)]
                if vals.empty:
                    continue
                idx = vals.idxmax()
                v = vals.loc[idx]
                if not np.isfinite(best["snr"]) or v > best["snr"]:
                    best.update(
                        task=idx, family=sub.loc[idx, "family"],
                        variant=variant, snr=float(v),
                    )
            rows.append(best)
    return pd.DataFrame(rows)


def per_language_best_variant_summary(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """For each language, pick the variant whose mean-across-tasks SNR (averaged
    over small sizes) is highest. Provides a 'this variant separates mixes most
    consistently for this language' view, complementing the per-task max above."""
    df = df.copy()
    df["language"] = [assign_language(t) for t in df.index]
    rows = []
    for lang, sub in df.groupby("language"):
        scores = {}
        for variant in variants:
            cols = [snr_col(variant, s) for s in SMALL_SIZES if snr_col(variant, s) in sub.columns]
            if not cols:
                continue
            vals = sub[cols].to_numpy().ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            scores[variant] = float(np.mean(vals))
        if not scores:
            continue
        best_variant = max(scores, key=scores.get)
        rows.append({
            "language": lang,
            "best_mean_variant": best_variant,
            "mean_snr": scores[best_variant],
            "n_tasks": int(sub.shape[0]),
        })
    return pd.DataFrame(rows).sort_values("mean_snr", ascending=False)


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


def per_language_correlation(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """Best variant by R^2(SNR, decision_acc) within each language."""
    df = df.copy()
    df["language"] = [assign_language(t) for t in df.index]
    out = []
    for lang, sub in df.groupby("language"):
        sub_no_lang = sub.drop(columns=["language"])
        corr = variant_decision_acc_correlation(sub_no_lang, variants)
        if corr.empty:
            continue
        top = corr.iloc[0].to_dict()
        out.append({
            "language": lang, "n_tasks": int(sub.shape[0]),
            "best_variant_by_r2": top["variant"],
            "r2": top["r2"],
            "pearson_r": top["pearson_r"],
            "spearman_r": top["spearman_r"],
        })
    return pd.DataFrame(out).sort_values("r2", ascending=False)


# --- driver -----------------------------------------------------------------

def main():
    df = pd.read_csv(CSV_PATH, index_col="task")
    variants = list_variants(df)
    print(f"Loaded {len(df)} tasks × {df.shape[1]} columns "
          f"({len(variants)} variants × {len(SIZES)} sizes "
          f"+ {len(SMALL_SIZES)} decision-acc columns)\n")

    languages = sorted({assign_language(t) for t in df.index})
    print("Tasks per language:")
    lang_counts = pd.Series([assign_language(t) for t in df.index]).value_counts()
    print(lang_counts.to_string(), "\n")

    # 1a) For each (language, size), the single highest SNR cell.
    best = per_language_best(df, variants)
    best_pivot_tasks = best.pivot(index="language", columns="size", values="task")
    best_pivot_variants = best.pivot(index="language", columns="size", values="variant")
    best_pivot_snr = best.pivot(index="language", columns="size", values="snr")
    print("=== Highest-SNR benchmark per language and size ===")
    print("(task)")
    print(best_pivot_tasks[SIZES].to_string(), "\n")
    print("(variant)")
    print(best_pivot_variants[SIZES].to_string(), "\n")
    print("(SNR value)")
    print(best_pivot_snr[SIZES].round(2).to_string(), "\n")

    # 1b) Per-language summary: variant whose average SNR-across-tasks is highest.
    summary = per_language_best_variant_summary(df, variants)
    print("=== Per language: variant with highest mean SNR (avg over small sizes & tasks) ===")
    print(summary.to_string(index=False), "\n")

    # 2a) Global ranking of variants by R^2 with decision accuracy.
    corr = variant_decision_acc_correlation(df, variants)
    print("=== Variants ranked by R² of log10(SNR) vs decision accuracy "
          "(pooled across tasks × {175M,350M,600M}) ===")
    print(corr.to_string(index=False), "\n")

    # 2b) Per-language: best variant by R^2.
    corr_per_lang = per_language_correlation(df, variants)
    print("=== Per language: best variant by R² with decision accuracy ===")
    print(corr_per_lang.to_string(index=False), "\n")

    # Persist tables.
    best.to_csv(OUT_DIR / "snr_variant_per_language_best.csv", index=False)
    summary.to_csv(OUT_DIR / "snr_variant_per_language_best_variant.csv", index=False)
    corr.to_csv(OUT_DIR / "snr_variant_vs_decision_acc.csv", index=False)
    corr_per_lang.to_csv(OUT_DIR / "snr_variant_per_language_best_variant_by_r2.csv", index=False)
    print(f"Wrote 4 analysis CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
