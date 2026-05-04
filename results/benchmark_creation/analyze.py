"""Per-family SNR vs benchmark-creation metadata analysis.

Reads:
  - ../snr_definition/snr_variants_per_task.csv  (Q1 output)
  - data_info.md                                 (this dir's metadata table)

Writes (this dir):
  - per_family_snr.csv             one row per family with SNR aggregates
                                   + curation/source category labels
  - per_task_snr.csv               one row per per-language aggregate task,
                                   carrying the per-task curation override
                                   (xnli_eu, paws_eu, xcopa_eu)
  - snr_by_curation_process.png    strip plot of family SNR by curation cat
  - snr_by_data_source.png         strip plot of family SNR by source-origin
  - snr_by_curation_per_task.png   per-task strip plot (catches the xnli_eu
                                   heterogeneity that family-level smears)
  - group_stats.csv                per-group n, mean, median, kruskal H, p

Q1's headline pick is `mpd` (mean pairwise distance, dispersion cluster);
SNR signal here is `snr_mpd_1B`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Reach the multilingual helpers without writing outside this dir.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multilingual.analyze_snr_variants import assign_language, benchmark_family  # noqa: E402
from multilingual.smooth_subtasks import _is_language_aggregate  # noqa: E402

HERE = Path(__file__).resolve().parent
SNR_CSV = ROOT / "results" / "snr_definition" / "snr_variants_per_task.csv"
SNR_COL = "snr_mpd_1B"

# --- Categorical labels for grouping -----------------------------------------
# Keep these in sync with the per-family paragraphs in data_info.md. Two
# views:
#   curation_category — how items in the per-language eval set were produced.
#   source_origin     — whether the source benchmark was English-only and
#                       translated, or originally multilingual / aggregated.
FAMILY_META: dict[str, dict[str, str]] = {
    "arc": {
        "data_source": "ARC (Clark et al. 2018), Okapi-translated",
        "curation_process": "machine translation by ChatGPT",
        "curation_category": "machine_translation",
        "source_origin": "english_translated",
    },
    "belebele": {
        "data_source": "FLORES-200 passages, custom MRC questions",
        "curation_process": "human translation by bilingual experts",
        "curation_category": "human_translation",
        # FLORES-200 passages are sourced from English news; the questions
        # are written in parallel by experts. Treat as english_translated
        # (with very high curation quality) rather than native-multilingual.
        "source_origin": "english_translated",
    },
    "global_mmlu": {
        "data_source": "MMLU (Hendrycks et al. 2021), Cohere Lite-style",
        "curation_process": "professional human translation + post-editing",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
    },
    "global_mmlu_full": {
        "data_source": "MMLU (Hendrycks et al. 2021), Cohere Full",
        "curation_process": "machine translation + crowd / expert post-editing",
        "curation_category": "mt_post_edited",
        "source_origin": "english_translated",
    },
    "global_piqa_completions": {
        "data_source": "originally-multilingual native authoring (Arnett 2025)",
        "curation_process": "participatory native-speaker authoring (no translation)",
        "curation_category": "originally_multilingual",
        "source_origin": "originally_multilingual",
    },
    "hellaswag": {
        "data_source": "HellaSwag (Zellers et al. 2019), Okapi-translated",
        "curation_process": "machine translation by ChatGPT",
        "curation_category": "machine_translation",
        "source_origin": "english_translated",
    },
    "multiblimp": {
        "data_source": "Universal Dependencies + UniMorph (Jumelet 2025)",
        "curation_process": "template-based automatic generation from UD/UniMorph",
        "curation_category": "template_generated",
        "source_origin": "originally_multilingual",
    },
    "paws": {
        "data_source": "PAWS (Zhang 2019); PAWS-X + HiTZ/PAWS-eu",
        "curation_process": "professional human translation (mixed sources for eu)",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
    },
    "xcopa": {
        "data_source": "COPA (Roemmele 2011); XCOPA + HiTZ/XCOPA-eu",
        "curation_process": "professional human translation + native re-annotation",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
    },
    "xnli": {
        "data_source": "MultiNLI (Williams 2018); XNLI + XNLIeu",
        "curation_process": "professional human translation (mt+post-edit for eu)",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
    },
    "xstorycloze": {
        "data_source": "Story Cloze Test (Mostafazadeh 2016), XStoryCloze",
        "curation_process": "professional human translation",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
    },
    "xwinograd": {
        "data_source": "aggregated native Winograd schemas",
        "curation_process": "originally-multilingual aggregation of native schemas",
        "curation_category": "originally_multilingual",
        "source_origin": "originally_multilingual",
    },
}

# Per-task overrides: (family, lang) → curation_category. Used when an `_eu`
# subset comes from a different paper with a different curation method than
# the rest of the family.
PER_TASK_OVERRIDES: dict[tuple[str, str], str] = {
    ("xnli", "eu"): "mt_post_edited",   # XNLIeu (Heredia et al. 2024)
    # paws_eu and xcopa_eu also come from separate papers, but their curation
    # method (professional human translation) matches the family default.
}

CATEGORY_ORDER = [
    "originally_multilingual",
    "human_translation",
    "template_generated",
    "mt_post_edited",
    "machine_translation",
]
ORIGIN_ORDER = ["originally_multilingual", "english_translated"]


def load_per_task_snr() -> pd.DataFrame:
    df = pd.read_csv(SNR_CSV, usecols=["task", SNR_COL])
    df["family"] = df["task"].map(benchmark_family)
    df["language"] = df["task"].map(assign_language)
    keep = [
        _is_language_aggregate(t, f) and f in FAMILY_META
        for t, f in zip(df["task"], df["family"])
    ]
    df = df[keep].copy()
    df = df.dropna(subset=[SNR_COL])
    return df


def per_family_aggregate(per_task: pd.DataFrame) -> pd.DataFrame:
    g = per_task.groupby("family")[SNR_COL]
    out = pd.DataFrame({
        "n_tasks": g.size(),
        "snr_median": g.median(),
        "snr_mean": g.mean(),
        "snr_max": g.max(),
    }).reset_index()
    meta = pd.DataFrame.from_dict(FAMILY_META, orient="index").reset_index().rename(
        columns={"index": "family"}
    )
    return out.merge(meta, on="family", how="left").sort_values("snr_median", ascending=False)


def per_task_with_overrides(per_task: pd.DataFrame) -> pd.DataFrame:
    out = per_task.copy()
    out["curation_category"] = [
        FAMILY_META[f]["curation_category"] for f in out["family"]
    ]
    out["source_origin"] = [
        FAMILY_META[f]["source_origin"] for f in out["family"]
    ]
    for (fam, lang), cat in PER_TASK_OVERRIDES.items():
        mask = (out["family"] == fam) & (out["language"] == lang)
        out.loc[mask, "curation_category"] = cat
    return out.sort_values(["family", "language"]).reset_index(drop=True)


def kruskal_wallis(values_by_group: dict[str, np.ndarray]) -> tuple[float, float, int]:
    """Return (H, p, n_groups). Skip groups with n<2."""
    groups = [v for v in values_by_group.values() if len(v) >= 2]
    if len(groups) < 2:
        return (float("nan"), float("nan"), len(groups))
    H, p = stats.kruskal(*groups)
    return (float(H), float(p), len(groups))


def _strip_plot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    order: list[str],
    label_col: str | None,
    title: str,
    out_path: Path,
) -> tuple[float, float, int]:
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 1.5))
    rng = np.random.default_rng(0)
    values_by_group: dict[str, np.ndarray] = {}
    for i, cat in enumerate(order):
        sub = df[df[group_col] == cat]
        vals = sub[value_col].to_numpy()
        values_by_group[cat] = vals
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(vals, np.full_like(vals, i, dtype=float) + jitter,
                   s=70, alpha=0.85, edgecolor="black", linewidth=0.6)
        if label_col is not None:
            for v, j, lbl in zip(vals, jitter, sub[label_col]):
                ax.annotate(lbl, (v, i + j), fontsize=7,
                            xytext=(4, 0), textcoords="offset points",
                            va="center")
        med = float(np.median(vals))
        ax.plot([med, med], [i - 0.32, i + 0.32], color="red", lw=1.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}\n(n={len(values_by_group[c])})" for c in order])
    ax.set_xscale("log")
    ax.set_xlabel(f"{value_col} (log scale)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    H, p, n_groups = kruskal_wallis(values_by_group)
    print(f"  Kruskal-Wallis: H = {H:.3f}, p = {p:.4f} ({n_groups} groups with n≥2)")
    return H, p, n_groups


_CATEGORY_COLORS = {
    "originally_multilingual": "#1f77b4",
    "human_translation": "#ff7f0e",
    "template_generated": "#2ca02c",
    "mt_post_edited": "#d62728",
    "machine_translation": "#9467bd",
}


def _ranked_bar(per_family: pd.DataFrame, out_path: Path) -> None:
    df = per_family.sort_values("snr_median", ascending=True)
    colors = [_CATEGORY_COLORS[c] for c in df["curation_category"]]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df) + 1.5))
    y = np.arange(len(df))
    ax.barh(y, df["snr_median"], color=colors, edgecolor="black", linewidth=0.6)
    for yi, (med, n) in enumerate(zip(df["snr_median"], df["n_tasks"])):
        ax.text(med, yi, f"  median={med:.2f}  (n={n})",
                va="center", ha="left", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["family"])
    ax.set_xscale("log")
    ax.set_xlabel("median snr_mpd_1B across the family's per-language tasks (log scale)")
    ax.set_title("Per-family SNR ranking (color = curation_category)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=col, ec="black", lw=0.6, label=cat)
        for cat, col in _CATEGORY_COLORS.items()
        if cat in set(df["curation_category"])
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Loading {SNR_CSV.relative_to(ROOT)}")
    per_task = load_per_task_snr()
    print(f"  → {len(per_task)} per-language aggregate tasks across "
          f"{per_task['family'].nunique()} families")

    per_family = per_family_aggregate(per_task)
    out_csv = HERE / "per_family_snr.csv"
    per_family.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv.name}")

    per_task_overrides = per_task_with_overrides(per_task)
    per_task_csv = HERE / "per_task_snr.csv"
    per_task_overrides.to_csv(per_task_csv, index=False)
    print(f"Wrote {per_task_csv.name}")

    group_rows: list[dict] = []

    # 1) Family-level strip plot by curation_category
    print("\nFamily-level Kruskal-Wallis by curation_category:")
    H, p, ng = _strip_plot(
        per_family,
        group_col="curation_category",
        value_col="snr_median",
        order=CATEGORY_ORDER,
        label_col="family",
        title=f"Per-family median {SNR_COL} by curation process",
        out_path=HERE / "snr_by_curation_process.png",
    )
    group_rows.append({"view": "family/curation", "H": H, "p": p, "n_groups": ng})

    # 2) Family-level strip plot by source_origin
    print("\nFamily-level Kruskal-Wallis by source_origin:")
    H, p, ng = _strip_plot(
        per_family,
        group_col="source_origin",
        value_col="snr_median",
        order=ORIGIN_ORDER,
        label_col="family",
        title=f"Per-family median {SNR_COL} by source origin",
        out_path=HERE / "snr_by_data_source.png",
    )
    group_rows.append({"view": "family/source", "H": H, "p": p, "n_groups": ng})

    # 3) Per-task strip plot by curation_category — exposes within-family
    #    heterogeneity the family-level view smears. Drop labels: too many
    #    points to annotate readably.
    print("\nPer-task Kruskal-Wallis by curation_category (with overrides):")
    H, p, ng = _strip_plot(
        per_task_overrides,
        group_col="curation_category",
        value_col=SNR_COL,
        order=CATEGORY_ORDER,
        label_col=None,
        title=f"Per-task {SNR_COL} by curation process (xnli_eu re-tagged as MT+post-edit)",
        out_path=HERE / "snr_by_curation_per_task.png",
    )
    group_rows.append({"view": "task/curation", "H": H, "p": p, "n_groups": ng})

    # 4) Headline: ranked bar chart of family medians, colored by curation.
    _ranked_bar(per_family, HERE / "snr_per_family_ranked.png")
    print("Wrote snr_per_family_ranked.png")

    pd.DataFrame(group_rows).to_csv(HERE / "group_stats.csv", index=False)
    print(f"\nWrote group_stats.csv")

    print("\nPer-family table (sorted by snr_median desc):")
    cols = ["family", "n_tasks", "snr_median", "snr_mean", "snr_max",
            "curation_category", "source_origin"]
    with pd.option_context("display.max_colwidth", 36, "display.width", 140):
        print(per_family[cols].to_string(index=False))


if __name__ == "__main__":
    main()
