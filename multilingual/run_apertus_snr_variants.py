"""Generate a per-benchmark CSV with decision accuracy per size and SNR per
variant per size, using every aggregator in snr.snr_variants.AGGREGATION_FUNCTIONS
(i.e. every SNR definition in snr_variants, excluding the simple
signal_to_noise_ratio used by snr_simple).

Reuses the Apertus loader and per-mix score arrays from snr_simple.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from tqdm import tqdm

from snr.constants import PLOT_DIR
from snr.dataloader import get_slice
from snr.download.apertus import load_apertus_eval_results
from snr.metrics import decision_acc_fast
from snr.snr_variants import AGGREGATION_FUNCTIONS

SMALL_SIZES = ["175M", "350M", "600M"]
TARGET_SIZE = "1B"
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]
LAST_N = 5
OUT_DIR = PLOT_DIR / "snr_definition"


def _safe(fn, *args, **kwargs):
    try:
        v = fn(*args, **kwargs)
        return float(v) if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def compute_decision_accuracy(df, task, small_size, target_size=TARGET_SIZE):
    scores_small = get_slice(df, size=small_size, task=task)
    scores_target = get_slice(df, size=target_size, task=task)
    if scores_small.empty or scores_target.empty:
        return float("nan")
    scores_small = scores_small.loc[scores_small.groupby("mix")["step"].idxmax()]
    scores_target = scores_target.loc[scores_target.groupby("mix")["step"].idxmax()]
    return decision_acc_fast(
        scores_small=scores_small.sort_values("model")["primary_score"],
        scores_target=scores_target.sort_values("model")["primary_score"],
    )


def per_mix_inputs(df, task, size, last_n=LAST_N):
    """Build the four per-mix arrays expected by snr_variants aggregators.

    Mirrors the conventions used in analysis/snr_variants.ipynb:
      step_noise          = per-mix std of the last `last_n` ckpts
      data_scores         = per-mix final-ckpt score
      data_noise          = per-mix std (here, std of the last `last_n` ckpts;
                            apertus has only one seed so cross-seed std isn't defined)
      data_scores_last_n  = per-mix mean of the last `last_n` ckpts
    """
    scores_df = get_slice(df, size=size, task=task).sort_values("step")
    if scores_df.empty:
        return None
    grouped = scores_df.groupby("mix")["primary_score"].apply(list)
    last_arrays = [np.asarray(s[-last_n:], dtype=float) for s in grouped]
    if any(len(a) == 0 for a in last_arrays) or len(last_arrays) < 2:
        return None
    step_noise = np.array([np.std(a) for a in last_arrays])
    data_scores = np.array([a[-1] for a in last_arrays])
    data_scores_last_n = np.array([a.mean() for a in last_arrays])
    # Cross-mix std of per-mix final scores. Only `rel_std_snr` reads this
    # (np.mean of an array of identicals = the scalar itself), but it's the
    # canonical SNR variant — passing per-mix step-std here collapses signal
    # to noise and yields ~1.0 for every cell.
    data_noise = np.full_like(data_scores, np.std(data_scores))
    return step_noise, data_scores, data_noise, data_scores_last_n


def variant_snr(inputs, agg_func):
    if inputs is None:
        return float("nan")
    _, _, snr = agg_func(*inputs)
    return snr


def variant_key(func_dict):
    """Stable column-name token derived from the function name, e.g. 'rel_std'."""
    name = func_dict["func"].__name__
    return name[:-4] if name.endswith("_snr") else name


def run():
    df = load_apertus_eval_results()
    tasks = sorted(df["task"].unique())
    print(f"Loaded {len(df):,} rows | {df['model'].nunique()} models | {len(tasks)} tasks")

    rows = []
    for task in tqdm(tasks, desc="Tasks"):
        row = {"task": task}
        for size in SMALL_SIZES:
            row[f"decision_acc_{size}"] = _safe(compute_decision_accuracy, df, task, size)

        size_inputs = {size: per_mix_inputs(df, task, size) for size in ALL_SIZES}
        for fd in AGGREGATION_FUNCTIONS:
            key = variant_key(fd)
            for size in ALL_SIZES:
                row[f"snr_{key}_{size}"] = _safe(variant_snr, size_inputs[size], fd["func"])
        rows.append(row)

    out = pd.DataFrame(rows).set_index("task").sort_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "snr_variants_per_task.csv"
    out.to_csv(csv_path)
    print(f"\nWrote CSV → {csv_path}")
    print(f"  {len(out)} tasks × {len(out.columns)} columns "
          f"({len(AGGREGATION_FUNCTIONS)} variants × {len(ALL_SIZES)} sizes "
          f"+ {len(SMALL_SIZES)} decision-accuracy columns)")


if __name__ == "__main__":
    run()
