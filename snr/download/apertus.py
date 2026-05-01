"""Load eval results for the 12 custom Apertus pretraining models from disk.

Models: apertus-{175M,350M,600M,1B}-fwEdu{30,60,90}-fw{270,240,210}-seed1904
Layout: <EVAL_ROOT>/<model>-iter<N>/harness/eval_*/results_*.json
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# Reuse the on-disk parser already maintained for the W&B push pipeline
# (swissai-evals-post-train) rather than reimplementing it here.
_SWISSAI = "/iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train"
if _SWISSAI not in sys.path:
    sys.path.insert(0, _SWISSAI)
from scripts.push_all_results import collect, aggregate_parents  # noqa: E402

DEFAULT_EVAL_ROOT = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/"
    "mariagrandury-epflnlp/snr-experiments"
)

_MODEL_RE = re.compile(
    r"^apertus-(?P<size>175M|350M|600M|1B)-fwEdu(?P<edu>30|60|90)-fw(?P<fw>270|240|210)-seed(?P<seed>\d+)-iter(?P<iter>\d+)$"
)

# Approximate non-embedding parameter counts (used to compute FLOPs ≈ 6·params·tokens).
_PARAMS = {"175M": 175e6, "350M": 350e6, "600M": 600e6, "1B": 1.0e9}
# Megatron training config: tokens per iter = micro_batch_size * seq_len = 504 * 4096
_TOKENS_PER_ITER = 504 * 4096


def load_apertus_eval_results(eval_root: str | Path = DEFAULT_EVAL_ROOT) -> pd.DataFrame:
    """Walk eval_root and return one row per (model, ckpt, task) with primary_score.

    Columns match the schema expected by snr.dataloader.get_slice:
    model, mix, size, step, task, primary_score, seed, plus tokens/compute.
    """
    eval_root = Path(eval_root)
    rows = []
    for ckpt_dir in eval_root.iterdir():
        m = _MODEL_RE.match(ckpt_dir.name)
        if not m:
            continue
        size = m["size"]
        mix = f"fwEdu{m['edu']}"
        seed = int(m["seed"])
        step = int(m["iter"])
        ckpt_scores = aggregate_parents(collect(ckpt_dir))
        if not ckpt_scores:
            continue
        tokens = step * _TOKENS_PER_ITER
        compute = 6 * _PARAMS[size] * tokens
        for task, scores in ckpt_scores.items():
            score = scores.get("acc,none", scores.get("exact_match,none"))
            if score is None:
                continue
            rows.append(
                dict(
                    model=f"apertus-{size}-{mix}",
                    mix=mix,
                    size=size,
                    step=step,
                    task=task,
                    primary_score=float(score),
                    seed=seed,
                    tokens=tokens,
                    compute=compute,
                )
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["size", "mix", "step", "task"]).reset_index(drop=True)
