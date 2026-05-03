"""Synthetic-data validation for the per-sample subset-search strategies.

The cluster runner (``run_per_sample_variants.py``) needs real eval_logs
under ``/iopsstor/...`` and can't be exercised off-cluster. This script
fabricates an (n_ckpts × n_samples) acc matrix with known structure and
runs every strategy against it, mirroring the on-cluster pipeline
end-to-end (matrix → strategy → cumulative sweep → summary).

Synthetic structure mimics what the real Apertus eval logs look like:

* 3 mixes × 5 last-N ckpts = 15 rows.
* A handful of "informative" samples whose mean acc differs by mix
  (these should bubble to the top of every strategy's ranking).
* A larger pool of "noisy" samples (high within-mix variance, no
  across-mix mean shift) — these should hurt SNR if included.
* A pile of "dead" samples (constant acc across all mixes) — Option D
  should drop them, A should sort them last, C may keep them if their
  total-score correlation is non-zero.

Outputs land under ``results/smooth_subtasks/per_sample_variants_synthetic/``
in the same layout as the real runner so ``analyze_per_sample_variants``
can be pointed at it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from multilingual.per_sample_strategies import (
    STRATEGIES,
    cumulative_subset_snrs,
    random_order_curve,
    run_strategy,
)

LANGS = ["de", "es", "fr", "zh", "ja", "ar", "ru", "vi", "th", "hi"]
SIZES = ["175M", "350M", "600M", "1B"]
N_CKPTS_PER_MIX = 5
MIXES = ["fwEdu30", "fwEdu60", "fwEdu90"]
OUT_ROOT = _REPO / "results" / "smooth_subtasks" / "per_sample_variants_synthetic"
# Cap Option B's budget tighter than the production default so the
# synthetic sweep finishes in seconds. The vectorised cumulative-SNR
# step keeps even the unconstrained sweep cheap (E with 32 perms over
# 5k samples is ~0.2s); B's quadratic-ish inner loop is the only one
# we cap. The cluster runner uses pool_cap=1000, budget=pool_cap.
B_OPTS = {"pool_cap": 800, "budget": 400}


def _make_matrix(rng: np.random.Generator, n_total: int, profile: str
                 ) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Synthetic acc matrix with realistic structure.

    ``profile`` controls the mix of sample types:

    * "easy"  — many informative samples, few dead, low noise
    * "hard"  — few informative, many dead, high noise
    * "mixed" — even split

    Returns (A, mix_rows). A is binary float32 (matches the real
    `samples_*.jsonl` ``acc`` field).
    """
    n_ckpts = len(MIXES) * N_CKPTS_PER_MIX
    A = np.zeros((n_ckpts, n_total), dtype=np.float32)

    if profile == "easy":
        n_inform = max(int(0.45 * n_total), 1)
        n_noisy = int(0.20 * n_total)
        noise_p = 0.08
    elif profile == "hard":
        n_inform = max(int(0.10 * n_total), 1)
        n_noisy = int(0.40 * n_total)
        noise_p = 0.18
    else:
        n_inform = max(int(0.25 * n_total), 1)
        n_noisy = int(0.30 * n_total)
        noise_p = 0.12
    n_dead = max(n_total - n_inform - n_noisy, 0)

    mix_rows: dict[str, list[int]] = {}
    for mi, mix in enumerate(MIXES):
        rows = list(range(mi * N_CKPTS_PER_MIX, (mi + 1) * N_CKPTS_PER_MIX))
        mix_rows[mix] = rows

    # Informative samples: each gets a per-mix base prob ascending with
    # the mix index (fwEdu30 < fwEdu60 < fwEdu90), with sample-level
    # offset so absolute level varies.
    for j in range(n_inform):
        base = rng.uniform(0.25, 0.65)
        spread = rng.uniform(0.05, 0.25)
        for mi, mix in enumerate(MIXES):
            p = float(np.clip(base + (mi - 1) * spread, 0.02, 0.98))
            for r in mix_rows[mix]:
                # Within-mix variation is small (last-N-ckpt stability).
                pj = float(np.clip(p + rng.normal(0, 0.03), 0.02, 0.98))
                A[r, j] = float(rng.random() < pj)

    # Noisy: roughly equal mean across mixes, high within-ckpt variance.
    for off, j in enumerate(range(n_inform, n_inform + n_noisy)):
        base = rng.uniform(0.20, 0.80)
        for r in range(n_ckpts):
            pj = float(np.clip(base + rng.normal(0, noise_p), 0.02, 0.98))
            A[r, j] = float(rng.random() < pj)

    # Dead: constant value (0 or 1) across every ckpt.
    for j in range(n_inform + n_noisy, n_total):
        v = float(rng.choice([0.0, 1.0], p=[0.55, 0.45]))
        A[:, j] = v

    return A, mix_rows


def _ckpts_from_mix_rows(mix_rows: dict[str, list[int]]
                         ) -> list[tuple[str, str, int]]:
    out = []
    for mix, rows in mix_rows.items():
        for s, r in enumerate(sorted(rows)):
            out.append(("dummy", mix, s))
    return out


def run_one(task: str, lang: str, size: str, n_total: int, profile: str,
            rng: np.random.Generator) -> list[dict]:
    A, mix_rows = _make_matrix(rng, n_total, profile)
    doc_ids = list(range(n_total))

    rows = []
    full_idx = np.arange(n_total)
    from multilingual.per_sample_strategies import signal_noise_1d
    _, _, full_set_snr = signal_noise_1d(A[:, full_idx].mean(axis=1), mix_rows)

    for name in ["A", "B", "C", "D", "E"]:
        opts = B_OPTS if name == "B" else {}
        try:
            r = run_strategy(name, A, mix_rows, doc_ids, rng=rng, **opts)
        except Exception as e:
            rows.append({"language": lang, "task": task, "size": size,
                         "strategy": name, "status": "error", "error": repr(e)})
            continue
        rows.append({
            "language": lang, "task": task, "size": size,
            "strategy": name, "status": "ok",
            "n_total": r["n_total"],
            "n_candidates": r["n_candidates"],
            "best_n": r["best_n"],
            "best_snr": r["best_snr"],
            "full_set_snr": full_set_snr,
            "snr_gain": (r["best_snr"] - full_set_snr)
                if np.isfinite(r["best_snr"]) and np.isfinite(full_set_snr)
                else float("nan"),
            "best_frac": (r["best_n"] / r["n_total"]) if r["n_total"] else float("nan"),
            "profile": profile,
        })
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Realistic-ish task footprint. A few benchmark families × per-language
    # tasks × ckpt sizes, with sample counts matching the real ones
    # (xnli ≈ 5k, arc ≈ 1k, belebele ≈ 900, mmlu ≈ 14k → cap at 1500 to
    # keep the synthetic experiment fast).
    benchmarks = [
        ("arc", 1100, "easy"),
        ("xnli", 5000, "hard"),
        ("belebele", 900, "mixed"),
        ("mmlu", 1500, "hard"),
    ]

    all_rows = []
    for lang in LANGS:
        for bm, n_total, profile in benchmarks:
            task = f"{bm}_{lang}"
            for size in SIZES:
                all_rows.extend(
                    run_one(task, lang, size, n_total, profile, rng)
                )
    df = pd.DataFrame(all_rows)
    out_csv = OUT_ROOT / "summary_all.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows → {out_csv}")

    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        agg = ok.groupby("strategy").agg(
            mean_best_snr=("best_snr", "mean"),
            median_best_snr=("best_snr", "median"),
            mean_full_set_snr=("full_set_snr", "mean"),
            mean_snr_gain=("snr_gain", "mean"),
            mean_best_frac=("best_frac", "mean"),
            n_runs=("best_snr", "size"),
        ).reset_index()
        agg.to_csv(OUT_ROOT / "by_strategy_means.csv", index=False)
        print("\n=== Synthetic per-strategy aggregates ===")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
