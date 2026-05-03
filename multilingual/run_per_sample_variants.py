"""Run every per-sample subset-search strategy (A/B/C/D/E) side by side.

Companion to ``smooth_subtasks_per_sample.py`` which only runs Option D
(see ``results/smooth_subtasks/per_sample/PROPOSALS.md``). This script
shares the same eval-log walker and (n_ckpts × n_samples) matrix
builder, but invokes every strategy from ``per_sample_strategies``
on each (task, size) and writes a side-by-side comparison.

Outputs are written under ``results/smooth_subtasks/per_sample_variants/``:

  <lang>/<task>/
      summary.csv               one row per (size, strategy)
      ranked_<strategy>.csv     order + per-sample SNR for each strategy
      best_subset_<size>_<strat>.txt   doc_ids of the best subset
      cumulative_snr_<size>.png        per-size, all strategies overlaid

  summary_all.csv               one row per (lang, task, size, strategy)
  by_strategy_means.csv         mean best/full SNR aggregates
  delta_vs_d.csv                per-strategy uplift over Option D (the
                                currently-shipped baseline)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from multilingual.analyze_snr_variants import assign_language
from multilingual.per_sample_strategies import (
    STRATEGIES,
    cumulative_subset_snrs,
    random_order_curve,
    run_strategy,
)
from multilingual.smooth_subtasks import collect_multilingual_families
from multilingual.smooth_subtasks_per_sample import (
    ALL_SIZES,
    _build_matrix,
    _last_n_rows_per_mix,
    load_samples,
)
from snr.constants import PLOT_DIR
from snr.download.apertus import load_apertus_eval_results

OUT_ROOT = PLOT_DIR / "smooth_subtasks" / "per_sample_variants"
STRATEGY_NAMES = ["A", "B", "C", "D", "E"]
STRATEGY_COLOURS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green",
                    "D": "tab:red", "E": "tab:purple", "random": "grey"}


def _run_one_size(A, ckpts, doc_ids, rng) -> dict[str, dict]:
    """Returns {strategy_name: result_dict | None} for one (task, size)."""
    mix_rows = _last_n_rows_per_mix(ckpts)
    if len(mix_rows) < 2:
        return {}

    out = {}
    for name in STRATEGY_NAMES:
        try:
            res = run_strategy(name, A, mix_rows, doc_ids, rng=rng)
        except Exception as exc:  # pragma: no cover — defensive
            res = {"strategy": name, "error": repr(exc),
                   "ordered": np.empty(0, dtype=np.int64),
                   "cumulative_snrs": [], "best_n": 0,
                   "best_snr": float("nan"),
                   "full_set_snr": float("nan"),
                   "per_sample_snrs": np.empty(0),
                   "n_total": A.shape[1], "n_candidates": 0}
        out[name] = res

    # Random-order baseline using A's ordering (matches Option D plot semantics).
    rand_curve = []
    if "A" in out and out["A"]["ordered"].size > 0:
        rand_curve = random_order_curve(A, out["A"]["ordered"], mix_rows, rng)
    out["_random_baseline_curve"] = rand_curve
    return out


def _plot_overlay(task: str, per_size: dict[str, dict[str, dict]],
                  save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    for size, by_strat in per_size.items():
        if not by_strat:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        any_drawn = False
        for name in STRATEGY_NAMES:
            r = by_strat.get(name)
            if not r or not r.get("cumulative_snrs"):
                continue
            curve = r["cumulative_snrs"]
            x = np.arange(1, len(curve) + 1)
            label = (f"{name} (n_cand={r.get('n_candidates', '?')}, "
                     f"best top-{r['best_n']}, snr {r['best_snr']:.3f})")
            ax.plot(x, curve, label=label,
                    color=STRATEGY_COLOURS[name], linewidth=1.0)
            if r["best_n"] > 0:
                ax.axvline(r["best_n"], color=STRATEGY_COLOURS[name],
                           linestyle=":", linewidth=0.6, alpha=0.5)
            any_drawn = True

        rand_curve = by_strat.get("_random_baseline_curve") or []
        if rand_curve:
            x = np.arange(1, len(rand_curve) + 1)
            ax.plot(x, rand_curve, label="random order",
                    color=STRATEGY_COLOURS["random"], linewidth=0.7,
                    alpha=0.7, linestyle="--")
            any_drawn = True

        full_snr = next((r.get("full_set_snr")
                         for r in by_strat.values()
                         if isinstance(r, dict) and r.get("full_set_snr") is not None),
                        None)
        if full_snr is not None and np.isfinite(full_snr):
            ax.axhline(full_snr, color="black", linewidth=0.6, linestyle="--",
                       alpha=0.4, label=f"full set snr {full_snr:.3f}")

        if not any_drawn:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        ax.set_title(f"{task} — {size}", fontsize=10)
        ax.set_xlabel("Subset size (samples in selection order, log scale)")
        ax.set_ylabel("Combined SNR")
        ax.grid(True, linestyle="-", alpha=0.2)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(save_dir / f"cumulative_snr_{size}.png", dpi=110)
        plt.close(fig)


def _write_per_task(task: str, per_size: dict[str, dict[str, dict]],
                    doc_ids_per_size: dict[str, list[int]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # summary.csv: (size, strategy)
    rows = []
    for size in ALL_SIZES:
        by_strat = per_size.get(size) or {}
        if not by_strat:
            for name in STRATEGY_NAMES:
                rows.append({"size": size, "strategy": name, "status": "no_data"})
            continue
        for name in STRATEGY_NAMES:
            r = by_strat.get(name)
            if not r:
                rows.append({"size": size, "strategy": name, "status": "no_data"})
                continue
            if r.get("error"):
                rows.append({"size": size, "strategy": name, "status": "error",
                             "error": r["error"]})
                continue
            rows.append({
                "size": size, "strategy": name, "status": "ok",
                "n_total": r["n_total"],
                "n_candidates": r["n_candidates"],
                "best_n": r["best_n"],
                "best_snr": r["best_snr"],
                "full_set_snr": r["full_set_snr"],
                "snr_gain": (r["best_snr"] - r["full_set_snr"])
                    if np.isfinite(r["best_snr"]) and np.isfinite(r["full_set_snr"])
                    else float("nan"),
                "best_frac": (r["best_n"] / r["n_total"]) if r["n_total"] else float("nan"),
            })
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    # ranked_<strat>.csv: doc_id, snr (the per-sample SNR), and per-size
    # in_best flag (one column per size that the strategy ran on).
    for name in STRATEGY_NAMES:
        all_doc_ids = sorted({d for ds in doc_ids_per_size.values() for d in ds})
        if not all_doc_ids:
            continue
        by_size_idx = {size: {d: i for i, d in enumerate(doc_ids_per_size.get(size, []))}
                       for size in ALL_SIZES}
        rows = []
        for d in all_doc_ids:
            row = {"doc_id": d}
            for size in ALL_SIZES:
                by_strat = per_size.get(size) or {}
                r = by_strat.get(name)
                if not r or "per_sample_snrs" not in r:
                    row[f"snr_{size}"] = float("nan")
                    row[f"in_best_{size}"] = False
                    continue
                idx = by_size_idx[size].get(d)
                if idx is None or idx >= r["per_sample_snrs"].shape[0]:
                    row[f"snr_{size}"] = float("nan")
                    row[f"in_best_{size}"] = False
                    continue
                row[f"snr_{size}"] = float(r["per_sample_snrs"][idx])
                best_set = (set(r["ordered"][: r["best_n"]].tolist())
                            if r["best_n"] > 0 else set())
                row[f"in_best_{size}"] = idx in best_set
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"ranked_{name}.csv", index=False)

    # best_subset_<size>_<strat>.txt
    for size in ALL_SIZES:
        by_strat = per_size.get(size) or {}
        for name in STRATEGY_NAMES:
            r = by_strat.get(name)
            if not r or r.get("best_n", 0) <= 0:
                continue
            doc_ids = doc_ids_per_size.get(size) or []
            if not doc_ids:
                continue
            best_cols = r["ordered"][: r["best_n"]].tolist()
            best_doc_ids = [doc_ids[c] for c in best_cols if c < len(doc_ids)]
            (out_dir / f"best_subset_{size}_{name}.txt").write_text(
                "\n".join(str(d) for d in best_doc_ids) + "\n"
            )

    _plot_overlay(task, per_size, out_dir)


def run_one_task(task: str, by_ckpt: dict, out_root: Path,
                 rng: np.random.Generator):
    lang = assign_language(task)
    if lang == "??":
        return None
    out_dir = out_root / lang / task

    per_size: dict[str, dict[str, dict]] = {}
    doc_ids_per_size: dict[str, list[int]] = {}
    for size in ALL_SIZES:
        built = _build_matrix(by_ckpt, size)
        if built is None:
            per_size[size] = {}
            continue
        A, ckpts, doc_ids = built
        doc_ids_per_size[size] = doc_ids
        per_size[size] = _run_one_size(A, ckpts, doc_ids, rng)

    _write_per_task(task, per_size, doc_ids_per_size, out_dir)
    return out_dir


def main():
    df_meta = load_apertus_eval_results()
    families = collect_multilingual_families(df_meta)
    tasks = sorted({t for ts in families.values() for t in ts})
    print(f"Targeting {len(tasks)} multilingual language-tasks "
          f"({len(families)} families).")

    samples = load_samples(set(tasks))
    print(f"Parsed samples for {len(samples)} tasks.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    written = 0
    for task in tqdm(tasks, desc="tasks"):
        if task not in samples:
            continue
        out = run_one_task(task, samples[task], OUT_ROOT, rng)
        if out is not None:
            written += 1

    # Roll-up CSVs.
    master_rows = []
    for csv in OUT_ROOT.rglob("summary.csv"):
        if csv.parent == OUT_ROOT:
            continue
        df = pd.read_csv(csv)
        df["task"] = csv.parent.name
        df["language"] = csv.parent.parent.name
        master_rows.append(df)
    if master_rows:
        master = pd.concat(master_rows, ignore_index=True)
        cols = ["language", "task"] + [c for c in master.columns
                                       if c not in ("language", "task")]
        master = master[cols].sort_values(["language", "task", "size", "strategy"])
        master_path = OUT_ROOT / "summary_all.csv"
        master.to_csv(master_path, index=False)
        print(f"Wrote roll-up → {master_path}")

        ok = master[master["status"] == "ok"].copy()
        if not ok.empty:
            agg = ok.groupby("strategy").agg(
                mean_best_snr=("best_snr", "mean"),
                median_best_snr=("best_snr", "median"),
                mean_full_set_snr=("full_set_snr", "mean"),
                mean_snr_gain=("snr_gain", "mean"),
                median_snr_gain=("snr_gain", "median"),
                mean_best_frac=("best_frac", "mean"),
                n_runs=("best_snr", "size"),
            ).reset_index()
            agg.to_csv(OUT_ROOT / "by_strategy_means.csv", index=False)
            print("Per-strategy aggregates:")
            print(agg.to_string(index=False))

            # Delta vs D.
            keys = ["language", "task", "size"]
            d_rows = ok[ok["strategy"] == "D"][keys + ["best_snr"]].rename(
                columns={"best_snr": "best_snr_D"})
            joined = ok.merge(d_rows, on=keys, how="left")
            joined["delta_vs_D"] = joined["best_snr"] - joined["best_snr_D"]
            joined[keys + ["strategy", "best_snr", "best_snr_D",
                          "delta_vs_D", "best_n", "n_candidates"]].to_csv(
                OUT_ROOT / "delta_vs_d.csv", index=False)

    print(f"Wrote {written} per-task output dirs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
