# Context for Claude — signal-and-noise (Apertus fork)

This is a local fork of [allenai/signal-and-noise](https://github.com/allenai/signal-and-noise),
augmented to run the SNR / decision-accuracy pipeline on the 12 **custom Apertus
pretraining checkpoints** evaluated in the sister repo
`swissai-evals-post-train`. The upstream README still applies for the AllenAI
DataDecide / OLMo path; this file documents the Apertus extension.

If you're picking this up cold, **read the upstream [README.md](README.md) first**
to understand what signal, noise, decision accuracy, and scaling-law error mean.
This file is the back-of-house Claude memo for the local additions.

---

## What's actually running

The Apertus pipeline reuses the upstream signal-and-noise compute + plotting
helpers, but loads scores from the cluster's eval_logs tree instead of the
HF parquet dataset.

```bash
cd /iopsstor/scratch/cscs/mariagrandury/signal-and-noise
python multilingual/run_apertus.py
```

That single entry point:
1. Walks `<EVAL_ROOT>/<model>-iter<N>/` and parses every `eval_*/results_*.json`
   into a long-form DataFrame (one row per `(model, ckpt, task)`).
2. Feeds it into `snr.snr_simple.main` with Apertus-specific sizes:
   `small_sizes=["175M", "350M", "600M"]`, `large_sizes_snr=["1B"]`,
   `target_size="1B"`, `target_step=None` (latest available step per mix).
3. Drops scaling-law error (`large_sizes_scaling=[]`) — we don't have a
   ladder of Apertus models, so prediction error is not computed.
4. Writes outputs under `img/apertus_multilingual/`:
   - `snr_per_task.csv` — one row per task with `decision_acc_<size>` /
     `snr_<size>` columns
   - `curves/<task>.png` — per-task training-curve plots, one panel per
     mix, generated via `analysis.plotting.datadecide.plot_task_curves`
   - `snr_vs_decision_accuracy.png` — multi-panel scatter, one panel per
     small size (175M/350M/600M → 1B), via `snr.plot.plot_snr_da_grid`

---

## Apertus models in scope (12 models)

```
apertus-{175M,350M,600M,1B}-fwEdu{30,60,90}-fw{270,240,210}-seed1904
```

- **mix** = `fwEdu{30,60,90}` (the parser strips the `fw270/240/210`
  complement, so a model's `mix` field carries only the FW-Edu ratio)
- **seed** = 1904 (constant; not the upstream DataDecide seed set)
- **size** = `175M`, `350M`, `600M`, `1B`

In `multilingual/run_apertus.py`:
- `SMALL_SIZES = ["175M", "350M", "600M"]`
- `TARGET_SIZE = "1B"`
- `PLOTTED_MIXES = ["fwEdu30", "fwEdu60", "fwEdu90"]`
- `SEED = 1904`

Half-trained models (600M-fwEdu90, 1B-fwEdu90, the three 175M-fwEdu*) may
have fewer than 5 ckpts on some mixes. `compute_snr_small_scale` in
`snr/snr_simple.py` already accommodates this — it keeps per-mix score
arrays as a jagged list rather than forcing a 2-D ndarray. Don't
"refactor" that back to a square array without re-handling the missing
ckpts, or SNR computation will crash on those mixes.

---

## Eval-results layout (read-only input)

`snr/download/apertus.py` reads from:

```
/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/
    mariagrandury-epflnlp/snr-experiments/
        <model>-iter<N>/
            harness/eval_*/results_*.json     (clean lm-eval output)
            harness/eval_*/per_task/<task>/   (partial, written per-task)
```

This tree is **populated by the `swissai-evals-post-train` repo**, not by
this one. Every Slurm eval job writes there. We just read it.

To avoid duplicating the parser, `snr/download/apertus.py` does:

```python
sys.path.insert(0, "/iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train")
from scripts.push_all_results import collect, aggregate_parents
```

If `swissai-evals-post-train/scripts/push_all_results.py` moves or its
`collect` / `aggregate_parents` API changes, this import breaks. Both
repos are under the same `/iopsstor/scratch/cscs/mariagrandury/` parent,
so they should usually be in sync — but the coupling is implicit, not
declared anywhere.

`collect` reads both `results` and `groups` from per-task fragments so
that aggregates like `mmlu` (which only live under `groups`) are
recovered when results are merged from sharded per-task runs.
`aggregate_parents` then folds e.g. `mmlu_anatomy`, `mmlu_humanities`, …
into `mmlu` when `mmlu` is present in the same ckpt's results.

The metric extracted per task is `acc,none` if present, else
`exact_match,none`. `acc_norm`, `acc_bytes`, `*_stderr`, `degeneration`
are intentionally dropped (matches the W&B push schema in the sister repo).

---

## Tokens / FLOPs (for the curves)

Computed inside `load_apertus_eval_results`:

- Tokens per iter: `_TOKENS_PER_ITER = 504 * 4096` (Megatron training config:
  `micro_batch_size * seq_len`).
- Tokens at iter N: `step * _TOKENS_PER_ITER`.
- Compute (FLOPs) ≈ `6 * params * tokens`, with `_PARAMS = {175M: 175e6,
  350M: 350e6, 600M: 600e6, 1B: 1.0e9}`.

These approximations are the same ones used by `swissai-evals-post-train`'s
W&B push (`MEG_TOKENS_PER_ITER = 504 * 4096`) so axes line up across the
two pipelines.

---

## Outputs

`img/apertus_multilingual/` is the destination. `PLOT_DIR` resolves to
`<repo>/img/` via `snr/constants/__init__.py`. Existing artifacts there
will be overwritten on each run.

- `snr_per_task.csv` is the table version of the upstream Rich-print.
- `curves/` contains one PNG per task that has enough `(size, mix)`
  coverage to render. Tasks with incomplete coverage are silently skipped
  (the loop in `_plot_curves` swallows exceptions and logs nothing — if
  you expect a task to be there and it isn't, that's why).
- `snr_vs_decision_accuracy.png` is a 1×3 grid (175M / 350M / 600M → 1B).

---

## Cluster gotchas

- **Login node:** `clariden-ln003`. System Python is 3.6 — use a recent
  env (`miniconda3/envs/snr` or similar) when running this; the upstream
  `pyproject.toml` requires Python ≥ 3.9.
- **No outbound internet from compute nodes.** If you need to call
  `pull_predictions_from_hf` (only used for the upstream DataDecide /
  OLMo path, not Apertus), do it from the login node.
- **`/iopsstor/scratch` is the work tree.** This repo and the eval_logs
  it reads both live there.

---

## Relationship to other local repos

| Repo | Path | Role |
|---|---|---|
| `signal-and-noise` (this) | `/iopsstor/scratch/cscs/mariagrandury/signal-and-noise` | SNR / decision-accuracy compute + plotting |
| `swissai-evals-post-train` | `/iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train` | Submits eval jobs, writes `eval_logs`, pushes to W&B |
| `data-mix-small` (Megatron-LM) | `/iopsstor/scratch/cscs/mariagrandury/data-mix-small` | Pretraining; checkpoints under `Megatron-LM/logs/Meg-Runs/...` |
| `pretrain` | `/iopsstor/scratch/cscs/mariagrandury/pretrain/megatron/data-mix-small` | Pretraining submitter (sbatch wrappers) |

Flow: pretrain → checkpoints → swissai-evals-post-train submits
`lm_eval` jobs → `eval_logs/.../snr-experiments/<model>-iter<N>/` →
`signal-and-noise/multilingual/run_apertus.py` reads those and produces
SNR tables + plots.

---

## When upstream changes

This repo tracks `allenai/signal-and-noise`. When pulling upstream, the
local additions to watch for are:

- `multilingual/run_apertus.py` (local-only entry point)
- `snr/download/apertus.py` (local-only loader)
- The lazy import of `run_ladder` inside `compute_scaling_law_error`
  (`snr/snr_simple.py`) — done so the Apertus path doesn't need
  `olmo-ladder` installed. If upstream re-imports it at module top, the
  Apertus run will fail with a missing-dependency error.
- The jagged-array tolerance in `compute_snr_small_scale`
  (`snr/snr_simple.py`) for half-trained mixes.

Re-apply those if a merge undoes them.
