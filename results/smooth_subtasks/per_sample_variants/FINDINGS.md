# Per-sample subset-search strategies — implementation + synthetic validation

This branch (`claude/implement-benchmark-proposals-UkhdX`) implements
Options **A, B, C, E** from
[`results/smooth_subtasks/per_sample/PROPOSALS.md`](../per_sample/PROPOSALS.md)
alongside the previously-shipped **Option D** so all five can be
compared head-to-head on the same (task, size) inputs.

## What's new

| File | Role |
| --- | --- |
| [`multilingual/per_sample_strategies.py`](../../../multilingual/per_sample_strategies.py) | Strategy implementations + shared primitives. Each `select_X` takes the (n_ckpts × n_samples) acc matrix and returns the cumulative-SNR sweep + best subset. |
| [`multilingual/run_per_sample_variants.py`](../../../multilingual/run_per_sample_variants.py) | Cluster runner. Walks Apertus eval_logs, builds matrices, runs all 5 strategies side-by-side, writes per-task and roll-up CSVs + per-size overlay plots. |
| [`multilingual/analyze_per_sample_variants.py`](../../../multilingual/analyze_per_sample_variants.py) | Post-hoc analysis. Reads `summary_all.csv`, emits per-strategy aggregates, win-rate vs Option D, per-language winners, and three comparison plots. |
| [`multilingual/validate_per_sample_strategies.py`](../../../multilingual/validate_per_sample_strategies.py) | Off-cluster smoke-test on synthetic data with realistic structure. Confirms the implementation runs end-to-end without needing `/iopsstor/...`. |

The original Option-D pipeline
(`multilingual/smooth_subtasks_per_sample.py`) is **untouched** — its
primitives are reused, not replaced.

### Strategy summary

* **A** — sort all samples by per-sample SNR, sweep cumulative subset.
  Dead samples (signal=0) come out NaN and sort to the back. Cheapest
  variant; matches the upstream AllenAI semantics.
* **B** — forward greedy. Seed with the highest-SNR sample, then at
  each step add the candidate that maximally raises *combined* SNR.
  Vectorised via `signal_noise_batch` (one numpy call evaluates every
  candidate's contribution per step). Pool capped by Option-A SNR
  (`pool_cap`, default 1000) to keep mmlu_* tractable.
* **C** — classical-test-theory discrimination filter + Option A. Items
  ranked by point-biserial correlation between per-sample acc and the
  ckpt total score; top `keep_frac` (default 0.5) survive, then sorted
  by per-sample SNR exactly as A does. The discrimination index is the
  rank-equivalent limit of the 2PL `a_i` parameter, so we sidestep
  fitting an IRT model while keeping its information-theoretic story.
* **D** — variance prefilter + Option A. Already shipped; kept in the
  strategy registry so the side-by-side runner is self-contained.
* **E** — random-search baseline. Tries `n_random_orders` (default 32)
  permutations, returns the one whose cumulative sweep peaks highest.
  This is the proper version of the "random order" dashed line that
  was already drawn next to Option D's curve.

Shared primitives (`signal_noise_1d`, `signal_noise_batch`,
`per_sample_snr`, `cumulative_subset_snrs`, …) live in
`per_sample_strategies` and are vectorised end-to-end. The cumulative
sweep (used by A/B/C/D for the per-N curve and by E for random
baselines) was the slow path — replacing the per-N Python loop with
one batched `signal_noise_batch` call took E from **16s** to **0.22s**
on a 5k-sample matrix.

## Synthetic validation

Cluster-only data lives under `/iopsstor/...`, which is not available
where this branch was developed, so the implementation was validated
on a synthetic acc matrix with the same structure as a real Apertus
eval (3 mixes × 5 last-N ckpts = 15 rows; binary 0/1 acc; deliberate
mix of "informative" / "noisy" / "dead" samples). The fabrication
recipe is in `validate_per_sample_strategies.py`.

Run footprint: 10 langs × 4 benchmark families (arc, xnli, belebele,
mmlu) × 4 sizes × 5 strategies = **800 strategy invocations**, total
wall time ≈ 30s. Outputs at
[`per_sample_variants_synthetic/`](../per_sample_variants_synthetic/).

### Synthetic-run aggregates

| Strategy | Mean best SNR | Median best SNR | Mean SNR gain over full set | Mean best_n / n_total |
| --- | --- | --- | --- | --- |
| **A** (sort)                | 2.404 | 2.416 | +0.140 | 10.3 % |
| **B** (greedy)              | **2.449** | **2.449** | **+0.186** | **2.6 %** |
| **C** (discrim + sort)      | 2.424 | 2.430 | +0.160 | 7.0 % |
| **D** (var-filter + sort)   | 2.404 | 2.416 | +0.140 | 11.6 % |
| **E** (random)              | 2.365 | 2.388 | +0.102 | 56.0 % |

### Win rate vs Option D (per ckpt size)

|        size | A wins | B wins | C wins | E wins |
| ---:        | ---:   | ---:   | ---:   | ---:   |
| 175M  | 35 % | **100 %** | 75 % | 13 % |
| 350M  | 28 % |  95 %     | 73 % | 20 % |
| 600M  | 45 % | **100 %** | 75 % | 20 % |
| 1B    | 48 % | **100 %** | 78 % | 10 % |

(40 (lang, task) pairs per row; "win" = strict > on `best_snr`.)

### Key findings (on synthetic data — must be re-verified on cluster)

1. **Option B (forward greedy) dominates on synthetic data.** It
   strictly beats Option D in **95–100 %** of (lang, task, size)
   triples, with a median uplift of **+0.033 SNR** and a mean uplift
   of **+0.045 SNR**. The win is universal across languages — every
   one of the 10 synthetic languages picks B as its winning strategy.

2. **B picks much smaller subsets.** Mean `best_n / n_total` is
   **2.6 %** for B vs 11.6 % for D and 10.3 % for A. On synthetic
   xnli (n=5000), B's best subset is typically ~50 samples; D's is
   typically ~600. This is the "interactions A misses" effect that
   `PROPOSALS.md` predicted: B finds tiny subsets where individually-
   mediocre samples reinforce the cross-mix mean separation.

3. **Option C (discrimination-filter + sort) is a cheap win.** It
   beats D in **73–78 %** of cases with a median uplift of **+0.010
   SNR**. The point-biserial discrimination index is essentially free
   to compute (one matrix dot-product) and gives a tighter
   informative-pool than the variance prefilter, which only excludes
   *strictly-constant* items. C peaks at a 7 % subset vs D's 12 %.

4. **A and D are statistically indistinguishable on this data.**
   Mean and median best SNR are identical to three decimals
   (A: 2.404 / 2.416; D: 2.404 / 2.416). Their delta-vs-D histogram
   is a spike at 0. The variance prefilter is conservative enough
   (drops only `signal=0` samples) that on an Option-A sort it just
   sends NaN-SNR survivors to the back of the queue — the same place
   the variance filter dropped them. **Option D's prefilter is
   redundant when the downstream step is Option A's sort.** The
   prefilter still earns its keep on the *runtime* axis (smaller
   sweep, smaller cumsum), but not on the *quality* axis.

5. **Random search (E) is a meaningful negative baseline.** E loses
   to D in 80–90 % of runs (median Δ −0.02 SNR) and peaks at >50 %
   of the sample pool — i.e., random orderings find their best by
   essentially using the whole task. This validates that the
   per-sample SNR ranking primitive is doing useful work and isn't
   getting beaten by chance.

6. **B's cap matters more than its budget.** With `pool_cap=800,
   budget=400`, B's best subset is typically reached well before the
   400-sample budget ends (`best_n` ≈ 50–200). Lowering `pool_cap`
   below the typical informative-sample count would hurt B; lowering
   `budget` below ~200 would not. For the production cluster run the
   default `pool_cap=1000, budget=pool_cap` is a safe ceiling.

> ⚠️ These findings are **synthetic-data findings**: the matrix
> generator in `validate_per_sample_strategies.py` is hand-written to
> have a clear informative/noisy/dead split, so the *ordering*
> A < D ≈ A < C < B is a property of that structure rather than of
> the Apertus eval logs. Run
> `python multilingual/run_per_sample_variants.py` on the cluster
> followed by
> `python multilingual/analyze_per_sample_variants.py` to refresh
> these tables against real eval data.

## How to reproduce

```bash
# On the cluster (real eval logs):
cd /iopsstor/scratch/cscs/mariagrandury/signal-and-noise
python multilingual/run_per_sample_variants.py
python multilingual/analyze_per_sample_variants.py

# Off-cluster smoke test (synthetic data):
python multilingual/validate_per_sample_strategies.py
python multilingual/analyze_per_sample_variants.py \
    --root results/smooth_subtasks/per_sample_variants_synthetic
```

## Output layout (real run)

```
results/smooth_subtasks/per_sample_variants/
    summary_all.csv             one row per (lang, task, size, strategy)
    by_strategy_means.csv       overall aggregates (written by runner)
    delta_vs_d.csv              per-row delta against D
    by_strategy_summary.csv     written by analyze script
    by_strategy_per_size.csv    same broken out by ckpt size
    win_rates.csv               per (size, strategy) win rate vs D
    per_language_winners.csv    winning strategy per language
    best_snr_box.png            distribution of best SNR per strategy
    delta_vs_d_hist.png         histogram of (best_snr - best_snr_D)
    best_frac_violin.png        distribution of best_n / n_total
    <lang>/<task>/
        summary.csv             one row per (size, strategy)
        ranked_<strat>.csv      doc_id × per-size SNR for each strategy
        best_subset_<size>_<strat>.txt   doc_ids of the best subset
        cumulative_snr_<size>.png        all 5 strategies overlaid
```
