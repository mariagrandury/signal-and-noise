# Apertus signal-and-noise — analysis summary & main-figure options

This document is the **analysis branch deliverable**. It (1) summarises every
result currently committed under `results/`, (2) distils the empirical
findings, and (3) proposes three concrete options for the *main figures* of
the paper (the 1–2 hero figures that anchor the story). Detailed numbers
quoted below come from [`snr_definition/README.md`](snr_definition/README.md);
the underlying CSVs are stored via git-lfs.

## 1. What's in `results/` (what was computed)

The repo runs three independent analysis pipelines on the 12 Apertus
checkpoints (`apertus-{175M,350M,600M,1B}-fwEdu{30,60,90}-…-seed1904`). Each
pipeline writes one tree under `results/`:

| Tree | Driver | What it produces |
|---|---|---|
| `acc_vs_flops/` | [`multilingual/run_apertus.py`](../multilingual/run_apertus.py) | 27 accuracy-vs-FLOPs grid PNGs: 15 per-benchmark-family (subplots = languages) + 12 per-language (subplots = benchmark families). Pure descriptive scaling view. |
| `snr_definition/` | [`run_apertus_snr_variants.py`](../multilingual/run_apertus_snr_variants.py) + [`analyze_snr_variants.py`](../multilingual/analyze_snr_variants.py) | The SNR-variant deep dive: 22 SNR variants × 4 sizes × 104 tasks ([`snr_variants_per_task.csv`](snr_definition/snr_variants_per_task.csv)). Six grid views (DA-size, DA-ckpt × {pooled, per-size}), two heatmaps per view (Pearson r per language), one 22×22 inter-variant correlation matrix, one DA-size-vs-DA-ckpt agreement scatter. Per-language slices are emitted for every language with ≥5 valid (task, size) cells. |
| `smooth_subtasks/` | [`smooth_subtasks.py`](../multilingual/smooth_subtasks.py) + [`smooth_subtasks_per_sample.py`](../multilingual/smooth_subtasks_per_sample.py) | Subset-search: for each multilingual benchmark family, rank its languages by per-language SNR and sweep cumulative subsets. Three CSV cuts: per-benchmark (Case 1), `global_mmlu_full` subjects pooled across 10 languages (Case 2), `global_mmlu_full` subjects per-language (Case 3). Plus per-(lang, benchmark) sample-level subset search using Option D from [`per_sample/PROPOSALS.md`](smooth_subtasks/per_sample/PROPOSALS.md) — 98 tasks, each with `summary.csv` + `ranked_samples.csv` + `cumulative_snr.png`. |

**Coverage caveats** documented in the code & READMEs:
- 3 mixes ⇒ DA is 4-level discrete (`{0, ⅓, ⅔, 1}`), so absolute Pearson-r
  magnitudes are capped well below 25-mix DataDecide. All `r` values should
  be read as ordinal evidence, not population correlations.
- de, fr, th have ≤4 valid (task, size) cells and are skipped from per-language
  panels. ja additionally drops out of DA-ckpt views.
- Half-trained mixes (600M-fwEdu90, 1B-fwEdu90, the three 175M-fwEdu*) reach
  fewer steps; the SNR pipeline tolerates this by keeping per-mix score arrays
  jagged ([`compute_snr_small_scale`](../snr/snr_simple.py)). Don't refactor
  back to a square `ndarray`.

## 2. Key findings (what the numbers say)

### F1 — Several SNR variants are algebraically redundant at n_mixes=3
With only 3 mixes the dispersion-family signals (`dispersion`, `range`, `mpd`,
`quartile_deviation`, `rms_deviation`) become linearly proportional — their
log10(SNR) values correlate at **r ≥ 0.999**. Two more redundancy clusters:
{`iqr`, `rel_dispersion`, `rel_mpd`, `rel_std`} (r ≥ 0.998) and
{`discrepancy`, `star_discrepancy`} (r ≈ 0.959). Effectively the 22 variants
collapse to ~12 distinct degrees of freedom in this experimental setup.

### F2 — Dispersion family is the best **global** default
Mean Pearson r across languages: top under DA-size is `mpd` /
`dispersion` / `range` / `quartile_deviation` (≈ +0.258). Top under DA-ckpt
is `dist_std` / `dispersion_shifted` / `aad` (≈ +0.23). Both DA definitions
agree the dispersion cluster is a sound default.

### F3 — But the **per-language** winner shifts substantially
| Language | Best variant family | Mean Pearson r |
|---|---|---|
| ru | dispersion cluster | **+0.68** (strongest in the dataset) |
| eu, es, sw, hi | `dispersion_shifted` / `dist_std` | +0.42 to +0.52 |
| tr | `mad` (outlier-robust spread) | +0.51 |
| en | `discrepancy` (DA-size, +0.15) → `mad` (DA-ckpt, +0.42) | DA-ckpt is the actionable signal for English |
| zh | best variant ≈ -0.02 (DA-size), +0.31 (DA-ckpt) | SNR is unreliable for zh under either DA |

### F4 — DA-size and DA-ckpt are not interchangeable
For `star_discrepancy_shifted`, `mad`, `gini`, `tukey`,
`dispersion_shifted`, `star_discrepancy` the DA-ckpt Pearson r is
0.08–0.12 higher than DA-size. **Practical rule** the data justifies:
choose DA-ckpt when the question is *"can I stop training early without
losing the right mix?"*; choose DA-size when *"can I extrapolate the right
mix to a larger model?"*

### F5 — Two variants are systematically anti-correlated
`projection_snr` and `tukey_snr` (depth-based formulations) sit at negative
mean r against DA across nearly every language and both DA definitions. They
should be excluded for Apertus-style 3-mix setups.

### F6 — `rel_std_snr` was degenerate; the broadcast fix made it usable
After broadcasting cross-mix std as `data_noise`,
`snr_rel_std_<size>` now spans roughly [0.06, 9] (was [0.97, 1.05]). Now sits
in the redundancy cluster with `iqr`/`rel_mpd`/`rel_dispersion`. (This is a
methods-section detail, not a hero-figure finding.)

### F7 — Subset search reduces benchmark size without losing SNR
[`smooth_subtasks/per_benchmark.csv`](smooth_subtasks/per_benchmark.csv) and
[`smooth_subtasks/global_mmlu_full.csv`](smooth_subtasks/global_mmlu_full.csv)
record `best_n` vs `full_set_snr`. The cumulative-SNR curves
([`per_benchmark_plots/`](smooth_subtasks/per_benchmark_plots/)) show that
across most multilingual families the SNR curve plateaus or peaks well before
all languages are included — i.e. a small "core set" of languages dominates
the family's ability to discriminate mixes. The same pattern holds for MMLU
subjects (Case 2) and per-language samples (per-sample tree).

## 3. Three options for the main figures

Each option below is a self-contained "hero figure" candidate. They embody
*different paper framings*; pick the option that matches the paper's
positioning. All three can coexist in the appendix; only one becomes the
front-page figure.

---

### Option A — "SNR predicts decision accuracy on multilingual evals"
*Framing: validation/proof-of-concept paper. Headline = the SNR framework
generalises from English DataDecide to multilingual + Apertus pretraining.*

A 4-panel figure (suggested 2×2, ~7" × 6"), single column-spanning width:

- **A1 (top-left).** Single big SNR-vs-DA scatter, all (variant=dispersion,
  size, task) pooled. Each point is one (task, size). Color = language,
  marker = size. Overlay the OLS log10(SNR)→DA line + Pearson r in the
  corner. Establishes the central positive trend.
- **A2 (top-right).** Per-language horizontal bar chart of Pearson r
  (DA-size, dispersion variant). Languages sorted by r. Asterisks for
  languages dropped due to insufficient coverage (de/fr/th). One glance at
  *who plays nice*.
- **A3 (bottom-left).** Variant × language Pearson r heatmap (RdBu, ±1
  anchored), reusing
  [`snr_definition/da_size/heatmap_pearson_r.png`](snr_definition/da_size/heatmap_pearson_r.png)
  but trimmed to the top-8 variants for readability. Annotate redundancy
  clusters with side brackets.
- **A4 (bottom-right).** "Practical payoff": for each language, plot DA at
  `best top-N tasks` vs DA at `all N tasks` (paired bars). Shows that the
  SNR ranking is actionable, not just descriptive.

**Pros.** Single figure carries the full contribution. Inherits visual
language from upstream DataDecide paper, so reviewers familiar with the
original SNR work read it instantly.
**Cons.** Bottom-right requires a small additional computation (DA on the
top-N task subset) we don't have a CSV for yet — would need to add a
`best_subset_da.csv` step before camera-ready. Density is high; A3
becomes unreadable below ~6" width.

**One-figure version (if pressed).** Drop A3 and A4; keep A1 + A2 side
by side. That's the minimum viable hero.

---

### Option B — "SNR definition matters; the right variant is language-dependent"
*Framing: methods/critique paper. Headline = the SNR scalar isn't unique,
and which definition you pick changes the conclusion you draw, especially
when you go beyond English.*

A 3-panel figure (suggested 1×3 row, ~9" × 3.5"):

- **B1.** Inter-variant correlation matrix
  ([`snr_definition/variant_correlation_matrix.png`](snr_definition/variant_correlation_matrix.png)),
  re-ordered to put redundancy clusters on the diagonal. Annotate the three
  blocks (dispersion / iqr-rel / discrepancy) with labels.
- **B2.** DA-size vs DA-ckpt agreement scatter
  ([`snr_definition/da_size_vs_da_ckpt.png`](snr_definition/da_size_vs_da_ckpt.png))
  with each variant labelled. Diagonal line = "both DA definitions agree."
  Highlight outliers (`mad`, `dispersion_shifted`, `tukey`) that move
  noticeably off-diagonal.
- **B3.** Two example SNR-vs-DA scatter columns side by side: ru (where the
  dispersion variant gives r ≈ +0.68) and zh (where every variant gives
  r ≈ 0). Same axes, same point style. The visual contrast carries the
  "language-dependent" claim without words.

**Pros.** Cleanly motivates an entire methods section ("how should we pick
an SNR variant?"). Three panels are independent enough that any one can
stand alone in a talk slide. All inputs already exist on disk.
**Cons.** Requires the reader to already accept that SNR is a useful scalar
— it doesn't sell the framework, it critiques it. Best paired with Option
A's headline scatter as Figure 2.

---

### Option C — "A small core of tasks/samples carries most of the signal"
*Framing: practical/efficiency paper. Headline = multilingual eval suites
are largely redundant; you can drop most of them and still rank data mixes
correctly.*

A 4-panel figure (suggested 2×2, ~7" × 6"):

- **C1 (top-left).** Combined-SNR-vs-subset-size curves overlaid for the 6
  multilingual families that have ≥5 languages (xnli, belebele, xcopa,
  global_piqa_completions, global_mmlu, xstorycloze). One panel; one curve
  per family at the 1B size. Shows where the curves peak vs `full_set_snr`.
- **C2 (top-right).** `best_n / full_n` as a small-multiples bar chart by
  size: shows the "compression ratio" — what fraction of the languages you
  actually need.
- **C3 (bottom-left).** Heatmap (benchmark × language) of "is this language
  in the best subset for this benchmark?" — sparse-looking matrix that
  immediately exposes which languages are the multilingual eval workhorses
  vs the noise-only ones. Pulled from `per_benchmark.csv:best_subset`.
- **C4 (bottom-right).** Cumulative-SNR curve for `global_mmlu_full`
  subjects (Case 2) plus the per-language curves for 3 representative
  languages (en, ru, zh). Demonstrates the same "core set" pattern at the
  *subject* level — supporting the per-sample story without needing an
  extra panel for it.

**Pros.** Most actionable framing for practitioners ("here's a 30%-size
benchmark that gives you 95% of the SNR"). Repurposes the
`smooth_subtasks/` outputs that no other paper-tier figure currently
showcases. Can be built entirely from existing CSVs.
**Cons.** Doesn't directly tell the reader what SNR *is* — assumes Option
A's framing is folded into the introduction. Subset selection is a greedy
single-pass (Option D in `PROPOSALS.md`); a reviewer may push for the
forward-greedy or IRT comparison from `PROPOSALS.md` before publication.

---

## 4. Recommendation

If the paper is one figure: **Option A**, because it sells the framework's
generalisation to multilingual + Apertus, which is the novel contribution
relative to upstream AllenAI.

If the paper has two main figures: **A** (Figure 1, the "it works" claim) +
**C** (Figure 2, the "and here's the practical payoff" claim).

Use **B** as Section-3 methods support if the paper carves out a serious
methodology contribution about SNR variant choice; otherwise relegate the B1
correlation matrix and B2 agreement scatter to the appendix and leave the
main story to A + C.
