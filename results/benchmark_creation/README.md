# `benchmark_creation/` — do high-SNR benchmarks share characteristics?

> Step-by-step plan: [INSTRUCTIONS.md](INSTRUCTIONS.md). Use one
> Claude session per research question — see
> [../PARALLEL_SESSIONS.md](../PARALLEL_SESSIONS.md).

## Research question

Do high-SNR benchmarks share traits? Start with **data source** and
**curation process**; other axes (format, domain, language family,
fewshot count, instance length) are out of scope for v0.

## Setup

- **SNR signal:** `snr_mpd_1B` from
  [../snr_definition/snr_variants_per_task.csv](../snr_definition/snr_variants_per_task.csv).
  Q1 picked `mpd` (mean pairwise distance) as the headline variant —
  highest mean Pearson r vs decision accuracy across languages, sitting
  in the dispersion redundancy cluster
  ([snr_definition/README.md](../snr_definition/README.md)).
- **Per-family aggregate:** median, mean, max of `snr_mpd_1B` across
  the family's per-language aggregate tasks (the `task` rows that pass
  `multilingual.smooth_subtasks._is_language_aggregate`).
- **Metadata:** [data_info.md](data_info.md) — paper-style paragraph +
  schema table per family, cross-referenced against the
  `lm-evaluation-harness` task READMEs for the exact dataset each task
  pulls from. The table is mirrored into the `FAMILY_META` dict in
  [analyze.py](analyze.py) so the analysis works from a single source
  of categorical labels.
- **Two grouping views:**
  - `curation_category` (5-way): `machine_translation`,
    `mt_post_edited`, `human_translation`, `template_generated`,
    `originally_multilingual`.
  - `source_origin` (2-way): `english_translated` (English source
    dataset translated outward; includes Belebele since FLORES-200
    passages are English-sourced) vs `originally_multilingual`
    (authored / aggregated natively per language).
- **Per-task curation override:** `xnli_eu` is XNLIeu (Heredia et al.
  2024, MT + post-edit) and is re-tagged at the per-task level; the
  rest of XNLI stays as `human_translation`.

## Coverage caveat

11 of 12 expected families produced an SNR value. **`global_mmlu`**
(Lite, 6 langs) is excluded: the parquet contains only one Apertus
model evaluated on it (350M, single config), so `mpd_1B` is NaN for
every `global_mmlu_<lang>` row. Several `_de` / `_fr` per-language
aggregates in `arc` and `hellaswag` are also NaN at 1B for the same
reason (matches the snr_definition note that de/fr/th have ≤4 valid
size cells). The `global_piqa_completions_spa_latn_spai` row is
filtered by `_is_language_aggregate` because it has three trailing
tokens after the family prefix; this is a known edge case in the Q1
helper.

Net: **88 per-language aggregate tasks across 11 families** feed the
analysis.

## Main results

### Headline ranking

![Per-family SNR ranking colored by curation category](snr_per_family_ranked.png)

Per-family median `snr_mpd_1B`:

| family | median | n_tasks | curation |
|---|---:|---:|---|
| multiblimp | 4.42 | 7 | template_generated |
| xstorycloze | 2.04 | 8 | human_translation |
| hellaswag | 2.04 | 6 | machine_translation |
| xwinograd | 1.56 | 4 | originally_multilingual |
| global_piqa_completions | 1.25 | 10 | originally_multilingual |
| paws | 1.05 | 5 | human_translation |
| xcopa | 0.92 | 6 | human_translation |
| xnli | 0.91 | 11 | human_translation |
| arc | 0.74 | 9 | machine_translation |
| belebele | 0.48 | 12 | human_translation |
| global_mmlu_full | 0.40 | 10 | mt_post_edited |

Full table with mean / max / data_source: [per_family_snr.csv](per_family_snr.csv).

### By curation process

![Per-family SNR by curation process](snr_by_curation_process.png)

Family-level Kruskal-Wallis across the 5 curation categories:
**H = 0.84, p = 0.66** — not significant. With only 11 families spread
across 5 categories (template_generated and mt_post_edited each have
n=1) the test has effectively no power, but the visual still shows:

- **`template_generated`** (multiblimp, single family but with 7
  per-language tasks, all > 2.5 SNR): a clear high-SNR outlier driven
  by *task design*, not curation. Minimal-pair tasks compare two
  sentences that differ in a single morphological feature, so any
  model that has captured that feature gives a sharp probability
  difference and the SNR is clean by construction.
- **`machine_translation`** (arc, hellaswag): wide spread — hellaswag
  median 2.04 vs arc 0.74. ChatGPT-translated benchmarks are not
  uniformly low-SNR; whatever HellaSwag's design contributes to its
  ~2× advantage over ARC survives the MT step.
- **`human_translation`** (5 families): wide spread too — xstorycloze
  2.04 down to belebele 0.48. The two families with the cleanest
  log-likelihood task design (xstorycloze: pick 1 of 2 endings; paws:
  binary paraphrase) sit at the top of this group; xnli (3-class
  entailment) and belebele (4-option MRC over a long passage) sit at
  the bottom.
- **`originally_multilingual`** (xwinograd, global_piqa_completions):
  median 1.41 — middle of the pack. Native authoring doesn't produce
  uniformly higher SNR than translation.
- **`mt_post_edited`** (global_mmlu_full only, single family): the
  lowest median. Plausibly because Global-MMLU-Full's per-item
  translation quality varies (mix of MT, crowd post-edit, expert
  post-edit) and the 57-subject MMLU spread further fragments any
  per-item signal.

**Per-task version** (with `xnli_eu` re-tagged):

![Per-task SNR by curation process](snr_by_curation_per_task.png)

The per-task plot pools across families and exposes within-group
spread. Per-task Kruskal-Wallis: **H = 32.3, p < 1e-6**, but this is
almost entirely driven by multiblimp tasks pulling `template_generated`
upward; if you drop multiblimp the residual differences across the
other four categories are not significant.

### By source origin (English-translated vs originally-multilingual)

![Per-family SNR by source origin](snr_by_data_source.png)

Family-level Kruskal-Wallis: **H = 2.67, p = 0.10** — directionally
suggestive (originally-multilingual median ≈ 1.56 vs English-translated
≈ 0.92) but does not reach significance with n=3 vs n=8. The
originally-multilingual side is dragged up by multiblimp; remove
multiblimp and the medians flatten.

### Group statistics

| view | H | p | n_groups (n≥2) |
|---|---:|---:|---:|
| family / curation_category | 0.84 | 0.66 | 3 |
| family / source_origin | 2.67 | 0.10 | 2 |
| task / curation_category | 32.3 | <1e-6 | 5 |

(See [group_stats.csv](group_stats.csv).)

## Takeaways

1. **The strongest single predictor of SNR in this dataset is task
   design, not curation.** MultiBLiMP's template-generated minimal
   pairs sit at SNR ≈ 4.4 — about 2× the next family. This is a
   binary-log-likelihood comparison between two minimally-different
   sentences, which is a fundamentally different signal than
   classification or completion accuracy. Don't read it as "automatic
   generation curates better data than humans."
2. **Curation method alone is not predictive once you control for
   task format.** Among the comparable mid-SNR families
   (xstorycloze, hellaswag, xwinograd, global_piqa, paws, xcopa,
   xnli, arc, belebele), curation labels — MT, MT+post-edit, human
   translation, native authoring — interleave freely. hellaswag (MT)
   matches xstorycloze (human translation) at SNR 2.04; arc (MT) is
   higher than belebele (human translation) despite the latter's
   reputation as a gold-standard human-curated benchmark.
3. **`global_mmlu_full` is the lowest-SNR family.** This is the only
   family using mixed MT + post-edit curation, and the only one with
   the 57-subject MMLU spread. Both confounds point the same way; we
   can't isolate which is responsible without a same-curation,
   same-format counterpart.
4. **The Belebele surprise.** Belebele has the strongest curation
   pedigree of any family here (fully human, end-to-end parallel
   construction by bilingual experts) and it lands second-from-last
   at SNR 0.48. The likely culprit is task format: 4-option MRC over
   a 100-word passage gives weaker per-item log-likelihood
   discrimination than 2-option completion (xstorycloze, hellaswag,
   xwinograd) at this scale. Worth a follow-up controlled comparison.
5. **Heterogeneous within-family curation matters less than expected.**
   `xnli_eu` (XNLIeu, MT + post-edit) is the lowest XNLI per-language
   SNR, but only by about 0.1 — within the family's normal spread.
   The family-level aggregate is therefore robust to this single
   outlier; we don't need to split the family.

**Recommended follow-up to actually answer the headline question:**
hold task format constant and compare curation methods within it.
E.g., compare hellaswag (MT) vs xstorycloze (human translation) — both
2-option completion. Or compare ARC (MT) vs Belebele (human MRC) at
matched difficulty. The current 11-family pool spans too many
task-format axes simultaneously to attribute SNR variance to curation
alone.

## Directory contents

- [INSTRUCTIONS.md](INSTRUCTIONS.md) — execution plan from the
  parallel-sessions split.
- [data_info.md](data_info.md) — per-family paper-style paragraphs +
  the schema table that drives `FAMILY_META` in `analyze.py`.
- [analyze.py](analyze.py) — runs Steps 2–4 of `INSTRUCTIONS.md`:
  loads the Q1 SNR table, joins metadata, emits CSVs and plots.
- [per_family_snr.csv](per_family_snr.csv) — one row per family with
  median / mean / max `snr_mpd_1B` plus all metadata columns.
- [per_task_snr.csv](per_task_snr.csv) — one row per per-language
  aggregate task, carrying the per-task curation override (xnli_eu
  re-tagged as `mt_post_edited`).
- [group_stats.csv](group_stats.csv) — Kruskal-Wallis (H, p, n_groups)
  for the three group views.
- [snr_per_family_ranked.png](snr_per_family_ranked.png) — headline
  ranked bar chart, color = curation category.
- [snr_by_curation_process.png](snr_by_curation_process.png) — strip
  plot of family medians by curation category.
- [snr_by_data_source.png](snr_by_data_source.png) — strip plot of
  family medians by source origin (English-translated vs originally-
  multilingual).
- [snr_by_curation_per_task.png](snr_by_curation_per_task.png) —
  per-task strip plot with the `xnli_eu` curation override applied.
