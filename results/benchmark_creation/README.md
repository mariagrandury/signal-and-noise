# `benchmark_creation/` — do high-SNR benchmarks share characteristics?

> Step-by-step plan: [INSTRUCTIONS.md](INSTRUCTIONS.md). Use one
> Claude session per research question — see
> [../PARALLEL_SESSIONS.md](../PARALLEL_SESSIONS.md).

## Research question

Do high-SNR benchmarks share traits? Start with **data source** and
**curation process**; other axes (format, domain, language family,
fewshot count, instance length) are out of scope for v0.

## Setup

- SNR signal comes from
  [../snr_definition/snr_variants_per_task.csv](../snr_definition/snr_variants_per_task.csv).
  The variant choice should be the one Question 1 picks as best (see
  [../snr_definition/README.md](../snr_definition/README.md)); default
  to `rel_std` if Question 1 hasn't decided.
- Benchmark metadata is hand-curated by the user in `data_info.md`
  (one row per family — keys must match
  `multilingual.analyze_snr_variants.benchmark_family`).
- Per-family aggregate: median / mean / max of `snr_<V>_1B` across
  the family's per-language tasks.

## Main results

*Pending — `data_info.md` is currently empty. Once it's populated,
`INSTRUCTIONS.md` Step 2 produces `per_family_snr.csv` and Step 3
emits `snr_by_data_source.png` + `snr_by_curation_process.png`. The
README will then summarise group-level findings.*

## Directory contents

- `INSTRUCTIONS.md` — execution plan.
- `data_info.md` — **user-provided**: per-family metadata table.
  Currently empty.
- *(produced by the analysis)*
  `per_family_snr.csv`,
  `snr_by_data_source.png`,
  `snr_by_curation_process.png`.
