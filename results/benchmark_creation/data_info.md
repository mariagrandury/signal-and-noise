# Benchmark metadata for `benchmark_creation/` analysis

Fill one row per benchmark family. `family` keys must match
`multilingual.analyze_snr_variants.benchmark_family` — i.e., the
prefix of the multilingual task name with the language token stripped.
The 12 families currently in scope (from
`collect_multilingual_families`) are listed below; replace the empty
cells with your description.

| family | data_source | curation_process | task_format | domain | n_languages |
|---|---|---|---|---|---|
| arc |  |  |  |  | 11 |
| belebele |  |  |  |  | 12 |
| global_mmlu |  |  |  |  | 6 |
| global_mmlu_full |  |  |  |  | 10 |
| global_piqa_completions |  |  |  |  | 10 |
| hellaswag |  |  |  |  | 8 |
| multiblimp |  |  |  |  | 7 |
| paws |  |  |  |  | 5 |
| xcopa |  |  |  |  | 6 |
| xnli |  |  |  |  | 11 |
| xstorycloze |  |  |  |  | 8 |
| xwinograd |  |  |  |  | 4 |

Notes:
- `data_source`: where the underlying QA / NLI / etc. items came from
  (e.g. "MMLU translated by Cohere", "human-authored", "PIQA
  translated by Google").
- `curation_process`: high-level method (machine translation, human
  translation, originally multilingual, expert-filtered, …).
- `task_format`: e.g. `mc` (multiple choice) / `gen` (free-form).
- `domain`: general-knowledge / STEM / commonsense / NLI / reading
  comprehension / etc.
- `n_languages`: how many per-language aggregates show up in the
  Apertus parquet (pre-filled — adjust if you change the language
  scope).

When this table is filled in, run the analysis described in
[INSTRUCTIONS.md](INSTRUCTIONS.md).
