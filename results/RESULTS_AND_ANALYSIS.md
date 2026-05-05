# Results and Analysis

We evaluate twelve Apertus pretraining runs spanning four model sizes
(175M, 350M, 600M and 1B parameters), three FineWeb-Edu data mixes
(`fwEdu30`, `fwEdu60`, `fwEdu90`, with the complement drawn from the
unfiltered FineWeb pool), all trained with the same random seed (1904)
and saved at thirteen intermediate checkpoints per run. The sections
below move from the raw learning curves to the signal-to-noise (SNR)
analysis that the curves motivate, then to a cross-corpus check against
the AllenAI DataDecide ladder, then to subset-search experiments that
attempt to lift SNR by pruning each multilingual benchmark, and finally
to a forward-looking question about what makes a benchmark high-SNR in
the first place.

## 5.1 Accuracy versus compute

This section asks how accuracy on each multilingual benchmark evolves
with compute across the twelve Apertus runs, and confirms that on most
benchmarks the three data mixes are not visibly separated by the eye at
any size we trained, which motivates an SNR-based approach to ranking
data mixes rather than a curve-by-curve inspection.

We use the standard `6 · N · T` compute proxy, where `N` is the
parameter count and `T` is the number of training tokens (Megatron's
configuration of 504 micro-batches × 4096 tokens per iteration). For
each checkpoint we report the primary `lm-eval` score (`acc,none` when
present, otherwise `exact_match,none`) on every multilingual task in
the parquet, dropping `acc_norm`, `acc_bytes` and the per-task standard
errors to match the schema used by the sister evaluation pipeline. We
arrange the resulting curves two ways: one figure per benchmark family
with subplots per language (`acc_vs_flops/per_benchmark/`), and one
figure per language with subplots per benchmark family
(`acc_vs_flops/per_language/`). Each subplot draws three curves, one
per data mix, and overlays the 1B target as a horizontal signal band.

The high-signal families separate the three data mixes by the 600M
scale: hellaswag, multiblimp and xstorycloze produce monotonically
rising curves with a visible spread between mixes well before the 1B
budget is exhausted. The remaining families are noisier than the
between-mix gap they are asked to resolve. Several language–task pairs
stay near the random baseline across the entire compute budget,
particularly the short global_mmlu variant, several xnli languages,
and most paws languages, indicating that compute alone cannot rescue an
under-resourced benchmark and that benchmark choice matters more than
training longer. The data-mix ordering visible by eye on the high-signal
families is informal evidence that data-mix differences leave a
measurable footprint on accuracy; the rest of this section quantifies
that footprint by computing SNR per task.

## 5.2 SNR definitions

This section sweeps twenty-two SNR variants and asks which one's
per-task SNR best tracks decision accuracy on Apertus, finding that
the dispersion family (`mpd`, `dispersion`, `range`,
`quartile_deviation`) gives the strongest correlation with decision
accuracy across languages while the upstream default `rel_std` lands
in a redundancy cluster with three other relative-spread variants.

Each variant defines a signal as some measure of cross-mix spread and a
noise as some measure of within-mix variation across the last five
checkpoints; the full list of formulas is given in
`snr_variants_definitions.csv`. We define decision accuracy two ways.
DA-size compares the rank of the three mixes at a small size against
their rank at the 1B target at the last checkpoint of each, which
captures whether a small-scale ranking transfers to scale. DA-ckpt
compares the rank of the three mixes at an early checkpoint against
their rank at the last checkpoint inside a single size, which captures
whether an early-stopping decision is reliable. With three mixes both
definitions are quantised to the four levels {0, ⅓, ⅔, 1}, so the
absolute Pearson r values are bounded well below the magnitudes
typically reported on the twenty-five-mix DataDecide ladder. We rank
variants by their mean Pearson correlation with decision accuracy
across the twelve languages in scope, and we read algebraic redundancy
off a 22 × 22 cross-variant correlation matrix.

The dispersion-family variants give a mean DA-size correlation of about
r ≈ +0.26 across languages, and the dispersion-shifted family takes
the lead on DA-ckpt at about r ≈ +0.23, both comfortably ahead of the
relative-spread cluster that includes the upstream default `rel_std`.
With only three mixes, `dispersion`, `range`, `mpd`,
`quartile_deviation` and `rms_deviation` are linearly proportional to
each other and produce identical correlations: practitioners should
treat them as a single variant rather than report all five. The best
variant differs noticeably across languages, with Russian peaking near
r ≈ +0.68 under the dispersion cluster, Turkish best served by the
outlier-robust `mad` at r ≈ +0.51, and English requiring the DA-ckpt
definition before any variant exceeds r ≈ +0.4. Two depth-based
variants (`projection_snr`, `tukey_snr`) are uniformly negatively
correlated with decision accuracy at every language and under both DA
definitions, and we recommend avoiding them on Apertus-style sweeps.
Among multilingual benchmarks, `multiblimp_<lang>` is the SNR leader
in every language for which it is available (ar, en, es, eu, hi, ru,
tr), often by a large margin (`multiblimp_hin` reaches SNR ≈ 16.9
against the next-best Hindi benchmark at ≈ 3.1); when multiblimp is
unavailable, `xstorycloze_<lang>` or `hellaswag_<lang>` typically
takes its place at the top.

## 5.3 Comparison with AllenAI DataDecide

This section asks whether the per-task SNR rankings learned on the
three-mix Apertus sweep transfer to the twenty-five-mix AllenAI
DataDecide ladder, and finds that discrepancy-family variants reach
r ≈ 0.7 across the sixty-one shared tasks while every relative-spread
variant — including the upstream default `rel_std` — is essentially
uncorrelated.

We rebuild the AllenAI per-task SNR table by running the same
twenty-two-variant sweep on the `allenai/signal-and-noise` core
parquet, then join it to the Apertus table on task name. Apertus only
ran the multilingual `global_mmlu_full_en_<subject>` view of MMLU on a
full checkpoint series, so we alias those names back to the vanilla
`mmlu_<subject>` keys used by AllenAI; without the alias the shared
universe collapses from sixty-one tasks to three. The headline
correlation is the Pearson r between `log10(snr_<V>_1B)` on each
corpus, computed over the sixty-one shared tasks. We also sweep the
correlation over four matched-size pairs (175M ↔ 150M, 350M ↔ 300M,
600M ↔ 750M, 1B ↔ 1B) so that we can ask at which size the cross-
corpus agreement is strongest. The single-mix HF reference models
(SmolLM3, Olmo-3, Apertus-8B) are excluded from this comparison
because their data-mix-spread term is undefined.

The headline number is r = 0.697 for the `discrepancy` variant, with
`star_discrepancy`, `rel_star_discrepancy`, `dispersion_shifted` and
`gini` close behind at r ≥ 0.55, while `rel_std`, `iqr`,
`rel_dispersion`, `rel_mpd` and `rel_mpsd` all fall below r = 0.06.
The relative-spread family does not transfer across corpora even
though it is the upstream default, and a practitioner who picks an
SNR variant on one corpus and applies it on another should prefer
discrepancy-style aggregators. Counter-intuitively, the cross-corpus
mean Pearson r is highest at the smallest matched-size pair (175M
↔ 150M, r = 0.517) and degrades monotonically up to the 1B ↔ 1B pair
(r = 0.257), because Apertus has only three mixes per size and one
of the three 1B mixes is half-trained, so the 1B SNR estimates are
the least statistically powerful. A practical consequence is that
benchmark vetting can be done at 175M instead of at 1B, saving an
order of magnitude of compute. Seven tasks (`arc_challenge`,
`arc_easy`, `hellaswag`, `mmlu`, `mmlu_moral_scenarios`,
`mmlu_professional_law`, `mmlu_professional_psychology`) appear in
the top ten of both corpora; we recommend them as the most
cross-corpus reliable benchmarks for ranking pretraining data mixes.

## 5.4 SNR subsets

This section asks whether a subset of a multilingual benchmark — fewer
languages, fewer MMLU subjects, or fewer document ids — can lift SNR
above the full-set baseline, and confirms that for nearly every
multilingual family a one-to-four element subset beats the full set,
with the largest gains on the noisiest families.

For each multilingual family we compute the standalone SNR of every
subtask, order the subtasks in descending SNR, and walk a cumulative
curve over the prefix subsets; the best subset is the prefix that
maximises the curve, and we report `snr_gain = best_snr − full_set_snr`.
A random-order baseline runs alongside the descending-SNR sweep to
confirm that the gain is not a sorting artefact. We repeat the sweep
in three flavours. The first treats each per-language aggregate as a
subtask of its multilingual family. The second treats each MMLU
subject as a subtask of `global_mmlu_full`, averaged across its ten
languages. The third refines the second to one MMLU subject inside
one language, asking whether the best MMLU subset is language-specific.
A fourth, cluster-only flavour replaces the subtask with the document
id, applies a variance pre-filter to drop "dead" samples, and walks
the same descending-SNR sweep over surviving samples.

The xcopa family is the most striking case. At 175M, 350M and 600M the
entire benchmark's signal collapses to `xcopa_tr` alone, every other
language adds noise, and the best-subset SNR jumps from 0.24 to 2.14 at
350M, a gain of 1.90 SNR points; at 1B the best subset shifts to
`xcopa_vi`. The Vietnamese global_mmlu_full at 600M is the largest
logical-subset gain in the sweep, lifting from SNR = 0.29 to 2.24 with
the two-subject subset {`high_school_computer_science`,
`college_computer_science`}. Already-saturated families
(`multiblimp`, `xstorycloze`) leave no headroom: any one-language
subset matches but does not exceed the full-set SNR, and the short
global_mmlu variant yields no usable signal at any size. Per-sample
search adds an order of magnitude more degrees of freedom than the
logical-subset case and pushes the top entries to SNR ≈ 3.46
(`xcopa_sw` at 1B, twelve document ids out of 500) from a full-set
baseline of 0.78. The practical message is that multilingual evaluation
suites should be filtered to high-SNR subsets before they are used to
rank data mixes; running every language is wasted compute at small
scale, and the language that carries the signal is sometimes a single
typologically-marked one (Turkish for xcopa) that an English-centric
practitioner would not have selected by intuition.

## 5.5 Benchmark creation

This section asks whether high-SNR benchmarks share traits along the
two simplest design axes — original data source and curation process —
to give practitioners a forward-looking heuristic when designing new
multilingual evaluations rather than only when filtering existing
ones.

We take the per-task SNR table from the SNR-definition sweep, restrict
to the 1B size and to the global-best variant, and aggregate to per-
family statistics (median, mean and maximum) across each family's
per-language tasks. Family-level metadata — data source, curation
process, task format, domain, year and language count — is hand-
curated against the twelve multilingual families currently in scope
(`arc`, `belebele`, `global_mmlu`, `global_mmlu_full`,
`global_piqa_completions`, `hellaswag`, `multiblimp`, `paws`,
`xcopa`, `xnli`, `xstorycloze`, `xwinograd`). Group-level differences
are summarised by a strip plot per categorical column (one dot per
family) and a one-way Kruskal-Wallis test, with the caveat that
twelve families is far too small a sample for the test to be more
than descriptive.

With twelve families this analysis is descriptive rather than
confirmatory. Two patterns nevertheless emerge already from the
SNR-definition sweep and speak directly to the question. First,
benchmarks built around language-specific phenomena rather than
translated content dominate the per-language top-five for every
language where they exist: `multiblimp`, the linguistic-acceptability
suite written natively in each target language, reaches SNR ≈ 16.9 on
Hindi against the next-best Hindi benchmark at SNR ≈ 3.1, and is the
SNR leader in every language for which Apertus ran it. Second,
machine-translated multiple-choice families sit at the bottom of the
SNR distribution at every size, and several languages of these
families have no usable signal at all on the 1B run; this includes
the short global_mmlu variant and most paws languages. A practitioner
designing a new multilingual benchmark should therefore prefer
linguistically-grounded native-language tasks over English-source
machine translations when the goal is to rank pretraining data mixes,
and should expect that benchmark format alone (multiple choice versus
free-form) is a weaker signal than benchmark provenance (translated
versus originally multilingual).
