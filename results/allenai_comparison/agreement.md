# Top-K reliability agreement

Variant used: **Discrepancy** (`discrepancy`)
Apertus SNR column: `snr_discrepancy_1B`  ·  AllenAI SNR column: `snr_discrepancy_1B`
Shared-task universe: **63** tasks.

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 3 | 0.60 | 0.43 | arc_easy, mmlu, mmlu_professional_law |
| 10 | 7 | 0.70 | 0.54 | arc_challenge, arc_easy, hellaswag, mmlu, mmlu_moral_scenarios, mmlu_professional_law, mmlu_professional_psychology |
| 20 | 13 | 0.65 | 0.48 | arc_challenge, arc_easy, hellaswag, mmlu, mmlu_elementary_mathematics, mmlu_high_school_mathematics, mmlu_high_school_psychology, mmlu_miscellaneous, mmlu_moral_disputes, mmlu_moral_scenarios, mmlu_nutrition, mmlu_professional_law, mmlu_professional_psychology |

## Top-20 per corpus

### Apertus

| task                         |    snr |
|:-----------------------------|-------:|
| arc_easy                     | 13.229 |
| mmlu_moral_scenarios         |  6.925 |
| arc_challenge                |  6.427 |
| mmlu_professional_law        |  5.97  |
| mmlu                         |  5.217 |
| hellaswag                    |  4.95  |
| mmlu_professional_psychology |  4.265 |
| mmlu_miscellaneous           |  2.815 |
| mmlu_prehistory              |  2.797 |
| mmlu_moral_disputes          |  2.7   |
| mmlu_human_sexuality         |  2.18  |
| mmlu_elementary_mathematics  |  1.97  |
| mmlu_high_school_psychology  |  1.945 |
| mmlu_high_school_biology     |  1.938 |
| mmlu_high_school_us_history  |  1.923 |
| mmlu_college_biology         |  1.86  |
| mmlu_college_mathematics     |  1.833 |
| mmlu_clinical_knowledge      |  1.818 |
| mmlu_high_school_mathematics |  1.788 |
| mmlu_nutrition               |  1.777 |

### AllenAI

| task                         |    snr |
|:-----------------------------|-------:|
| piqa                         | 47.692 |
| mmlu                         | 19.878 |
| hellaswag                    | 18.196 |
| arc_easy                     | 16.954 |
| mmlu_professional_law        | 14.206 |
| mmlu_moral_scenarios         |  9.511 |
| mmlu_security_studies        |  7.515 |
| arc_challenge                |  6.795 |
| mmlu_professional_accounting |  6.617 |
| mmlu_professional_psychology |  6.539 |
| mmlu_miscellaneous           |  6.446 |
| mmlu_moral_disputes          |  5.757 |
| mmlu_philosophy              |  5.367 |
| openbookqa                   |  5.334 |
| mmlu_nutrition               |  5.317 |
| mmlu_professional_medicine   |  5.293 |
| mmlu_elementary_mathematics  |  5.089 |
| mmlu_high_school_mathematics |  5.008 |
| mmlu_high_school_statistics  |  4.96  |
| mmlu_high_school_psychology  |  4.843 |
