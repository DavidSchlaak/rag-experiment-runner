# Human evaluation: significance testing

## Input files
- Human_Evaluation_Qwen3-235B_top10_(1).xlsx
- Human_Evaluation_Qwen3-235B_top10_(2).xlsx
- Human_Evaluation_Qwen3-235B_top10_(3).xlsx
- Human_Evaluation_Admin_Mapping_Qwen3-235B_top10.xlsx (response anonymization key)

N = 10 paired test cases; evaluators = 3; alpha = 0.05.
Within each test case, ratings/ranks are averaged across evaluators before Friedman/Wilcoxon.
Likert Friedman p-values are exact, then Holm-adjusted across five dimensions.
Pairwise Wilcoxon p-values are Holm-adjusted within each outcome.
Post-hoc comparisons are interpreted only following a significant omnibus test.

## Table 1. Likert ratings, reliability, and Friedman tests

| Dimension | DPC M (SD) | DP M (SD) | GT M (SD) | Krippendorff alpha | Friedman Q | Exact Holm-p | Kendall W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Coverage (Broad) | 4.87 (0.35) | 4.80 (0.61) | 3.93 (0.91) | 0.907 | 10.138 | 0.016 | 0.507 |
| Coherence (Logical) | 4.97 (0.18) | 4.83 (0.38) | 4.03 (0.67) | 0.580 | 14.312 | 0.002 | 0.716 |
| User Intent Alignment | 4.90 (0.31) | 4.77 (0.73) | 3.97 (0.93) | 0.989 | 8.720 | 0.016 | 0.436 |
| Cyclicality | 4.80 (0.41) | 4.67 (0.48) | 3.80 (0.89) | 0.432 | 10.000 | 0.016 | 0.500 |
| Helpfulness | 4.83 (0.38) | 4.70 (0.65) | 3.87 (0.94) | 0.845 | 8.061 | 0.016 | 0.403 |

## Table 2. Ranking results

| System | Mean rank (SD) | First place n (%) | Last place n (%) |
| --- | --- | --- | --- |
| DPC | 1.43 (0.57) | 18 (60.0%) | 1 (3.3%) |
| DP | 1.93 (0.69) | 8 (26.7%) | 6 (20.0%) |
| GT | 2.63 (0.72) | 4 (13.3%) | 23 (76.7%) |

## Ranking: inter-rater reliability

Mean Kendall W = 0.889 (SD = 0.216; Median = 1.000; Min-Max = 0.333-1.000).

## Ranking: Friedman test

Q = 7.538, exact p = 0.021, Kendall W = 0.377.

## Pairwise post-hoc tests for significant Friedman outcomes

### Coverage (Broad)

- DPC vs DP: Wilcoxon W = 1.000; Holm-p = 1.000; rank-biserial r = 0.333.
- DPC vs GT: Wilcoxon W = 3.000; Holm-p = 0.059; rank-biserial r = 0.867.
- DP vs GT: Wilcoxon W = 8.000; Holm-p = 0.195; rank-biserial r = 0.644.

### Coherence (Logical)

- DPC vs DP: Wilcoxon W = 0.000; Holm-p = 0.250; rank-biserial r = 1.000.
- DPC vs GT: Wilcoxon W = 0.000; Holm-p = 0.012; rank-biserial r = 1.000.
- DP vs GT: Wilcoxon W = 2.000; Holm-p = 0.012; rank-biserial r = 0.927.

### User Intent Alignment

- DPC vs DP: Wilcoxon W = 0.000; Holm-p = 1.000; rank-biserial r = 1.000.
- DPC vs GT: Wilcoxon W = 2.500; Holm-p = 0.117; rank-biserial r = 0.861.
- DP vs GT: Wilcoxon W = 7.000; Holm-p = 0.281; rank-biserial r = 0.611.

### Cyclicality

- DPC vs DP: Wilcoxon W = 12.500; Holm-p = 0.500; rank-biserial r = 0.306.
- DPC vs GT: Wilcoxon W = 0.000; Holm-p = 0.023; rank-biserial r = 1.000.
- DP vs GT: Wilcoxon W = 1.500; Holm-p = 0.023; rank-biserial r = 0.933.

### Helpfulness

- DPC vs DP: Wilcoxon W = 2.500; Holm-p = 0.500; rank-biserial r = 0.500.
- DPC vs GT: Wilcoxon W = 4.000; Holm-p = 0.082; rank-biserial r = 0.822.
- DP vs GT: Wilcoxon W = 11.000; Holm-p = 0.180; rank-biserial r = 0.600.

### Ranking

- DPC vs DP: Wilcoxon W = 13.000; Holm-p = 0.199; rank-biserial r = -0.527.
- DPC vs GT: Wilcoxon W = 2.000; Holm-p = 0.047; rank-biserial r = -0.911.
- DP vs GT: Wilcoxon W = 11.000; Holm-p = 0.199; rank-biserial r = -0.600.

## Statistical notes

- Higher Likert ratings are better; lower ranking positions are better.
- The sign of rank-biserial r follows first system minus second system.
- CSV outputs retain unrounded statistics, including gated/non-significant comparisons.
