# Results and comparison with Figure 5

## Reproduced results

| dimensions | method | outlier_ranks | average_precision |
| --- | --- | --- | --- |
| 25 | abod | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |
| 25 | fastabod_k100 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |
| 25 | lof | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |
| 50 | abod | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |
| 50 | fastabod_k100 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |
| 50 | lof | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |
| 100 | abod | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |
| 100 | fastabod_k100 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |
| 100 | lof | 1;2;3;4;5;6;7;8;9;10 | 1.000000 |

All three methods ranked the ten generated outliers in positions 1–10 in every
dataset, so their average precision was 1.0. This is the measured result for
the generated datasets in this repository; it is **not** the numerical result
shown in the paper.

## How this differs from the paper

Figure 5 reports a harder experiment:

- **25 dimensions:** ABOD was best, but all methods had difficulty as recall increased.
- **50 dimensions:** ABOD placed all ten outliers within the first eleven ranks; FastABOD found only two before a normal observation, and LOF performed poorly.
- **100 dimensions:** ABOD placed all ten outliers first; FastABOD performed poorly, while LOF improved at larger recall.

Therefore, the exact ABOD trend is partly reproduced—especially at 100
dimensions—but the present FastABOD and LOF results are substantially better
than Figure 5.

## Why the results differ

The paper gives the number of observations, clusters, outliers and dimensions,
but it does not provide the original datasets, random seeds, numerical centre
range or numerical standard-deviation range. This reproduction therefore uses
centres from `U[-10,10]`, standard deviations from `U[0.5,1.5]`, and seeds
10, 20 and 30. These choices produced compact, widely separated clusters.

The nearest-neighbour separation ratios were 25D: 3.57, 50D: 3.70, 100D: 3.99. A value greater
than 1 means even the closest outlier-to-normal distance exceeded the largest
normal-to-normal nearest-neighbour distance. This explains why every method
found the outliers easily. The difference is therefore best attributed to
unavailable data-generation details and the explicit reproduction choices,
not to evidence of an implementation error.

The weighted ABOF implementation was checked against a direct pair-by-pair
calculation on four points. Their maximum absolute difference was
`1.492e-16`.

## FastABOD robustness check

| dimensions | k | outlier_ranks | average_precision | rank_correlation_with_k100 |
| --- | --- | --- | --- | --- |
| 25 | 100 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 | 1.000000 |
| 25 | 50 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 | 0.991505 |
| 50 | 100 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 | 1.000000 |
| 50 | 50 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 | 0.983559 |
| 100 | 100 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 | 1.000000 |
| 100 | 50 | 1;2;3;4;5;6;7;8;9;10 | 1.000000 | 0.974947 |

Changing `k` from 100 to 50 did not change which observations occupied the
top ten. The correlations below 1 show that some normal observations changed
order, so the two settings were not numerically identical.

## Where to find details

- `experiment_summary.csv`: the nine main method-by-dimension results.
- `robustness_summary.csv`: the six FastABOD robustness results.
- `validation_summary.csv`: concise implementation and data checks.
- `raw/scores_and_ranks.csv`: observation-level scores and ranks.
- `raw/precision_recall.csv`: values used to draw the two figures.
