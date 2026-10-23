# ABOD artificial-data reproduction

This repository reproduces the 1,000-observation artificial-data experiment
from Section 4.1 and Figure 5 of Kriegel, Schubert and Zimek (2008),
*Angle-Based Outlier Detection in High-dimensional Data*.

Start with the executed `abod_task2_reproduction.ipynb`. For a short reading
of the outcome and its differences from the paper, open
`results/results_summary.md`.

## What is included

- `abod_task2_reproduction.ipynb` — the explained and executed experiment.
- `abod_experiment.py` — reusable data-generation and method code.
- `data/` — the three generated datasets and their parameters.
- `results/results_summary.md` — the reader-friendly result and paper comparison.
- `results/experiment_summary.csv` — nine main method-by-dimension results.
- `results/robustness_summary.csv` — the `k=100` versus `k=50` results.
- `results/validation_summary.csv` — essential implementation and data checks.
- `results/raw/` — observation-level scores, ranks and plotting values.
- `figures/` — the main precision-recall and robustness figures.
- `requirements.txt` — required Python packages.

## Experimental settings

Each dataset contains 1,000 observations: 990 normal observations split equally
between five Gaussian clusters and ten uniform outliers. The dimensions are 25,
50 and 100. Exact ABOD uses all other observations, FastABOD uses `k=100`, and
LOF uses `minpts=20`. Nearest-neighbour searches use Euclidean distance.

The paper does not provide the original datasets, random seeds, numerical
centre range or numerical standard-deviation range. The seeds 10/20/30, centres
from `U[-10,10]`, standard deviations from `U[0.5,1.5]`, and the precise
feature-wise outlier box are explicit reproduction choices. Full settings are
recorded in `data/experiment_parameters.json`.

## Main result

In the generated datasets, exact ABOD, FastABOD and LOF all ranked the ten
outliers in positions 1–10 at every dimensionality. This is stronger than
Figure 5 and does not reproduce its numerical curves. The generated data are
especially easy: the nearest-neighbour separation ratios are 3.57, 3.70 and
3.99 for 25D, 50D and 100D.

The weighted ABOF code agrees with a direct pair-by-pair implementation on four
points to a maximum absolute difference of about `1.49e-16`. This supports the
implementation, while the separation diagnostic supports unavailable
data-generation details and reproduction choices as the main explanation for
the different results.

FastABOD with `k=50` gives the same top-ten outliers. The complete-rank
correlations with `k=100` are 0.992, 0.984 and 0.975.

## Run the project

Install the packages:

```bash
python3 -m pip install -r requirements.txt
```

Then run all notebook cells, or run the reusable file from the repository
folder:

```bash
python3 abod_experiment.py
```

Either route regenerates the datasets, consolidated result tables and figures.

## Reference

Kriegel, H.-P., Schubert, M. and Zimek, A. (2008). *Angle-Based Outlier
Detection in High-dimensional Data*. Proceedings of the 14th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining.
