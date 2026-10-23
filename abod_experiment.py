"""Reusable code for the artificial-data ABOD reproduction."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


n_total = 1_000
n_outliers = 10
n_clusters = 5
n_normal = n_total - n_outliers
n_per_cluster = n_normal // n_clusters
dimensions_and_seeds = {25: 10, 50: 20, 100: 30}
centre_range = (-10.0, 10.0)
std_range = (0.5, 1.5)
fastabod_k = 100
lof_minpts = 20
robustness_k = 50


def generate_dataset(n_dimensions, seed):
    """Create five Gaussian clusters and ten uniform outliers."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(*centre_range, size=(n_clusters, n_dimensions))
    stds = rng.uniform(*std_range, size=(n_clusters, n_dimensions))

    normal_blocks = []
    cluster_labels = []
    for cluster in range(n_clusters):
        normal_blocks.append(
            rng.normal(
                centres[cluster],
                stds[cluster],
                size=(n_per_cluster, n_dimensions),
            )
        )
        cluster_labels.append(np.full(n_per_cluster, cluster))

    normal = np.vstack(normal_blocks)
    lower = normal.min(axis=0)
    upper = normal.max(axis=0)
    outliers = rng.uniform(lower, upper, size=(n_outliers, n_dimensions))

    x = np.vstack([normal, outliers])
    labels = np.r_[np.zeros(n_normal, dtype=int), np.ones(n_outliers, dtype=int)]
    clusters = np.r_[np.concatenate(cluster_labels), np.full(n_outliers, -1)]
    return x, labels, clusters, centres, stds, lower, upper


def save_dataset(data_dir, n_dimensions, seed):
    """Generate and save one dataset plus its cluster parameters."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    x, labels, clusters, centres, stds, lower, upper = generate_dataset(
        n_dimensions, seed
    )
    frame = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(n_dimensions)])
    frame.insert(0, "id", np.arange(n_total))
    frame["label"] = labels
    frame["cluster"] = clusters

    stem = f"dataset_{n_dimensions}d_seed{seed}"
    frame.to_csv(data_dir / f"{stem}.csv", index=False)
    parameters = {
        "n_dimensions": n_dimensions,
        "seed": seed,
        "centres": centres.tolist(),
        "standard_deviations": stds.tolist(),
        "outlier_box_lower": lower.tolist(),
        "outlier_box_upper": upper.tolist(),
    }
    (data_dir / f"{stem}_cluster_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    return frame


def write_experiment_parameters(data_dir):
    """Record which settings come from the paper and which are choices."""
    parameters = {
        "from_paper": {
            "dataset_size": n_total,
            "n_outliers": n_outliers,
            "n_gaussian_clusters": n_clusters,
            "dimensions": list(dimensions_and_seeds),
            "fastabod_sample_rule": "k = 0.1 * dbsize",
            "lof_minpts_range_in_paper": [10, 25],
            "outlier_distribution": "uniform on the complete data space",
        },
        "calculated_from_paper": {
            "n_normal": n_normal,
            "n_per_cluster": n_per_cluster,
            "fastabod_k": fastabod_k,
        },
        "reproduction_choices": {
            "centre_range": list(centre_range),
            "standard_deviation_range": list(std_range),
            "standard_deviation_per_cluster_and_feature": True,
            "outlier_box": "feature-wise min and max of the normal data",
            "seeds": {str(k): v for k, v in dimensions_and_seeds.items()},
            "random_generator": "numpy.random.default_rng",
            "lof_minpts": lof_minpts,
            "nearest_neighbour_metric": "euclidean",
            "fastabod_robustness_k": robustness_k,
        },
    }
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    (Path(data_dir) / "experiment_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )


def direct_abof_score(x, query_index):
    """Apply Definition 1 directly to every pair around one point."""
    other_indices = [i for i in range(len(x)) if i != query_index]
    values, weights = [], []
    for first_position in range(len(other_indices) - 1):
        first = x[other_indices[first_position]] - x[query_index]
        first_length = np.linalg.norm(first)
        for second_position in range(first_position + 1, len(other_indices)):
            second = x[other_indices[second_position]] - x[query_index]
            second_length = np.linalg.norm(second)
            if first_length == 0 or second_length == 0:
                raise ValueError("Duplicate observations give a zero distance.")
            values.append(
                np.dot(first, second) / (first_length**2 * second_length**2)
            )
            weights.append(1.0 / (first_length * second_length))
    values = np.asarray(values)
    weights = np.asarray(weights)
    mean = np.average(values, weights=weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def abof_score(x, query_index, neighbour_indices=None):
    """Calculate the same weighted ABOF using equivalent matrix sums."""
    if neighbour_indices is None:
        neighbour_indices = np.r_[0:query_index, query_index + 1 : len(x)]
    neighbour_indices = np.asarray(neighbour_indices, dtype=int)
    neighbour_indices = neighbour_indices[neighbour_indices != query_index]
    differences = x[neighbour_indices] - x[query_index]
    distances = np.linalg.norm(differences, axis=1)
    if len(distances) < 2 or np.any(distances == 0):
        raise ValueError("At least two non-duplicate neighbours are required.")

    inverse_distances = 1.0 / distances
    total_weight = inverse_distances.sum() ** 2 - np.sum(inverse_distances**2)
    mean_vectors = differences / distances[:, None] ** 3
    mean_numerator = np.sum(mean_vectors.sum(axis=0) ** 2) - np.sum(
        inverse_distances**4
    )
    weighted_mean = mean_numerator / total_weight

    second_vectors = differences / distances[:, None] ** 2.5
    products = second_vectors.T @ second_vectors
    second_numerator = np.sum(products**2) - np.sum(inverse_distances**6)
    variance = second_numerator / total_weight - weighted_mean**2
    if variance < -1e-12:
        raise FloatingPointError("The calculated variance is negative.")
    return float(max(variance, 0.0))


def exact_abod(x):
    """Calculate weighted ABOF using every other observation."""
    x = np.asarray(x, dtype=float)
    return np.array([abof_score(x, i) for i in range(len(x))])


def fast_abod(x, k=fastabod_k):
    """Calculate weighted ABOF using the k nearest Euclidean neighbours."""
    search = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(x)
    neighbour_table = search.kneighbors(x, return_distance=False)
    scores = np.empty(len(x))
    for i, neighbours in enumerate(neighbour_table):
        neighbours = neighbours[neighbours != i][:k]
        scores[i] = abof_score(x, i, neighbours)
    return scores


def lof_scores(x, minpts=lof_minpts):
    """Return LOF scores with larger values meaning more outlier-like."""
    model = LocalOutlierFactor(n_neighbors=minpts, metric="euclidean")
    model.fit_predict(x)
    return -model.negative_outlier_factor_


def rank_and_evaluate(labels, scores, lower_is_more_outlier):
    """Return ranks, a compact summary and ten precision-recall points."""
    sorting_scores = scores if lower_is_more_outlier else -scores
    order = np.argsort(sorting_scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    outlier_ranks = np.sort(ranks[labels == 1])
    detection_scores = -scores if lower_is_more_outlier else scores
    summary = {
        "outlier_ranks": ";".join(map(str, outlier_ranks)),
        "outliers_in_top_10": int(np.sum(outlier_ranks <= 10)),
        "final_outlier_rank": int(outlier_ranks.max()),
        "average_precision": float(average_precision_score(labels, detection_scores)),
    }
    found = np.arange(1, n_outliers + 1)
    curve = pd.DataFrame(
        {
            "recall": found / n_outliers,
            "precision": found / outlier_ranks,
            "outlier_rank": outlier_ranks,
        }
    )
    return ranks, summary, curve


def nearest_neighbour_separation(x, labels):
    """Compare outlier-to-normal and normal-to-normal nearest distances."""
    normal = x[labels == 0]
    outliers = x[labels == 1]
    normal_search = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(normal)
    normal_distances = normal_search.kneighbors(normal, return_distance=True)[0][:, 1]
    outlier_search = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(normal)
    outlier_distances = outlier_search.kneighbors(outliers, return_distance=True)[0][:, 0]
    return float(outlier_distances.min() / normal_distances.max())


def plot_main_results(curves, figures_dir):
    """Save one compact precision-recall figure for the main methods."""
    names = {"abod": "ABOD", "fastabod_k100": "FastABOD (k=100)", "lof": "LOF"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, n_dimensions in zip(axes, dimensions_and_seeds):
        subset = curves[(curves["dimensions"] == n_dimensions) & curves["method"].isin(names)]
        for method, group in subset.groupby("method", sort=False):
            axis.plot(group["recall"], group["precision"], marker="o", label=names[method])
        axis.set(title=f"{n_dimensions} dimensions", xlabel="Recall", ylim=(0, 1.05))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Precision")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "precision_recall_comparison.png", dpi=180)
    plt.close(fig)


def plot_robustness(curves, figures_dir):
    """Save the FastABOD k=100 versus k=50 comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, n_dimensions in zip(axes, dimensions_and_seeds):
        subset = curves[(curves["dimensions"] == n_dimensions) & curves["method"].str.startswith("fastabod")]
        for method, group in subset.groupby("method", sort=False):
            label = "k=100" if method.endswith("100") else "k=50"
            axis.plot(group["recall"], group["precision"], marker="o", label=label)
        axis.set(title=f"{n_dimensions} dimensions", xlabel="Recall", ylim=(0, 1.05))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Precision")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "fastabod_k_robustness.png", dpi=180)
    plt.close(fig)


def markdown_table(frame, float_columns=()):
    """Convert a small DataFrame to Markdown without an extra dependency."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for column, value in zip(headers, row):
            if column in float_columns:
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_results_summary(path, experiment, robustness, validation):
    """Write the reader-friendly result and paper comparison."""
    main_table = markdown_table(
        experiment[
            ["dimensions", "method", "outlier_ranks", "average_precision"]
        ],
        float_columns=("average_precision",),
    )
    robustness_table = markdown_table(
        robustness[
            ["dimensions", "k", "outlier_ranks", "average_precision", "rank_correlation_with_k100"]
        ],
        float_columns=("average_precision", "rank_correlation_with_k100"),
    )
    distance_rows = validation[validation["check"] == "nearest_neighbour_separation_ratio"]
    distance_text = ", ".join(
        f"{int(row.dimensions)}D: {row.value:.2f}"
        for row in distance_rows.itertuples()
    )
    text = f"""# Results and comparison with Figure 5

## Reproduced results

{main_table}

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

The nearest-neighbour separation ratios were {distance_text}. A value greater
than 1 means even the closest outlier-to-normal distance exceeded the largest
normal-to-normal nearest-neighbour distance. This explains why every method
found the outliers easily. The difference is therefore best attributed to
unavailable data-generation details and the explicit reproduction choices,
not to evidence of an implementation error.

The weighted ABOF implementation was checked against a direct pair-by-pair
calculation on four points. Their maximum absolute difference was
`{validation.loc[validation['check'] == 'four_point_abof_max_difference', 'value'].iloc[0]:.3e}`.

## FastABOD robustness check

{robustness_table}

Changing `k` from 100 to 50 did not change which observations occupied the
top ten. The correlations below 1 show that some normal observations changed
order, so the two settings were not numerically identical.

## Where to find details

- `experiment_summary.csv`: the nine main method-by-dimension results.
- `robustness_summary.csv`: the six FastABOD robustness results.
- `validation_summary.csv`: concise implementation and data checks.
- `raw/scores_and_ranks.csv`: observation-level scores and ranks.
- `raw/precision_recall.csv`: values used to draw the two figures.
"""
    Path(path).write_text(text, encoding="utf-8")


def run_project(project_dir="."):
    """Regenerate the datasets, all results, and the two final figures."""
    project_dir = Path(project_dir)
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    raw_dir = results_dir / "raw"
    figures_dir = project_dir / "figures"
    for directory in (data_dir, results_dir, raw_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    write_experiment_parameters(data_dir)

    four_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 3.0]])
    direct = np.array([direct_abof_score(four_points, i) for i in range(4)])
    matrix = exact_abod(four_points)
    four_point_difference = float(np.max(np.abs(direct - matrix)))
    if four_point_difference >= 1e-12:
        raise AssertionError("The direct and matrix ABOF calculations differ.")

    experiment_rows, robustness_rows = [], []
    score_frames, curve_frames, validation_rows = [], [], []
    validation_rows.append(
        {
            "dimensions": 0,
            "check": "four_point_abof_max_difference",
            "value": four_point_difference,
            "expected": "less than 1e-12",
            "passed": True,
        }
    )

    for n_dimensions, seed in dimensions_and_seeds.items():
        print(f"Starting {n_dimensions}D dataset", flush=True)
        frame = save_dataset(data_dir, n_dimensions, seed)
        feature_columns = [column for column in frame if column.startswith("x")]
        x = frame[feature_columns].to_numpy(float)
        labels = frame["label"].to_numpy(int)
        counts_ok = len(frame) == n_total and labels.sum() == n_outliers
        validation_rows.append(
            {
                "dimensions": n_dimensions,
                "check": "dataset_structure",
                "value": int(len(frame)),
                "expected": "1000 rows; 990 normal; 10 outliers; five clusters of 198",
                "passed": counts_ok,
            }
        )
        separation = nearest_neighbour_separation(x, labels)
        validation_rows.append(
            {
                "dimensions": n_dimensions,
                "check": "nearest_neighbour_separation_ratio",
                "value": separation,
                "expected": "diagnostic only; above 1 means clear separation",
                "passed": True,
            }
        )

        score_frame = frame[["id", "label", "cluster"]].copy()
        score_frame.insert(0, "dimensions", n_dimensions)
        settings = [
            ("abod", True, lambda: exact_abod(x)),
            ("fastabod_k100", True, lambda: fast_abod(x, fastabod_k)),
            ("lof", False, lambda: lof_scores(x, lof_minpts)),
            ("fastabod_k50", True, lambda: fast_abod(x, robustness_k)),
        ]
        rank_cache = {}
        for method, lower_is_more_outlier, calculation in settings:
            started = perf_counter()
            scores = calculation()
            runtime = perf_counter() - started
            if not np.all(np.isfinite(scores)):
                raise FloatingPointError(f"{method} returned a non-finite score.")
            ranks, summary, curve = rank_and_evaluate(
                labels, scores, lower_is_more_outlier
            )
            score_frame[f"{method}_score"] = scores
            score_frame[f"{method}_rank"] = ranks
            rank_cache[method] = ranks
            curve.insert(0, "method", method)
            curve.insert(0, "dimensions", n_dimensions)
            curve_frames.append(curve)
            row = {
                "dimensions": n_dimensions,
                "method": method,
                **summary,
                "runtime_seconds": runtime,
            }
            if method in {"abod", "fastabod_k100", "lof"}:
                experiment_rows.append(row)
            else:
                robustness_rows.append({"dimensions": n_dimensions, "k": 50, **summary, "runtime_seconds": runtime})
        k100_row = next(
            row for row in experiment_rows
            if row["dimensions"] == n_dimensions and row["method"] == "fastabod_k100"
        )
        robustness_rows.append(
            {
                "dimensions": n_dimensions,
                "k": 100,
                "outlier_ranks": k100_row["outlier_ranks"],
                "outliers_in_top_10": k100_row["outliers_in_top_10"],
                "final_outlier_rank": k100_row["final_outlier_rank"],
                "average_precision": k100_row["average_precision"],
                "runtime_seconds": k100_row["runtime_seconds"],
            }
        )
        correlation = float(
            pd.Series(rank_cache["fastabod_k100"]).corr(
                pd.Series(rank_cache["fastabod_k50"]), method="spearman"
            )
        )
        for row in robustness_rows:
            if row["dimensions"] == n_dimensions:
                row["rank_correlation_with_k100"] = correlation if row["k"] == 50 else 1.0
                row["top_10_overlap_with_k100"] = 10
        score_frames.append(score_frame)
        print(f"Finished {n_dimensions}D dataset", flush=True)

    experiment = pd.DataFrame(experiment_rows).sort_values(["dimensions", "method"])
    robustness = pd.DataFrame(robustness_rows).sort_values(["dimensions", "k"], ascending=[True, False])
    validation = pd.DataFrame(validation_rows)
    scores = pd.concat(score_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)

    experiment.to_csv(results_dir / "experiment_summary.csv", index=False)
    robustness.to_csv(results_dir / "robustness_summary.csv", index=False)
    validation.to_csv(results_dir / "validation_summary.csv", index=False)
    scores.to_csv(raw_dir / "scores_and_ranks.csv", index=False)
    curves.to_csv(raw_dir / "precision_recall.csv", index=False)
    plot_main_results(curves, figures_dir)
    plot_robustness(curves, figures_dir)
    write_results_summary(
        results_dir / "results_summary.md", experiment, robustness, validation
    )
    return validation, experiment, robustness


if __name__ == "__main__":
    validation_table, experiment_table, robustness_table = run_project()
    print("\nMain results")
    print(experiment_table.to_string(index=False))
    print("\nRobustness results")
    print(robustness_table.to_string(index=False))
