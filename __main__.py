import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import seaborn as sns
import igraph as ig

from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from typing import List, Optional

from data import Dataset, load_datasets
from metric import MinDasguptaMetric, AverageDasguptaMetric, MeanDasguptaMetric, MetricEvaluator
from local_change import simulate_single_trajectory
from matrix import IGraphWeightMatrix

sns.set_theme(style="whitegrid")


def get_args():
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--resources", "-r", type=str, required=True,
        help="Directory containing go-basic.obo, GAF files, and BioGRID tables",
    )
    common_parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Species model: HUMAN, YEAST, MOUSE, FRUIT FLY (default: all)",
    )
    common_parser.add_argument(
        "--max-genes", "-g", type=int, default=1000,
        help="Maximum number of genes to use",
    )
    common_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    biological_contrast_parser = subparsers.add_parser("biological-contrast", parents=[common_parser])
    biological_contrast_parser.add_argument(
        "--bootstrap-reps", "-b", type=int, default=50,
        help="Number of bootstrap resamples over genes for 95%% CI (0 to disable)",
    )
    biological_contrast_parser.add_argument(
        "--save", "-s", type=str, default="biological_contrast.pdf",
        help="Output path for figure",
    )

    monotonicity_parser = subparsers.add_parser("perturbation-monotonicity", parents=[common_parser])
    monotonicity_parser.add_argument(
        "--rounds", "-R", type=int, default=50,
        help="Number of perturbation rounds/trajectories",
    )
    monotonicity_parser.add_argument(
        "--steps", "-k", type=int, default=20,
        help="Number of perturbation steps per trajectory",
    )
    monotonicity_parser.add_argument(
        "--jobs", "-j", type=int, default=None,
        help="Number of parallel worker processes (default: all cores - 2)",
    )
    monotonicity_parser.add_argument(
        "--samples", type=int, default=10000,
        help="Sample size for Monte Carlo distance calculation",
    )
    monotonicity_parser.add_argument(
        "--epsilon", type=float, default=0.15,
        help="Epsilon parameter for local change operators",
    )
    monotonicity_parser.add_argument(
        "--save", "-s", type=str, default="perturbation_monotonicity.txt",
        help="Text file for the results table, appended after each dataset",
    )

    return parser.parse_args()


def evaluate_biological_contrast(datasets: List[Dataset], save_path: str, n_boot: int = 50, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)

    p_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

    series = []
    for dataset in datasets:
        elements = dataset.elements
        dag = dataset.dag
        matrix = dataset.matrix

        if hasattr(matrix, "_graph"):
            orig_matrix = matrix
        elif isinstance(matrix, ig.Graph):
            orig_matrix = IGraphWeightMatrix(matrix)
        else:
            orig_matrix = matrix

        eval_metrics = [MinDasguptaMetric(), AverageDasguptaMetric()] + [MeanDasguptaMetric(p) for p in p_values]

        evaluator = MetricEvaluator(eval_metrics)
        print(f"\nEvaluating {dataset.name} across {len(eval_metrics)} metrics...")
        _, raw_matrices = evaluator.evaluate(elements, dag, orig_matrix)

        W = raw_matrices["W"]
        Hp_dict = raw_matrices["Hp"]
        min_lca_mat = raw_matrices["min_lca"]
        avg_lca_mat = raw_matrices["avg_lca"]

        n = len(elements)
        tril_i, tril_j = np.tril_indices(n, k=-1)
        y_true = (W[tril_i, tril_j] >= 0.5).astype(int)
        prevalence = float(np.mean(y_true))
        print(
            f"Dataset {dataset.name}: {n} genes | Pairs: {len(y_true):,} | "
            f"Interacting (W >= 0.5): {int(y_true.sum()):,} ({prevalence * 100:.2f}%)"
        )

        score_mats = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            score_mats["lca-min"] = np.where(min_lca_mat > 0, 1.0 / min_lca_mat, 0.0)
            score_mats["lca-avg"] = np.where(avg_lca_mat > 0, 1.0 / avg_lca_mat, 0.0)
            for p in p_values:
                score_mats[p] = np.where(Hp_dict[p] > 0, 1.0 / Hp_dict[p], 0.0)

        keys = ["lca-min", "lca-avg", *p_values]
        stats = {
            key: {"ap": float(average_precision_score(y_true, score_mats[key][tril_i, tril_j]))}
            for key in keys
        }

        boot_prs = {key: [] for key in keys}
        if n_boot > 0:
            print(f"Computing 95% bootstrap confidence intervals over {n} genes ({n_boot} resamples)...")
            y_mat = ((W + W.T) >= 0.5).astype(np.int8)
            for _ in tqdm(range(n_boot), desc=f"Bootstrap CI ({dataset.name})", unit="rep"):
                boot_idx = np.random.choice(n, size=n, replace=True)
                b_i = np.maximum(boot_idx[tril_i], boot_idx[tril_j])
                b_j = np.minimum(boot_idx[tril_i], boot_idx[tril_j])
                valid = b_i != b_j
                b_i = b_i[valid]
                b_j = b_j[valid]
                y_b = y_mat[b_i, b_j]
                if y_b.sum() == 0 or y_b.sum() == len(y_b):
                    continue
                for key in keys:
                    boot_prs[key].append(average_precision_score(y_b, score_mats[key][b_i, b_j]))

        for key in keys:
            samples = boot_prs[key]
            if len(samples) > 1:
                stats[key]["ci"] = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
            else:
                stats[key]["ci"] = (stats[key]["ap"], stats[key]["ap"])

        print(f"{dataset.name} Average Precision (AP)")
        for key in keys:
            row = stats[key]
            label = key if isinstance(key, str) else f"p={key:.1f}"
            if n_boot > 0:
                print(f"  {label:<8} {row['ap']:.4f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]")
            else:
                print(f"  {label:<8} {row['ap']:.4f}")

        series.append({"name": dataset.name, "stats": stats, "prevalence": prevalence})

    ncols = 2
    nrows = max(2, (len(series) + ncols - 1) // ncols)
    fig, axes = \
        plt.subplots(nrows, ncols, figsize=(12.4, 4.6 * nrows), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    p_arr = np.array(p_values, dtype=float)

    for index, item in enumerate(series):
        ax = axes_flat[index]
        stats = item["stats"]
        prs = np.array([stats[p]["ap"] for p in p_values])
        pr_low = np.array([stats[p]["ci"][0] for p in p_values])
        pr_high = np.array([stats[p]["ci"][1] for p in p_values])

        ax.plot(p_arr, prs, marker="s", linewidth=2.2, color="tab:red", label=r"GO DAG ($1/H_p$)")
        if n_boot > 0:
            ax.fill_between(p_arr, pr_low, pr_high, color="tab:red", alpha=0.20, label="95% Bootstrap CI")
        ax.axhline(
            stats["lca-avg"]["ap"], linestyle="--", color="tab:blue", alpha=0.85, linewidth=1.5,
            label=f'$c_{{\\text{{trans}}}}$ / Pan et al. ({stats["lca-avg"]["ap"]:.3f})',
        )
        ax.axhline(
            stats["lca-min"]["ap"], linestyle="-.", color="tab:orange", alpha=0.85, linewidth=1.5,
            label=f'$c_{{\\text{{min}}}}$ / MCA-min ({stats["lca-min"]["ap"]:.3f})',
        )
        ax.axhline(
            item["prevalence"], linestyle=":", color="gray", alpha=0.85, linewidth=1.5,
            label=f'Random Baseline ({item["prevalence"]:.3f})',
        )
        ax.set_xticks(p_arr)
        ax.set_title(item["name"], fontsize=12, fontweight="bold")
        ax.legend(fontsize=7.5, loc="best")
        if index % ncols == 0:
            ax.set_ylabel("Average Precision (AP)", fontsize=11)
        if index >= (nrows - 1) * ncols:
            ax.set_xlabel("Power-mean parameter (p)", fontsize=11)

    for ax in axes_flat[len(series):]:
        ax.set_visible(False)

    plt.savefig(save_path, bbox_inches="tight")
    png_path = save_path.rsplit(".", 1)[0] + ".png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved biological contrast publication figures to:\n  - {save_path}\n  - {png_path}")

    return {item["name"]: item["stats"] for item in series}


def evaluate_perturbation_monotonicity(datasets: List[Dataset], save_path: str, rounds: int = 50, steps: int = 20, n_jobs: Optional[int] = None, num_samples: int = 10000, epsilon: float = 0.15, seed: Optional[int] = None):
    metric_evaluator = MetricEvaluator([MinDasguptaMetric(), AverageDasguptaMetric(), MeanDasguptaMetric(1), MeanDasguptaMetric(0), MeanDasguptaMetric(-1), MeanDasguptaMetric(-2)])

    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 4) - 2)
    actual_jobs = min(n_jobs, rounds)
    metrics = metric_evaluator.metrics
    root, ext = os.path.splitext(save_path)
    txt_path = save_path if ext.lower() == ".txt" else f"{root}.txt"
    with open(txt_path, "w", encoding="utf-8"):
        pass

    series = []
    for dataset in datasets:
        report_lines: List[str] = []

        def emit(line: str = "") -> None:
            print(line)
            report_lines.append(line)

        elements = dataset.elements
        dag = dataset.dag
        matrix = dataset.matrix

        n = len(elements)
        if isinstance(matrix, np.ndarray):
            W = np.tril(matrix, k=-1) if matrix.shape == (n, n) else matrix
        elif hasattr(matrix, "to_dataframe"):
            mat_df = matrix.to_dataframe(elements)
            W = np.tril(mat_df.to_numpy(dtype=float), k=-1)
        elif isinstance(matrix, pd.DataFrame):
            W = np.tril(matrix.loc[elements, elements].to_numpy(dtype=float), k=-1)
        else:
            W = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i):
                    W[i, j] = matrix.at(elements[i], elements[j])

        sep_len = 120
        emit("\n" + "=" * sep_len)
        emit(f"{(dataset.name + ': STRUCTURAL PERTURBATION MONOTONICITY'):^{sep_len}}")
        emit(f"Rounds: {rounds} | Steps per trajectory: k={steps} | MC Sample Pairs: {num_samples} | CPU Workers: {actual_jobs}")
        emit("=" * sep_len)

        raw_results = []
        if actual_jobs > 1 and rounds > 1:
            with ProcessPoolExecutor(max_workers=actual_jobs) as executor:
                futures = {
                    executor.submit(simulate_single_trajectory,
                        round_idx=r,
                        dag=dag,
                        elements=elements,
                        W=W,
                        evaluator=metric_evaluator,
                        k=steps,
                        epsilon=epsilon,
                        num_samples=num_samples,
                        seed=(seed + r * 10007) if seed is not None else None,
                    ): r
                    for r in range(rounds)
                }
                with tqdm(total=rounds, desc=f"Perturbation rounds ({dataset.name})", unit="round") as pbar:
                    for f in as_completed(futures):
                        raw_results.append(f.result())
                        pbar.update(1)
        else:
            with tqdm(total=rounds, desc=f"Perturbation rounds ({dataset.name})", unit="round") as pbar:
                for r in range(rounds):
                    res = simulate_single_trajectory(
                        round_idx=r,
                        dag=dag,
                        elements=elements,
                        W=W,
                        evaluator=metric_evaluator,
                        k=steps,
                        epsilon=epsilon,
                        num_samples=num_samples,
                        seed=(seed + r * 10007) if seed is not None else None,
                    )
                    raw_results.append(res)
                    pbar.update(1)

        raw_results.sort(key=lambda x: x["round_idx"])

        corrs_by_metric = {m.short_name: [] for m in metrics}
        for res in raw_results:
            for m in metrics:
                name = m.short_name
                rho = res["correlations"][name]
                if res["is_flat"][name] or np.isnan(rho):
                    continue
                corrs_by_metric[name].append(rho)

        sep_len = 93
        emit("\n" + "=" * sep_len)
        emit(f"{(dataset.name + ': PERTURBATION MONOTONICITY RESULTS'):^{sep_len}}")
        emit(f"(Trajectories: N={rounds} | Steps: k={steps} | Distance Metric: Topological Cluster Distance)")
        emit("=" * sep_len)
        emit(f"{'Metric':<26} | {'Spearman rho (Mean +/- SD)':<24} | {'Median [IQR]':<24} | {'% rho > 0':<10}")
        emit("-" * sep_len)

        stats = {}
        for m in metrics:
            name = m.short_name
            vals = np.array(corrs_by_metric[name], dtype=float)
            n_valid = len(vals)

            if n_valid > 0:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if n_valid > 1 else 0.0
                med_val = float(np.median(vals))
                iqr_25 = float(np.percentile(vals, 25))
                iqr_75 = float(np.percentile(vals, 75))
                pct_pos = float(100.0 * np.mean(vals > 0))
            else:
                mean_val, std_val, med_val, iqr_25, iqr_75, pct_pos = [np.nan] * 6

            label = getattr(m, "paper_name", m.name)
            mean_str = f"{mean_val:+.3f} +/- {std_val:.3f}" if not np.isnan(mean_val) else "N/A"
            med_str = f"{med_val:+.3f} [{iqr_25:+.2f}, {iqr_75:+.2f}]" if not np.isnan(med_val) else "N/A"
            pct_str = f"{pct_pos:5.1f}%" if not np.isnan(pct_pos) else "N/A"
            emit(f"{label:<26} | {mean_str:<24} | {med_str:<24} | {pct_str:<10}")

            stats[name] = {"mean": mean_val, "std": std_val, "median": med_val, "iqr_25": iqr_75, "iqr_75": iqr_75, "pct_pos": pct_pos}
        emit("=" * sep_len + "\n")

        with open(txt_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(report_lines) + "\n")
        print(f"Appended {dataset.name} to:\n  - {txt_path}")

        series.append({"name": dataset.name, "stats": stats})

    return {item["name"]: item["stats"] for item in series}


def main():
    args = get_args()

    datasets = load_datasets(resources_dir=args.resources, model=args.model, max_genes=args.max_genes)
    if args.experiment == "biological-contrast":
        evaluate_biological_contrast(datasets, save_path=args.save, n_boot=args.bootstrap_reps, seed=args.seed)
    elif args.experiment == "perturbation-monotonicity":
        evaluate_perturbation_monotonicity(datasets, save_path=args.save, rounds=args.rounds, steps=args.steps, n_jobs=args.jobs, num_samples=args.samples, epsilon=args.epsilon, seed=args.seed)


if __name__ == '__main__':
    main()
