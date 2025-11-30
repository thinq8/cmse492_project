"""Aggregate metric JSON files into tables and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("data/processed/baseline_cv_metrics.json"),
        help="Path to baseline metrics JSON (train_baselines output).",
    )
    parser.add_argument(
        "--mlp-metrics",
        type=Path,
        default=Path("data/processed/mlp_metrics.json"),
        help="Path to neural network metrics JSON (train_mlp output).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/model_performance"),
        help="Destination for summary CSV/plots.",
    )
    return parser.parse_args()


def load_baseline_records(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    records: List[Dict[str, float]] = []
    for model_name, metrics in payload.get("metrics", {}).items():
        record = {
            "model": model_name,
            "source": "baseline_script",
        }
        for metric_name, stats in metrics.get("cv", {}).items():
            record[f"cv_{metric_name}_mean"] = stats.get("mean")
            record[f"cv_{metric_name}_std"] = stats.get("std")
        for metric_name, value in metrics.get("test", {}).items():
            record[f"test_{metric_name}"] = value
        records.append(record)
    return records


def load_mlp_records(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    record = {
        "model": payload.get("model", "mlp"),
        "source": "mlp_script",
    }
    for metric_name, stats in payload.get("cv", {}).items():
        record[f"cv_{metric_name}_mean"] = stats.get("mean")
        record[f"cv_{metric_name}_std"] = stats.get("std")
    for metric_name, value in payload.get("test", {}).items():
        record[f"test_{metric_name}"] = value
    return [record]


def create_metric_plot(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["accuracy", "f1", "roc_auc"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 9))
    for ax, metric in zip(axes, metrics):
        column = f"test_{metric}"
        subset = df[["model", column]].dropna()
        ax.barh(subset["model"], subset[column], color="#1b9e77")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"Test {metric.upper()} by Model")
        for idx, value in enumerate(subset[column]):
            ax.text(value + 0.01, idx, f"{value:.3f}")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "test_metric_comparison.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_baseline_records(args.baseline_metrics)
    records.extend(load_mlp_records(args.mlp_metrics))
    if not records:
        raise FileNotFoundError(
            "No metric files found. Run training scripts before report generation."
        )
    df = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "model_metrics_summary.csv"
    df.to_csv(summary_csv, index=False)
    create_metric_plot(df, args.output_dir)
    summary_json = args.output_dir / "model_metrics_summary.json"
    summary_json.write_text(df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
