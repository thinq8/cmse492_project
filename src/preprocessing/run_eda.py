"""Generate exploratory data analysis artifacts for the League of Legends dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MPL_CACHE = Path(".cache/matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib
matplotlib.use("Agg")  # Render plots without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/high_diamond_ranked_10min.csv"),
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/eda"),
        help="Directory for saving generated plots.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for storing derived summaries.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the data split.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset missing at {args.data_path}")

    df = pd.read_csv(args.data_path)
    target = "blueWins"

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df[target],
        random_state=args.random_state,
    )

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    sample_path = args.processed_dir / "sample_matches.csv"
    train_df.sample(n=min(200, len(train_df)), random_state=args.random_state).to_csv(
        sample_path, index=False
    )

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns

    summary = {
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "n_features": int(train_df.shape[1] - 1),
        "class_balance_train": train_df[target]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "missing_rates": train_df.isnull().mean().round(4).to_dict(),
    }

    train_df[numeric_cols].describe().T.round(3).to_csv(
        args.processed_dir / "numeric_feature_summary.csv"
    )

    corr_with_target = (
        train_df[numeric_cols]
        .corr()[target]
        .drop(target)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    summary["top_correlated_features"] = (
        corr_with_target.head(10).round(4).to_dict()
    )

    summary_path = args.processed_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    split_meta = {
        "random_state": args.random_state,
        "test_size": args.test_size,
        "stratified": True,
        "train_count": int(train_df.shape[0]),
        "test_count": int(test_df.shape[0]),
    }
    (args.processed_dir / "split_metadata.json").write_text(
        json.dumps(split_meta, indent=2)
    )

    missing_rates = (
        pd.Series(summary["missing_rates"]).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    missing_rates.head(15).plot(kind="bar", color="#7570b3", ax=ax)
    ax.set_ylabel("Fraction Missing")
    ax.set_title("Top Feature Missingness (Train Split)")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "missingness.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts = train_df[target].value_counts().sort_index()
    ax.bar(
        ["Red Wins (0)", "Blue Wins (1)"],
        class_counts.values,
        color=["#d95f02", "#1b9e77"],
    )
    ax.set_ylabel("Match Count")
    ax.set_title("Train Split Class Balance")
    for patch, count in zip(ax.patches, class_counts.values):
        ax.annotate(
            f"{count}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(args.figures_dir / "class_balance.png", dpi=300)
    plt.close(fig)

    key_features = [
        "blueGoldDiff",
        "blueExperienceDiff",
        "blueKills",
        "blueDeaths",
        "blueEliteMonsters",
        "blueDragons",
        "blueTowersDestroyed",
    ]
    fig, axes = plt.subplots(len(key_features), 1, figsize=(7, 2.6 * len(key_features)))
    for feature, ax in zip(key_features, axes):
        sns.histplot(
            train_df,
            x=feature,
            hue=target,
            element="step",
            stat="density",
            common_norm=False,
            palette=["#d95f02", "#1b9e77"],
            ax=ax,
        )
        ax.set_title(f"Distribution of {feature} by Match Outcome")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "feature_distributions.png", dpi=300)
    plt.close(fig)

    objective_features = [
        "blueDragons",
        "blueHeralds",
        "blueEliteMonsters",
        "blueTowersDestroyed",
    ]
    obj_stats = (
        train_df.groupby(target)[objective_features]
        .mean()
        .rename(index={0: "Red Victory", 1: "Blue Victory"})
        .T
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    obj_stats.plot(kind="bar", ax=ax, color=["#d95f02", "#1b9e77"])
    ax.set_ylabel("Average Count (First 10 Minutes)")
    ax.set_title("Objective Control by Winning Team")
    ax.legend(title="Outcome")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "objective_control.png", dpi=300)
    plt.close(fig)

    selected = list(corr_with_target.head(12).index)
    selected.append(target)

    corr_matrix = train_df[selected].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Top Outcome-Linked Features")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "top_feature_correlation_heatmap.png", dpi=300)
    plt.close(fig)

    outlier_features = ["blueGoldDiff", "blueExperienceDiff", "blueKills"]
    outlier_summary = {}
    for feat in outlier_features:
        series = train_df[feat]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)
        outliers = series[(series < lower) | (series > upper)]
        outlier_summary[feat] = {
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_fraction": float(len(outliers) / len(series)),
        }

    (args.processed_dir / "outlier_summary.json").write_text(
        json.dumps(outlier_summary, indent=2)
    )

    feature_cols = [
        col for col in train_df.columns if col not in {target, "gameId"}
    ]
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    baseline_results: dict[str, dict[str, float]] = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    baseline_results["dummy_majority"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_dummy)),
        "f1": float(f1_score(y_test, y_pred_dummy, zero_division=0)),
    }

    log_reg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    y_proba_lr = log_reg.predict_proba(X_test)[:, 1]
    baseline_results["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_lr)),
        "f1": float(f1_score(y_test, y_pred_lr)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_lr)),
    }

    (args.processed_dir / "baseline_metrics.json").write_text(
        json.dumps(baseline_results, indent=2)
    )


if __name__ == "__main__":
    run(parse_args())
