"""Utility functions for loading and preprocessing the League of Legends dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_PATH = Path("data/raw/high_diamond_ranked_10min.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed")
TARGET = "blueWins"
LOG_COLUMNS = (
    "blueTotalGold",
    "blueTotalExperience",
    "redTotalGold",
    "redTotalExperience",
    "blueGoldPerMin",
    "redGoldPerMin",
)
DROP_COLUMNS = ("gameId", "redGoldDiff", "redExperienceDiff")


@dataclass
class DatasetSplits:
    """Container for train/test splits."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def load_raw_dataframe(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file with minimal validation."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset missing at {path}")
    return pd.read_csv(path)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable ratio features inspired by the project proposal."""

    result = df.copy()
    objective_numerator = (
        result["blueDragons"]
        + result["blueHeralds"]
        + result["blueEliteMonsters"]
    )
    objective_denominator = objective_numerator + (
        result["redDragons"] + result["redHeralds"] + result["redEliteMonsters"]
    )
    result["blue_objective_share"] = objective_numerator / np.maximum(
        1, objective_denominator
    )

    vision_denominator = result["blueWardsPlaced"] + result["redWardsPlaced"]
    result["blue_vision_share"] = result["blueWardsPlaced"] / np.maximum(
        1, vision_denominator
    )

    kill_denominator = result["blueKills"] + result["redKills"]
    result["blue_kill_share"] = result["blueKills"] / np.maximum(1, kill_denominator)

    return result


def apply_log_transforms(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Attach log1p-transformed variants of skewed columns."""

    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[f"log_{col}"] = np.log1p(result[col].clip(lower=0))
    return result


def prepare_features(
    df: pd.DataFrame,
    *,
    log_columns: Sequence[str] = LOG_COLUMNS,
    add_ratios: bool = True,
    drop_columns: Sequence[str] = DROP_COLUMNS,
) -> pd.DataFrame:
    """Derive modeling features according to preprocessing guidelines."""

    features = df.copy()
    if add_ratios:
        features = add_ratio_features(features)
    features = apply_log_transforms(features, log_columns)
    drop_targets = [col for col in drop_columns if col in features.columns]
    if drop_targets:
        features = features.drop(columns=drop_targets)
    return features


def split_dataset(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> DatasetSplits:
    """Split the dataset into train/test partitions."""

    feature_cols = [col for col in df.columns if col != TARGET]
    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return DatasetSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
    )


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_splits(splits: DatasetSplits, processed_dir: Path) -> dict[str, str]:
    """Persist split datasets to CSV files for reproducibility."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_features": processed_dir / "train_features.csv",
        "test_features": processed_dir / "test_features.csv",
        "train_labels": processed_dir / "train_labels.csv",
        "test_labels": processed_dir / "test_labels.csv",
    }
    splits.X_train.to_csv(paths["train_features"], index=False)
    splits.X_test.to_csv(paths["test_features"], index=False)
    splits.y_train.to_csv(paths["train_labels"], index=False, header=[TARGET])
    splits.y_test.to_csv(paths["test_labels"], index=False, header=[TARGET])
    return {key: str(path) for key, path in paths.items()}


def export_metadata(
    *,
    processed_dir: Path,
    splits: DatasetSplits,
    scaler_path: Path,
    dataset_path: Path,
    test_size: float,
    random_state: int,
    log_columns: Sequence[str],
    ratio_features: bool,
    split_paths: dict[str, str],
) -> None:
    """Write a metadata JSON blob that captures preprocessing decisions."""

    metadata = {
        "dataset_path": str(dataset_path),
        "test_size": test_size,
        "random_state": random_state,
        "n_features": len(splits.feature_names),
        "feature_names": splits.feature_names,
        "log_transforms": list(log_columns),
        "ratio_features": ratio_features,
        "split_paths": split_paths,
        "scaler_path": str(scaler_path),
    }
    (processed_dir / "data_loader_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory for processed artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout size for test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--disable-ratio-features",
        action="store_true",
        help="Skip engineered ratio features.",
    )
    parser.add_argument(
        "--log-columns",
        nargs="*",
        default=LOG_COLUMNS,
        help="Columns to log-transform (default is a vetted subset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = load_raw_dataframe(args.data_path)
    augmented = prepare_features(
        raw_df,
        log_columns=args.log_columns,
        add_ratios=not args.disable_ratio_features,
    )
    splits = split_dataset(
        augmented,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    scaler = fit_scaler(splits.X_train)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = args.processed_dir / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    split_paths = save_splits(splits, args.processed_dir)
    export_metadata(
        processed_dir=args.processed_dir,
        splits=splits,
        scaler_path=scaler_path,
        dataset_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        log_columns=args.log_columns,
        ratio_features=not args.disable_ratio_features,
        split_paths=split_paths,
    )


if __name__ == "__main__":
    main()
