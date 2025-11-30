"""Train baseline models defined in configs/baselines.yaml."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.data.loader import (  # noqa: E402
    load_raw_dataframe,
    prepare_features,
    split_dataset,
)

METRIC_FUNCS = {
    "accuracy": lambda y_true, y_pred, y_proba: accuracy_score(y_true, y_pred),
    "f1": lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred),
    "roc_auc": lambda y_true, y_pred, y_proba: roc_auc_score(y_true, y_proba),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines.yaml"),
        help="Path to the model configuration file (JSON/YAML).",
    )
    return parser.parse_args()


def read_config(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Config file at {path} is not valid JSON/YAML: {exc}")


def build_estimator(name: str, params: Dict[str, Any]) -> Any:
    if name == "logistic_regression":
        model = LogisticRegression(**params)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    if name == "random_forest":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unsupported model '{name}'")


def compute_metrics(
    y_true,
    y_pred,
    y_proba,
    metric_keys,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for key in metric_keys:
        if key == "roc_auc" and y_proba is None:
            raise ValueError("roc_auc requires probability estimates")
        metric_fn = METRIC_FUNCS[key]
        results[key] = float(metric_fn(y_true, y_pred, y_proba))
    return results


def cross_validate_model(
    estimator,
    X,
    y,
    cv,
    metric_keys,
) -> Dict[str, Dict[str, float]]:
    per_metric: Dict[str, list[float]] = {key: [] for key in metric_keys}
    for train_idx, val_idx in cv.split(X, y):
        cloned = clone(estimator)
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        cloned.fit(X_train_fold, y_train_fold)
        y_pred = cloned.predict(X_val_fold)
        y_proba = (
            cloned.predict_proba(X_val_fold)[:, 1]
            if hasattr(cloned, "predict_proba")
            else None
        )
        fold_metrics = compute_metrics(y_val_fold, y_pred, y_proba, metric_keys)
        for key, value in fold_metrics.items():
            per_metric[key].append(value)
    return {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for key, values in per_metric.items()
    }


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    data_path = Path(config.get("data_path", "data/raw/high_diamond_ranked_10min.csv"))
    processed_dir = Path(config.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_df = load_raw_dataframe(data_path)
    features_df = prepare_features(
        raw_df,
        log_columns=config.get("log_columns", []),
        add_ratios=config.get("ratio_features", True),
        drop_columns=config.get("drop_columns", []),
    )
    splits = split_dataset(
        features_df,
        test_size=float(config.get("test_size", 0.2)),
        random_state=int(config.get("random_state", 42)),
    )
    X_train = splits.X_train.reset_index(drop=True)
    y_train = splits.y_train.reset_index(drop=True)
    X_test = splits.X_test
    y_test = splits.y_test

    cv_conf = config.get("cv", {})
    cv = StratifiedKFold(
        n_splits=int(cv_conf.get("n_splits", 5)),
        shuffle=bool(cv_conf.get("shuffle", True)),
        random_state=int(config.get("random_state", 42)),
    )

    metric_keys = config.get("scoring_metrics", ["accuracy", "f1", "roc_auc"])

    artifact_dir = Path(config.get("artifact_dir", "models/baselines"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for name, params in config.get("models", {}).items():
        estimator = build_estimator(name, params)
        cv_scores = cross_validate_model(estimator, X_train, y_train, cv, metric_keys)
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        y_proba = (
            fitted.predict_proba(X_test)[:, 1]
            if hasattr(fitted, "predict_proba")
            else None
        )
        test_metrics = compute_metrics(y_test, y_pred, y_proba, metric_keys)
        model_path = artifact_dir / f"{name}.joblib"
        joblib.dump(fitted, model_path)
        results[name] = {
            "cv": cv_scores,
            "test": test_metrics,
            "artifact_path": str(model_path),
        }
    return {
        "metrics": results,
        "config": config,
        "feature_count": len(splits.feature_names),
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    results = run_training(config)
    metrics_output = Path(config.get("metrics_output", "data/processed/baseline_cv_metrics.json"))
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
