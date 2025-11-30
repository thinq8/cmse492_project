"""Train a TensorFlow MLP with CV + hold-out evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.data.loader import load_raw_dataframe, prepare_features, split_dataset  # noqa: E402

METRIC_KEYS = ("accuracy", "f1", "roc_auc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mlp.json"),
        help="JSON file with data/training settings.",
    )
    return parser.parse_args()


def read_config(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - config guard
        raise ValueError(f"Invalid JSON config at {path}: {exc}")


def build_model_builder(training_cfg: Dict[str, Any], input_dim: int):
    hidden_units = training_cfg.get("hidden_units", [64, 32])
    dropout_rate = float(training_cfg.get("dropout_rate", 0.3))
    l2_value = float(training_cfg.get("l2", 5e-4))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))

    def _builder() -> tf.keras.Model:
        model = tf.keras.Sequential(name="mlp_classifier")
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        for units in hidden_units:
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_value),
                )
            )
            model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="roc_auc"),
            ],
        )
        return model

    return _builder


def compute_metric_bundle(y_true, y_pred_labels, y_pred_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred_labels)),
        "f1": float(f1_score(y_true, y_pred_labels)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
    }


def summarize_cv(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for key, values in metrics.items()
    }


def run_cross_validation(
    X_train,
    y_train,
    cv_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    input_dim: int,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    cv = StratifiedKFold(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        shuffle=bool(cv_cfg.get("shuffle", True)),
        random_state=int(cv_cfg.get("random_state", 42)),
    )
    metric_accumulator: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
    histories: List[Dict[str, float]] = []

    builder = build_model_builder(training_cfg, input_dim)
    patience = int(training_cfg.get("patience", 12))
    batch_size = int(training_cfg.get("batch_size", 64))
    epochs = int(training_cfg.get("epochs", 120))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train[train_idx])
        X_val_fold = scaler.transform(X_train[val_idx])
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        model = builder()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_roc_auc",
                patience=patience,
                mode="max",
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        histories.append(
            {
                "fold": fold_idx,
                "epochs_ran": len(history.history["loss"]),
                "best_val_roc_auc": float(max(history.history.get("val_roc_auc", [0.0]))),
            }
        )
        y_val_proba = model.predict(X_val_fold, verbose=0).flatten()
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        metrics = compute_metric_bundle(y_val_fold, y_val_pred, y_val_proba)
        for key, value in metrics.items():
            metric_accumulator[key].append(value)

    return summarize_cv(metric_accumulator), histories


def train_final_model(
    X_train,
    y_train,
    X_test,
    y_test,
    training_cfg: Dict[str, Any],
    input_dim: int,
) -> Tuple[tf.keras.Model, StandardScaler, Dict[str, float], Dict[str, Any]]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    builder = build_model_builder(training_cfg, input_dim)
    model = builder()
    patience = int(training_cfg.get("patience", 12))
    batch_size = int(training_cfg.get("batch_size", 64))
    epochs = int(training_cfg.get("epochs", 120))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            patience=patience,
            mode="max",
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    y_test_proba = model.predict(X_test_scaled, verbose=0).flatten()
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    metrics = compute_metric_bundle(y_test, y_test_pred, y_test_proba)

    history_summary = {
        "epochs_ran": len(history.history["loss"]),
        "best_val_roc_auc": float(max(history.history.get("val_roc_auc", [0.0]))),
    }
    return model, scaler, metrics, history_summary


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    data_path = Path(config.get("data_path", "data/raw/high_diamond_ranked_10min.csv"))
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

    X_train = splits.X_train.reset_index(drop=True).to_numpy(dtype=np.float32)
    y_train = splits.y_train.reset_index(drop=True).to_numpy(dtype=np.float32)
    X_test = splits.X_test.to_numpy(dtype=np.float32)
    y_test = splits.y_test.to_numpy(dtype=np.float32)

    cv_cfg = config.get("cv", {})
    cv_cfg.setdefault("random_state", int(config.get("random_state", 42)))
    training_cfg = config.get("training", {})
    input_dim = X_train.shape[1]

    cv_start = time.perf_counter()
    cv_scores, cv_histories = run_cross_validation(
        X_train,
        y_train,
        cv_cfg,
        training_cfg,
        input_dim,
    )
    cv_duration = time.perf_counter() - cv_start

    final_start = time.perf_counter()
    model, scaler, test_metrics, history_summary = train_final_model(
        X_train,
        y_train,
        X_test,
        y_test,
        training_cfg,
        input_dim,
    )
    final_duration = time.perf_counter() - final_start

    artifact_dir = Path(config.get("artifact_dir", "models/neural_net"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "mlp_classifier.keras"
    scaler_path = artifact_dir / "mlp_scaler.joblib"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return {
        "model": "mlp",
        "cv": cv_scores,
        "cv_histories": cv_histories,
        "cv_duration_sec": cv_duration,
        "test": test_metrics,
        "final_training_summary": history_summary,
        "final_training_duration_sec": final_duration,
        "artifact_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_count": int(input_dim),
        "config": config,
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    results = run_training(config)
    metrics_path = Path(config.get("metrics_output", "data/processed/mlp_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
