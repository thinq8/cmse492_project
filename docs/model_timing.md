# Model Timing Measurements

Measured on an Apple M2 Pro (32 GB RAM) running Python 3.11 inside `.venv` while refreshing `data/processed/*metrics.json` on 2024-02-26. Each run used the canonical 7,903/1,976 stratified split.

| Model | Train Command | Train Time (s) | Inference Time on Test (ms) | Notes |
| --- | --- | --- | --- | --- |
| Logistic Regression | `python src/models/train_baselines.py --config configs/baselines.yaml` (logistic section) | 0.62 | 8 | StandardScaler + LBFGS converged in <20 iterations; inference measured via `.predict_proba` on 1,976 rows. |
| Random Forest | Same command (random_forest section) | 3.45 | 45 | 400 trees, unlimited depth; inference time measured with `.predict_proba` on the hold-out set. |
| MLP | `python src/models/train_mlp.py --config configs/mlp.json` | 2.10 | 30 | Training duration from `mlp_metrics.json` (`final_training_duration_sec`); inference measured via TensorFlow `.predict` on scaled test features. |

These numbers feed directly into the README/Overview and satisfy the CMSE 492 requirement for reporting training and inference durations.
