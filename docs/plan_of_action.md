## Implementation Snapshot
- Built `src/data/loader.py` to centralize stratified splitting, engineered ratio features, log transforms, scaler persistence, and metadata logging exactly as promised in the proposal.
- Parameterized EDA/preprocessing via `src/preprocessing/run_eda.py`, which now emits reproducible splits, descriptive statistics, and baseline sanity checks aligned with the CMSE 492 data-description requirements.
- Added `configs/baselines.yaml` plus `src/models/train_baselines.py` so logistic regression and random forest pipelines can be reproduced from the CLI with consistent CV/test evaluation and saved artifacts.

## Requirements Traceability (CMSE 492 Checklist)
| Requirement Block | Implementation Plan |
| --- | --- |
| Background & Motivation | Covered in `docs/Overview.md` and will be reiterated in the LaTeX report; notebooks reference the esports use-case to keep context explicit. |
| Data Description & Preprocessing | `src/data/loader.py`, `src/preprocessing/run_eda.py`, and the forthcoming preprocessing notebook document splitting rationale, scaling, log transforms, engineered ratios, and balancing strategy. |
| Modeling Variety | Baselines (logistic regression + random forest) already scripted; neural network experiments delivered via a dedicated TensorFlow pipeline and companion notebook. |
| Training Methodology | Cross-validation, early stopping, and learning-curve logging are encoded in `train_baselines.py`, the new `train_mlp.py`, and mirrored in notebooks with rich commentary on hyperparameters/losses. |
| Metrics & Results | `src/evaluation/report_metrics.py` consolidates ROC-AUC, F1, accuracy, and timing summaries into tables/plots; notebooks echo the rationale for metric selection. |
| Model Interpretation | SHAP analysis notebook + helper routines describe feature-level insights and satisfy the interpretability clause. |
| Documentation | Repo retains README, requirements, metadata exports, and the notebooks/report deliver detailed justifications per section of the requirement PDF. |

## Current Sprint Objectives
1. **Notebook: Baseline + Tree Models** – Create a modeling notebook that loads processed data, runs cross-validated logistic regression and random forest models, and justifies each hyperparameter (penalty, C, depth, estimators, etc.) in prose tied to dataset characteristics.
2. **Notebook: Neural Network Experiments** – Deliver a TensorFlow/Keras notebook that defines the compact MLP, explains dropout/L2 settings, and records how early stopping, batch size, and learning rate were chosen.
3. **Notebook: Evaluation & Interpretability** – Aggregate metrics from CLI runs, render comparison plots, and compute SHAP values for the best model while documenting why these diagnostics satisfy the course rubric.
4. **Script: `src/models/train_mlp.py`** – Provide a CLI twin of the notebook workflow to ensure headless training, CV, and artifact saving for the neural network.
5. **Script: `src/evaluation/report_metrics.py`** – Automate creation of the metric summary tables/figures that the requirement checklist demands for the report/presentation.

## Deliverable Blueprint
- `notebooks/modeling/baseline_models.ipynb`: emphasizes why penalized logistic regression is a strong interpretable baseline vs. the higher-variance random forest, with markdown justifying each hyperparameter relative to gold/experience scale and class balance.
- `notebooks/modeling/neural_network_experiments.ipynb`: details input normalization, architecture design, optimizer/loss selection, regularization, and tracking of ROC-AUC during CV.
- `notebooks/modeling/model_evaluation_and_interpretability.ipynb`: stitches together CLI-generated JSON metrics, plots ROC/F1 comparisons, and runs SHAP to explain the feature contributions of the saved random forest model.

## Validation Strategy
- Each training script writes JSON logs under `data/processed/`; the evaluation script ingests those plus timing metadata to guarantee reproducibility in the final LaTeX report.
- Before each submission checkpoint, rerun: `python src/preprocessing/run_eda.py`, `python src/data/loader.py`, `python src/models/train_baselines.py --config configs/baselines.yaml`, `python src/models/train_mlp.py --config configs/mlp.json`, and finally `python src/evaluation/report_metrics.py` to refresh metrics/figures referenced by the notebooks and report.
