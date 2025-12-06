# Predicting League of Legends Outcomes (CMSE 492 Project)

This repo tracks the CMSE 492 capstone where we forecast 10-minute match outcomes for League of Legends. It offers a clean analysis stack—split data artifacts, reproducible preprocessing scripts, and a growing "model zoo" of notebooks—so teammates can focus on experimentation instead of wiring.

## Quick Start Checklist
1. **Create/activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. **Install the requirements** (scientific stack, TensorFlow, TF-DF, XGBoost, notebooks):
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Kaggle dataset** (not versioned)
   ```bash
   kaggle datasets download -d bobbyscience/league-of-legends-diamond-ranked-games-10-min
   unzip league-of-legends-diamond-ranked-games-10-min.zip -d data/raw
   ls data/raw/high_diamond_ranked_10min.csv  # sanity check
   ```
4. **Regenerate the processed artifacts + baseline split**
   ```bash
   python src/preprocessing/run_eda.py
   ```
   This script runs the initial EDA, creates `train/test` splits, and writes summary JSON/CSVs to `data/processed/`.
5. **Open `notebooks/` in Jupyter or VS Code** and run any of the exploratory/modeling workflows below.

## Data & Artifacts
- `data/raw/high_diamond_ranked_10min.csv` — Riot Games telemetry for 9,879 Diamond-tier solo queue matches sampled exactly at the ten-minute mark and republished on Kaggle by Bobby Chen. Each record covers 39 numeric features (objective control, vision, gold/XP, combat) plus the binary `blueWins` label. Cite the Kaggle page in reports but describe the true origin as Riot's match API per the course rubric.
- `data/processed/` stores light, shareable files: train/test splits, engineered ratios, metrics JSON, etc.
- Feature engineering helpers live inside notebooks for now; when they stabilize, move them into `src/preprocessing/`.
- Engineered feature exports (`engineered_features.csv`, `engineered_train.csv`, `engineered_test.csv`) include domain ratios such as `blue_objective_share`, `blue_vision_share`, `blue_kill_share`, ward efficiency, XP/gold momentum, and log-transformed totals. These match the write-up requirements for describing preprocessing decisions.

## Project Layout
```
.
├── configs/                 # future experiment configs
├── data/
│   ├── raw/                 # downloaded Kaggle CSV (gitignored)
│   └── processed/           # reproducible artifacts + train/test splits
├── docs/                    # planning docs, assignment prompts
├── figures/                 # plots exported from notebooks/EDA
├── notebooks/
│   ├── exploratory/         # initial EDA + baseline classifier
│   └── modeling/            # model zoo (see below)
├── src/
│   ├── data/                # data loading utilities
│   ├── preprocessing/       # scripts to build processed/ artifacts
│   ├── models/              # reusable training/evaluation helpers
│   └── evaluation/
├── models/                  # serialized checkpoints if desired
├── requirements.txt
└── README.md
```

## Modeling Notebook Guide
| Notebook | Technique | Notes |
| --- | --- | --- |
| `exploratory/eda_baseline.ipynb` | Data audit + logistic baseline | Mirrors Part B deliverable, exports summary CSV/plots. |
| `modeling/logistic_regression_tensorflow.ipynb` | TF logistic regression | Interpretable coefficients + engineered feature variant. |
| `modeling/mlp_tensorflow.ipynb` | TF multi-layer perceptron | Nonlinear neural baseline with EarlyStopping. |
| `modeling/random_forest_tensorflow.ipynb` | TF Decision Forest (bagging) | Tree ensemble without gradient boosting. |
| `modeling/xgboost_gradient_boosted_trees.ipynb` | XGBoost GBDT | Implements the requested boosted-tree method + engineered feature experiment. |
| `modeling/hyperparameter_sweeper_tensorflow.ipynb` | Keras Tuner sweeps | Template for scanning TF architectures. |

Tips:
- Reuse the CSV splits under `data/processed/` to keep metrics apples-to-apples.
- `engineered_*` CSVs bundle domain ratios (vision share, gold momentum, etc.) for quick experimentation.
- When a notebook graduates to "production", move reusable code into `src/models/` or `src/preprocessing/`.

## Reproducing Experiments
- **End-to-end EDA + split generation**: `python src/preprocessing/run_eda.py`
- **Train/evaluate notebooks**: launch Jupyter (`jupyter lab`, `code .`, or VS Code + Python extension) and run the desired notebook.
- **Batch metrics**: store outputs to `data/processed/*metrics.json` so comparisons remain version-controlled.

## Maintenance & Repo Hygiene
- Large raw data stays local; keep `data/raw/` gitignored.
- Before committing, ensure notebooks are stripped of superfluous outputs (Clear All / Restart & Run All).
- Prefer referencing shared helpers from `src/` rather than duplicating logic once it stabilizes.
- Run `pip list --outdated` occasionally so TensorFlow/XGBoost security patches do not lag.

## Exploratory Highlights (from the baseline run)
- Stratified 80/20 split → 7,903 training matches / 1,976 test matches with balanced `blueWins` labels (~50%).
- No missingness across the 39 numeric Kaggle features; outliers stay under 1.3% for economy/XP metrics.
- Economic/XP differentials (`blueGoldDiff`, `blueExperienceDiff`, etc.) dominate correlation and tree importances.
- Logistic regression already surpasses the dummy baseline (accuracy ~0.72, ROC-AUC ~0.81), setting a clear performance floor.
- Figures saved under `figures/eda/` (class balance, correlation heatmap, etc.) are drop-in assets for reports.
