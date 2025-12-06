# Final Project Compliance Audit Plan

Goal: verify that the repository, documentation, notebooks, and pending LaTeX report satisfy every requirement described in `CMSE492_FinalProject_Requirements - Copy.pdf`.

## Checklist to Execute
1. **Dataset Requirements**
   - Confirm dataset source, size, and documentation meet guidelines.
   - Ensure README/Docs describe origin beyond "Kaggle" and mention sample/feature counts.
2. **Repository Deliverables**
   - Verify `README.md` includes setup instructions, problem description, links to notebooks, and repo hygiene.
   - Confirm `requirements.txt` covers every dependency referenced in notebooks/scripts.
   - Check `src/` scripts exist for preprocessing/evaluation and are documented.
3. **Exploratory Data Analysis**
   - Inspect notebooks (`notebooks/exploratory/`) for: missingness analysis, class balance, descriptive stats, correlations, outliers.
   - Ensure EDA occurs after train/test split per requirement; document this in README/report.
4. **Preprocessing Documentation**
   - Review `src/preprocessing/run_eda.py` + feature engineering notebooks for: splitting rationale, scaling/encoding steps, feature engineering definitions.
   - Confirm engineered features saved in `data/processed/` are described somewhere (README or docs).
5. **Model Coverage**
   - Validate at least three models of increasing complexity exist (e.g., logistic, RF, XGBoost/MLP) with clear notebooks describing hyperparameters and training methodology.
   - For neural models, capture architecture details and regularization settings.
6. **Training Methodology Evidence**
   - Locate plots or logged metrics illustrating learning curves / EarlyStopping for neural models.
   - Ensure notes exist on hyperparameter tuning (e.g., `hyperparameter_sweeper_tensorflow.ipynb`).
7. **Metrics & Evaluation**
   - Confirm metrics (accuracy, F1, ROC-AUC, etc.) recorded per model—ideally in JSON under `data/processed/`.
   - Check for inference/training time info; record gaps for follow-up.
8. **Model Interpretation**
   - Verify notebooks provide feature importance / SHAP / coefficient analysis.
9. **Reporting Assets**
   - Gather necessary figures/tables from `figures/` and metrics JSONs for the LaTeX report.
   - Identify any missing content required for report sections (Background, Data, Methods, Results, Conclusion).
10. **LaTeX Report Assembly**
    - Use `docs/CMSE_492_Project_Template.tex` as base; fill sections with repository findings in the user’s voice/style.

## Execution Notes
- Track gaps/missing deliverables while auditing; log actionable fixes in this doc before editing code/docs.
- After repository audit is complete, draft the final LaTeX report reflecting the confirmed content and highlighting any outstanding TODOs.

## Audit Findings

### Dataset Selection & Documentation
- ✅ Dataset meets size (9,879 matches) and feature-count requirements; documented via `data/processed/eda_summary.json` and `docs/Overview.md`.
- ✅ README now spells out the Riot API provenance and sample/feature counts (see `README.md:28-32`).

### Data Description & Preprocessing
- ✅ `src/preprocessing/run_eda.py` and `src/data/loader.py` perform stratified splitting before EDA, log transforms, engineered ratios, and metadata exports.
- ✅ `docs/Overview.md` + EDA notebook discuss missingness, class balance, correlation, and outliers.
- ✅ README + `docs/Overview.md` now enumerate engineered features and motivations for reuse.

### Modeling Variety & Coverage
- ✅ Logistic regression, random forest (TF-DF + sklearn), MLP, and XGBoost notebooks exist under `notebooks/modeling/`.
- ✅ Baseline scripts (`src/models/train_baselines.py`, `src/models/train_mlp.py`) provide reproducible training with CV.
- ✅ XGBoost notebook now includes a SHAP-style attribution cell writing to `data/processed/xgboost_shap_importance.csv` when executed.

### Training Methodology & Metrics
- ✅ Cross-validation stats stored in `data/processed/baseline_cv_metrics.json` and `mlp_metrics.json`; `report_metrics.py` aggregates into figures/tables.
- ✅ Timing details captured in `docs/model_timing.md` and echoed in both `docs/Overview.md` and the LaTeX report.
- ✅ Final report now describes the actual loss functions, regularization, and hyperparameter sweeps instead of future plans.

### Model Interpretation
- ✅ Logistic notebook discusses coefficient inspection; random forest notebook references TF-DF inspector importance.
- ✅ Boosted-tree notebook contains the SHAP export cell, and the LaTeX report cites those importances alongside coefficients.

### Documentation & Reporting
- ✅ README provides environment setup and repo layout with refreshed provenance + engineered-feature notes.
- ✅ `docs/Overview.md` now contains the full metric table and training/inference timing snapshot tied to `figures/model_performance/` artifacts.
- ✅ `docs/CMSE_492_Project_Template.tex` has been rewritten as the final report with empirical findings and references to stored artifacts.
