# Project Plan

## Objective
Build and document a machine learning pipeline that predicts League of Legends match outcomes from ten-minute statistics while meeting CMSE 492 final project requirements.

## Key Deliverables
- Reproducible data pipeline for acquisition, cleaning, and feature engineering.
- Training scripts for baseline, tree-based, and neural network models with comparison plots/tables.
- Model interpretability assets (SHAP or equivalent) and discussion-ready figures.
- LaTeX report populated using `CMSE_492_Project_Template.tex` and aligned with requirement checklist.
- GitHub-ready repository with README, environment specification, and usage instructions.

## Workstreams & Next Actions
1. **Data Acquisition & Profiling**
   - Download the Kaggle Diamond Ranked Games (10 min) dataset; document source metadata.
   - Perform exploratory analysis: distributions, pairwise correlations, class balance, missingness validation.
2. **Feature Engineering & Preprocessing**
   - Create derived metrics (e.g., gold/XP differences, objective control rates).
   - Define train/validation/test splits; implement scaling pipelines for models that need it.
3. **Model Development**
   - Implement logistic regression baseline with regularization.
   - Train and tune a random forest; log feature importances.
   - Build a fully connected neural network with early stopping and hyperparameter tuning.
4. **Evaluation & Interpretation**
   - Compare models on ROC-AUC, F1, accuracy; record training/inference times.
   - Generate SHAP-based explanations and supporting visualizations.
5. **Reporting & Documentation**
   - Populate LaTeX template sections; insert figures/tables per requirements.
   - Maintain repository hygiene: README, `requirements.txt` or `environment.yml`, reproducible scripts/notebooks.

## Milestones
- **Week 1:** Dataset acquired, exploratory notebook drafted, preprocessing decisions finalized.
- **Week 2:** Baseline and ensemble models tuned; interim metrics reported.
- **Week 3:** Neural network experiments completed; SHAP analysis generated.
- **Week 4:** Draft report in LaTeX reviewed; repository validated against submission checklist.

## Risks & Mitigations
- **Data limitations:** If features lack predictive power, expand engineered features or consider time-extended dataset.
- **Model overfitting:** Use cross-validation, early stopping, and regularization; monitor learning curves.
- **Time management:** Follow milestone calendar; use version control checkpoints to track progress.

