# CMSE 492 Project Template

This repository scaffolds a CMSE 492 final project focused on predicting League of Legends match outcomes from ten-minute game statistics. It provides a clean directory layout for data, notebooks, source code, figures, and documentation, so coursework deliverables can be added without restructuring the repo later.

## Directory Structure

```
.
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploratory/
├── src/
│   ├── preprocessing/
│   ├── models/
│   └── evaluation/
├── figures/
├── docs/
└── requirements.txt
```

- `data/raw/` holds the original dataset (ignored by git once added locally).
- `data/processed/` stores cleaned artifacts ready for modeling.
- `notebooks/exploratory/` is for EDA and experimentation notebooks.
- `src/` houses production code for preprocessing, model training, and evaluation.
- `figures/` collects plots for reports and presentations.
- `docs/` contains course templates, requirements, and planning documents.

## Getting Started

1. Create and activate a Python environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install dependencies once `requirements.txt` is populated: `pip install -r requirements.txt`.
3. Add the raw Kaggle dataset to `data/raw/` and begin exploratory work in `notebooks/exploratory/`.

Update `requirements.txt` as packages are added, and use git commits to track progress through the course milestones.

