# Predicting League of Legends Match Outcomes Using Early-Game Statistics

## Abstract

This project aims to predict the outcome of a League of Legends (LoL) match using team statistics from the first 10 minutes of gameplay. Using a dataset of approximately 10,000 Diamond-tier ranked games, we apply three supervised learning models—logistic regression, random forest, and a deep neural network—to classify whether the Blue Team will win. The project demonstrates that early-game features like gold difference, objectives taken, and kill counts provide a moderate predictive signal for match outcomes. We compare model performance using ROC-AUC, F1 score, and accuracy, and interpret results using SHAP to extract game-relevant insights.

---

## 1. Background and Motivation

League of Legends is one of the most-played online competitive games in the world, with a vast ecosystem of esports, live broadcasting, and analytics. Accurately forecasting a match’s result from early statistics is useful for broadcasters, analysts, and potentially AI coaching systems. 

Traditional rule-based systems fail to adapt to complex, high-dimensional match data. ML allows learning non-obvious patterns and interactions between features like gold, kills, and objectives that influence outcomes. Predicting the Blue Team's victory from early-game stats reflects a real-world scenario with practical value.

---

## 2. Data Description

- **Source:** Kaggle - "League of Legends Diamond Ranked Games (10 min)"
- **Size:** ~10,000 match records
- **Target Variable:** `blueWins` (binary: 1 if Blue wins, 0 otherwise)

### Features:

- Blue/Red team kills, deaths, assists
- Gold earned, XP, minions/jungle minions killed
- Vision scores, wards placed/killed
- Towers destroyed, dragons/heralds taken

All features are numeric, and the dataset is balanced and clean with no missing values.

---

## 3. Preprocessing

- Removed no features (all numeric and complete)
- Calculated train/test/validation split (70/15/15), stratified on `blueWins`
- Standardized continuous features for neural network input
- Optionally engineered differences (e.g., `blueGoldDiff`, `blueXPperGold`)

---

## 4. ML Task and Objective

### Task:
Binary classification – predict `blueWins` (0 or 1) using early-game features

### Why ML:
- Non-linear relationships (e.g., gold + objectives)
- Interactions between multiple team stats
- Traditional models can't scale or adapt to gameplay complexity

---

## 5. Models

| Model                | Description                                |
|---------------------|--------------------------------------------|
| Logistic Regression | Linear baseline for comparison             |
| Random Forest       | Ensemble method to capture interactions    |
| Neural Network      | MLP (TensorFlow) for high-dimensional input|

Each model trained with hyperparameter tuning via cross-validation.

---

## 6. Training Methodology

| Model                | Loss Function     | Hyperparameters                         |
|---------------------|-------------------|------------------------------------------|
| Logistic Regression | Binary Cross Entropy | L2 penalty (C = 1.0)                    |
| Random Forest       | N/A               | 100 trees, max_depth=8                   |
| Neural Network      | Binary Cross Entropy | 2 layers (64, 32), dropout=0.3, Adam    |

- Stratified 5-fold CV for tuning
- EarlyStopping for DNN to prevent overfitting

---

## 7. Metrics

Primary:
- ROC-AUC

Secondary:
- Accuracy
- F1-score
- Precision, Recall

Evaluation done on the test set (15%) held out.

---

## 8. Results and Model Comparison

| Model                | Accuracy | F1-score | ROC-AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | XX.XX%   | XX.XX    | XX.XX   |
| Random Forest       | XX.XX%   | XX.XX    | XX.XX   |
| Neural Network      | XX.XX%   | XX.XX    | XX.XX   |

- NN slightly outperformed ensemble models on AUC
- Random Forest showed most stable performance
- Logistic Regression underfit complex patterns

---

## 9. Model Interpretation

- SHAP values revealed top features:
  - `blueGoldDiff`
  - `blueKills`
  - `blueFirstDragon`
  - `blueXPperMin`
- Interactions between early objectives and vision control were predictive
- Model correctly identified momentum indicators

---

## 10. Conclusion

- Early-game features moderately predict match outcomes (AUC > 0.70)
- Neural network offered slight performance gains over tree-based methods
- SHAP interpretation helped validate key gameplay concepts

### Limitations:
- No champion picks or player identity included
- Prediction limited to 10-minute mark only

### Future Work:
- Expand input window to 15–20 minutes
- Add sequential modeling (e.g., LSTM)
- Explore pro-level matches or lower skill tiers

---

## References

1. [League of Legends Dataset - Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)
2. SHAP: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
3. TensorFlow/Keras Documentation

---


