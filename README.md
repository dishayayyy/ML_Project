# Disease Prediction Using Machine Learning

A machine learning project that predicts diseases based on patient-reported symptoms using three classification models — Decision Tree, Random Forest, and Bernoulli Naive Bayes.

---

## Authors

| Name | Roll Number |
|------|-------------|
| Disha Sharma | 245UAI038 |
| Kinjal Nigam | 245UAI063 |

---

## Problem Statement

Given a set of symptoms reported by a patient, predict the most likely disease from a set of 41 possible conditions. This is a **multi-class classification problem** with 132 binary input features (symptom present = 1, absent = 0).

---

## Dataset

- **Source:** [Kaggle — Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- **Files:** `Training.csv`, `Testing.csv`
- **Features:** 132 binary symptom columns
- **Target:** `prognosis` — 41 disease classes
- **Training samples:** 4920 (120 per disease — perfectly balanced)
- **Test samples:** 42 (1 per disease)

---

## Exploratory Data Analysis

The following EDA plots were generated:

1. **Disease Class Distribution** — Confirmed perfectly balanced dataset (120 samples per disease)
2. **Top 20 Most Frequent Symptoms** — `fatigue` and `vomiting` are the most common across all diseases
3. **Total Active Symptoms per Disease** — Common Cold and Tuberculosis have the most symptoms; Fungal infection has the fewest
4. **Disease-Symptom Heatmap** — Visualizes which of the top 20 symptoms are associated with each disease, revealing distinct diagnostic patterns

---

## Preprocessing

- Dropped the extra unnamed column (`Unnamed: 133`) from `Training.csv`
- Separated features (`X`) and target (`y`)
- Encoded target labels using `LabelEncoder`
- No scaling required — all features are binary (0/1)
- Used the pre-provided train/test split from Kaggle

---

## Models Used

| Model | Description |
|-------|-------------|
| Decision Tree | Simple, interpretable baseline model |
| Random Forest | Ensemble of 100 decision trees (`n_estimators=100`) |
| Bernoulli Naive Bayes | Probabilistic model, theoretically suited for binary features |

---

## Results

| Model | Accuracy |
|-------|----------|
| Decision Tree | 97.62% |
| Random Forest | 97.62% |
| Bernoulli NB | **100.00%** |

- **Bernoulli NB** achieved perfect accuracy — well-suited for binary feature data
- **Decision Tree & Random Forest** both misclassified 1 out of 42 test samples (Fungal infection vs overlapping skin-symptom diseases)
- All three models show near-perfect precision and recall across all 41 disease classes

---

## Project Structure

```
End_sem project/
│
├── Training.csv          # Training dataset (4920 samples)
├── Testing.csv           # Testing dataset (42 samples)
├── disease_prediction.ipynb  # Main Jupyter Notebook
└── README.md             # Project documentation
```

---

## Libraries Used

```
pandas
numpy
matplotlib
seaborn
scikit-learn
warnings
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

1. Clone or download this repository
2. Place `Training.csv` and `Testing.csv` in the same folder as the notebook
3. Open `disease_prediction.ipynb` in Jupyter Notebook
4. Run all cells sequentially from top to bottom

---

## Key Observations

- The dataset is **perfectly balanced** — no class imbalance issues
- **Fatigue and vomiting** are the most generic symptoms, appearing across many diseases
- **Hepatitis variants** (A, B, C, D, E) share strong symptom overlap (`yellowish_skin`, `yellowing_of_eyes`)
- **Bernoulli NB** is theoretically the best match for this binary feature structure
- **Random Forest** is recommended for real-world use due to robustness against noise and unseen symptom combinations

---

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
- Scikit-learn Documentation: https://scikit-learn.org
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
