# Osteoporosis Risk Prediction

Machine learning project that predicts whether a patient is likely to have osteoporosis from demographic, lifestyle and medical-history features. It covers the full workflow: data cleaning, exploratory data analysis (EDA), model training with hyperparameter tuning, and model comparison.

![Python]
![scikit-learn]
![Jupyter

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis or clinical decisions.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings from EDA](#key-findings-from-eda)
- [Models and Results](#models-and-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Background

Osteoporosis weakens bones and makes them fragile and more likely to break. It develops slowly over several years and is often only diagnosed after a minor fall or impact causes a fracture. It is more common in older people. Identifying at-risk individuals early allows for preventive measures and targeted intervention.

## Dataset

`osteoporosis.csv` contains **1,958 patient records** and **16 columns**. The target classes are perfectly balanced (50% osteoporosis, 50% no osteoporosis).

| Column | Description | Values |
|---|---|---|
| `Id` | Unique patient identifier (dropped before modelling) | integer |
| `Age` | Age of the patient | 18 to 90 |
| `Gender` | Gender of the patient | Male / Female |
| `Hormonal Changes` | Hormonal status (postmenopausal also covers reduced testosterone in males) | Normal / Postmenopausal |
| `Family History` | Family history of osteoporosis | Yes / No |
| `Race/Ethnicity` | Race or ethnicity | African American / Asian / Caucasian |
| `Body Weight` | Body weight category | Normal / Underweight |
| `Calcium Intake` | Calcium intake level | Low / Adequate |
| `Vitamin D Intake` | Vitamin D intake level | Sufficient / Insufficient |
| `Physical Activity` | Activity level | Active / Sedentary |
| `Smoking` | Whether the patient smokes | Yes / No |
| `Alcohol Consumption` | Alcohol use | Moderate / missing |
| `Medical Conditions` | Relevant conditions | Hyperthyroidism / Rheumatoid Arthritis / missing |
| `Medications` | Relevant medications | Corticosteroids / missing |
| `Prior Fractures` | Previous fracture history | Yes / No |
| `Osteoporosis` | **Target**: 1 = has osteoporosis, 0 = does not | 0 / 1 |

**Missing values:** `Alcohol Consumption` (50.5%), `Medical Conditions` (33.0%) and `Medications` (50.3%) contain blanks. These were interpreted as "none" (no alcohol, no condition, no medication) and filled with the category `None`.

## Project Workflow

1. **Load and inspect** the data (shape, types, missing values, value counts, descriptive statistics).
2. **Preprocess (part 1):** fill missing values with `None`, drop the `Id` column.
3. **Exploratory data analysis:** plot every feature against the target.
4. **Preprocess (part 2):** label-encode all categorical columns and plot a correlation heatmap.
5. **Split:** 70% train / 30% test (`random_state=101`).
6. **Train and tune** four classifiers with `GridSearchCV` (5-fold cross-validation).
7. **Evaluate** with confusion matrices, accuracy, MAE/MSE/RMSE/R², and feature-importance plots.

## Key Findings from EDA

**Associated with higher osteoporosis risk**

- **Older age.** Patients aged roughly 20 to 40 had a noticeably lower risk.
- **Hormonal changes** (postmenopausal status).
- **Lower body weight** (underweight).
- **Low calcium and vitamin D intake.**
- **Sedentary lifestyle.**
- **Hyperthyroidism** and **corticosteroid use.**

**Little or no clear relationship observed**

- Gender, family history, race/ethnicity, smoking, alcohol consumption and prior fractures.

## Models and Results

Four models were trained and tuned:

| Model | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 81.6% | 0.83 | 0.80 | 0.81 |
| Random Forest | 87.4% | 0.97 | 0.77 | 0.86 |
| **Decision Tree** | **90.6%** | **0.98** | **0.83** | **0.90** |
| Support Vector Classifier (linear) | 82.7% | 0.85 | 0.78 | 0.82 |

*Evaluated on the 588-sample hold-out test set. Figures were reproduced by re-running the notebook's final model settings; exact values may differ slightly across library versions.*

The **Decision Tree** performed best on the hold-out set. In the feature-importance analysis, **Age** was by far the most influential predictor for the tree-based models, followed by smaller contributions from race/ethnicity, medical conditions, medications, vitamin D intake and hormonal changes.

## Repository Structure

```
Osteoporosis-Risk-Prediction/
├── Osteoporosis Risk Prediction.ipynb   # Full analysis and modelling notebook
├── Osteoporosis Risk Prediction.pdf     # PDF export of the notebook with outputs
├── osteoporosis.csv                     # Dataset
├── description.md                       # Project summary and data dictionary
├── requirements.txt                     # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or newer (developed with 3.12)
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/savvyoliviya/Osteoporosis-Risk-Prediction.git
cd Osteoporosis-Risk-Prediction

python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
jupyter notebook "Osteoporosis Risk Prediction.ipynb"
```

Run all cells from top to bottom. The notebook reads `osteoporosis.csv` from the same folder. The grid searches can take a few minutes.

## Limitations

- **Small dataset** (1,958 rows) with mostly binary features, so results may not generalise to real clinical populations.
- **Age dominates** the models. Many other features show weak signal, which suggests the data may be simplified or synthetic. Treat the accuracy figures with caution.
- **Single train/test split.** Reported scores come from one split and were not averaged across multiple runs or cross-validated on the test data.
- **No external validation** on an independent dataset.
- **Missing values** were assumed to mean "none", which may not hold for every record.
- Label encoding was applied to nominal features such as race/ethnicity, which implies an artificial ordering. One-hot encoding would be more appropriate for linear models.

## Future Work

- Add ROC-AUC, precision-recall curves and cross-validated scores to the evaluation.
- Replace label encoding with one-hot encoding for nominal features.
- Try gradient boosting models (XGBoost, LightGBM) and compare.
- Add model explainability with SHAP.
- Wrap the best model in a small web app (Streamlit or Flask) for interactive predictions.

## Author

**savvyoliviya** — [GitHub](https://github.com/savvyoliviya)
