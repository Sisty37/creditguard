# 🛡️ CreditGuard — Credit Default Risk Prediction

Real-time credit default risk assessment using **XGBoost + Optuna hyperparameter tuning**, with **SHAP-based explainability** and an interactive **Streamlit** web app.

> Developed for **COMPAS 2026 — IEEE 3rd International Conference**, University of Dhaka.

---

## 📌 Overview

CreditGuard predicts the probability that a borrower will default on a loan within **2 years**, using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 records). The final model is an **XGBoost classifier** tuned with **Optuna**, achieving an **AUC of 0.8734**. Predictions are explained at the individual borrower level using **SHAP** (SHapley Additive exPlanations), so every risk score comes with a transparent, feature-level justification.

## ✨ Features

- 🔍 **Real-time risk scoring** — instant default probability from borrower inputs
- 📊 **SHAP explanations** — see exactly which features push risk up or down
- 🎛️ **Interactive UI** — adjustable sliders and inputs for all borrower attributes
- 📤 **JSON API-style output** — structured prediction response for downstream integration
- 🧪 **Full experiment trail** — baseline, tuned, and Optuna-optimized model notebooks with statistical validation (cross-validation, Wilcoxon test, ablation study)

## 🖼️ App Preview

Run the app locally (see [Usage](#-usage)) to see the live UI — borrower information on the left, prediction result, SHAP explanation, and JSON response on the right.

## 🧠 Model

| | |
|---|---|
| **Algorithm** | XGBoost (Optuna Tuned) |
| **Dataset** | Give Me Some Credit — 150,000 records |
| **Best AUC** | 0.8734 |
| **Explainability** | SHAP (TreeExplainer) |
| **Target** | Probability of default within 2 years |

Full experiment results (baseline vs. tuned vs. Optuna, ablation study, cross-validation, Wilcoxon significance test) are in [`outputs/`](./outputs) and the accompanying notebooks.

## 📂 Project Structure

```
credit_scoring_project/
├── app/
│   └── streamlit_app.py           # Streamlit web app
├── models/
│   ├── best_credit_model.pkl      # Trained XGBoost model
│   ├── feature_names.json         # Feature order used by the model
│   └── scaler.pkl                 # Fitted feature scaler
├── notebooks/
│   ├── 01_without_tuning.ipynb    # Baseline model
│   ├── 02_with_tuning.ipynb       # Manually tuned model
│   ├── 03_optuna_tuning.ipynb     # Optuna hyperparameter search
│   └── 04_shap.ipynb              # SHAP explainability analysis
├── outputs/
│   ├── baseline_results.csv
│   ├── tuned_results.csv
│   ├── optuna_results.csv
│   ├── ablation_study.csv
│   ├── cv_results.csv
│   └── wilcoxon_test.csv
├── figures/
│   ├── ablation_study.png
│   ├── calibration_comparison.png
│   ├── performance_baseline.png
│   ├── performance_optuna.png
│   ├── optuna_best_model_roc.png
│   ├── baseline_confusion_matrix.png
│   ├── shap_summary.png
│   └── shap_importance_chart.png
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** raw/processed datasets are not included in this repo (see [`.gitignore`](./.gitignore)) due to file size. See [Data](#-data) below to obtain them.

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/credit_scoring_project.git
cd credit_scoring_project

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 🚀 Usage

Run the Streamlit app locally:

```bash
streamlit run app/streamlit_app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) and enter borrower details to get an instant risk prediction with SHAP explanation.

## 📊 Data

This project uses the **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)** dataset from Kaggle (150,000 records). Raw and processed data files are excluded from this repository to keep it lightweight — download `cs-training.csv` / `cs-test.csv` from Kaggle and place them under `data/raw/` to reproduce the notebooks.

## 🔬 Reproducing the Experiments

The `notebooks/` directory walks through the full experimental pipeline in order:

1. **`01_without_tuning.ipynb`** — baseline XGBoost model
2. **`02_with_tuning.ipynb`** — manual/grid hyperparameter tuning
3. **`03_optuna_tuning.ipynb`** — Optuna-based Bayesian hyperparameter optimization (final model)
4. **`04_shap.ipynb`** — SHAP-based feature importance and explainability analysis

Results and figures produced by these notebooks are saved to `outputs/` and `figures/` respectively.

## 📜 Citation

If you use this work, please cite:

```
A. J. Sisty, "CreditGuard: Explainable Credit Default Risk Prediction with
XGBoost and Optuna," COMPAS 2026 — IEEE 3rd International Conference,
University of Dhaka, 2026.
```

## 👤 Author

**Abu Jafar Sisty**
COMPAS 2026 — IEEE 3rd International Conference · University of Dhaka

## 📄 License

This project is released for academic and research purposes. Please contact the author for reuse beyond citation.
