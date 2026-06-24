import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="CreditGuard",
    page_icon="🛡️",
    layout="wide"
)

# ─── Load Model & Assets ───────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    model_path        = os.path.join(BASE_DIR, 'models', 'best_credit_model.pkl')
    feature_names_path = os.path.join(BASE_DIR, 'models', 'feature_names.json')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    return model, feature_names

model, feature_names = load_model()

# ─── Title ─────────────────────────────────────────────────

st.title("🛡️ CreditGuard — Credit Risk Assessment System")
st.markdown("**Real-time credit default prediction using XGBoost + Optuna**")
st.markdown("*COMPAS 2026 — IEEE Conference | University of Dhaka*")
st.divider()

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.title("📋 About")
st.sidebar.info(
    "This system predicts the probability of a borrower "
    "defaulting on a loan within 2 years using Machine Learning.\n\n"
    "**Model:** XGBoost (Optuna Tuned)\n\n"
    "**Dataset:** Give Me Some Credit (150,000 records)\n\n"
    "**Best AUC:** 0.8734"
)
st.sidebar.divider()
st.sidebar.markdown("**Author:** Abu Jafar Sisty")
st.sidebar.markdown("**Conference:** COMPAS 2026, IEEE")

# ─── Input Form ────────────────────────────────────────────
st.subheader("📝 Enter Borrower Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Information**")
    age = st.slider("Age", min_value=18, max_value=100, value=40)
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

with col2:
    st.markdown("**Financial Information**")
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=5000)
    debt_ratio = st.slider("Debt Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    revolving_utilization = st.slider("Revolving Utilization of Unsecured Lines",
                                       min_value=0.0, max_value=1.0, value=0.3, step=0.01)

with col3:
    st.markdown("**Credit History**")
    num_open_credit = st.number_input("Number of Open Credit Lines & Loans",
                                       min_value=0, max_value=50, value=5)
    num_real_estate = st.number_input("Number of Real Estate Loans or Lines",
                                       min_value=0, max_value=20, value=1)
    num_30_59 = st.number_input("Times 30-59 Days Past Due (Not Worse)",
                                 min_value=0, max_value=10, value=0)
    num_60_89 = st.number_input("Times 60-89 Days Past Due (Not Worse)",
                                 min_value=0, max_value=10, value=0)
    num_90_late = st.number_input("Times 90+ Days Late",
                                   min_value=0, max_value=10, value=0)

st.divider()

# ─── Feature Engineering ───────────────────────────────────
def engineer_features(age, monthly_income, debt_ratio, revolving_utilization,
                       num_open_credit, num_real_estate, num_30_59, num_60_89,
                       num_90_late, number_of_dependents):

    monthly_debt         = debt_ratio * monthly_income
    net_monthly_income   = monthly_income - monthly_debt
    income_per_dependent = monthly_income / (number_of_dependents + 1)
    debt_per_dependent   = monthly_debt   / (number_of_dependents + 1)
    total_past_due       = num_30_59 + num_60_89 + num_90_late
    weighted_late_pay    = num_30_59 * 1 + num_60_89 * 2 + num_90_late * 3
    high_utilization     = 1 if revolving_utilization > 0.7 else 0
    maxed_out            = 1 if revolving_utilization > 0.95 else 0
    is_young             = 1 if age < 30 else 0
    is_senior            = 1 if age > 60 else 0
    has_late_pay         = 1 if total_past_due > 0 else 0
    has_serious_late     = 1 if num_90_late > 0 else 0
    has_open_loans       = 1 if num_open_credit > 0 else 0
    has_real_estate      = 1 if num_real_estate > 0 else 0
    is_negative_cash     = 1 if net_monthly_income < 0 else 0
    utilization_x_late   = revolving_utilization * total_past_due
    debt_ratio_x_age     = debt_ratio * age

    age_bins   = [0, 25, 35, 45, 55, 65, 100]
    age_labels = [0, 1, 2, 3, 4, 5]
    age_bucket = 0
    for i in range(len(age_bins)-1):
        if age_bins[i] < age <= age_bins[i+1]:
            age_bucket = age_labels[i]
            break

    features = {
        'RevolvingUtilizationOfUnsecuredLines': revolving_utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': num_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': num_open_credit,
        'NumberOfTimes90DaysLate': num_90_late,
        'NumberRealEstateLoansOrLines': num_real_estate,
        'NumberOfTime60-89DaysPastDueNotWorse': num_60_89,
        'NumberOfDependents': number_of_dependents,
        'TotalTimesPastDue': total_past_due,
        'WeightedLatePay': weighted_late_pay,
        'MonthlyDebt': monthly_debt,
        'NetMonthlyIncome': net_monthly_income,
        'IncomePerDependent': income_per_dependent,
        'DebtPerDependent': debt_per_dependent,
        'HighUtilization': high_utilization,
        'MaxedOut': maxed_out,
        'IsYoung': is_young,
        'IsSenior': is_senior,
        'AgeBucket': age_bucket,
        'HasLatePay': has_late_pay,
        'HasSeriousLatePay': has_serious_late,
        'HasOpenLoans': has_open_loans,
        'HasRealEstate': has_real_estate,
        'IsNegativeCash': is_negative_cash,
        'UtilizationXLatePay': utilization_x_late,
        'DebtRatioXAge': debt_ratio_x_age
    }
    return features

# ─── Predict Button ────────────────────────────────────────
predict_btn = st.button("🔍 Predict Credit Risk", type="primary", use_container_width=True)

if predict_btn:
    features = engineer_features(
        age, monthly_income, debt_ratio, revolving_utilization,
        num_open_credit, num_real_estate, num_30_59, num_60_89,
        num_90_late, number_of_dependents
    )

    input_df = pd.DataFrame([features])[feature_names]
    input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    prob_default    = model.predict_proba(input_df)[0][1]
    prob_no_default = 1 - prob_default
    prediction      = model.predict(input_df)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK — Likely to Default")
        else:
            st.success("✅ LOW RISK — Unlikely to Default")
    with col_r2:
        st.metric("Default Probability", f"{prob_default*100:.2f}%")
    with col_r3:
        st.metric("No Default Probability", f"{prob_no_default*100:.2f}%")

    st.divider()
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Risk Level:**")
        if prob_default < 0.3:
            st.progress(prob_default, text=f"Low Risk ({prob_default*100:.1f}%)")
        elif prob_default < 0.6:
            st.progress(prob_default, text=f"Medium Risk ({prob_default*100:.1f}%)")
        else:
            st.progress(prob_default, text=f"High Risk ({prob_default*100:.1f}%)")
    with col_g2:
        st.markdown("**Risk Category:**")
        if prob_default < 0.2:
            st.info("Very Low Risk — Excellent creditworthiness")
        elif prob_default < 0.4:
            st.info("Low-Medium Risk — Good creditworthiness")
        elif prob_default < 0.6:
            st.warning("Medium Risk — Fair creditworthiness")
        else:
            st.error("High Risk — Poor creditworthiness")

    # SHAP
    st.divider()
    st.subheader("🔍 SHAP Explanation — Why this prediction?")
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_vals[0],
            'Feature Value': input_df.iloc[0].values
        }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Features Increasing Risk:**")
            for _, row in shap_df[shap_df['SHAP Value'] > 0].head(5).iterrows():
                st.markdown(f"🔴 **{row['Feature']}** = {row['Feature Value']:.3f} "
                           f"(+{row['SHAP Value']:.4f})")
        with col_s2:
            st.markdown("**Features Decreasing Risk:**")
            for _, row in shap_df[shap_df['SHAP Value'] < 0].head(5).iterrows():
                st.markdown(f"🟢 **{row['Feature']}** = {row['Feature Value']:.3f} "
                           f"({row['SHAP Value']:.4f})")

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#F44336' if v > 0 else '#2196F3' for v in shap_df['SHAP Value']]
        ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors, edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_title('SHAP Feature Contributions', fontsize=12, fontweight='bold')
        ax.set_xlabel('SHAP Value (impact on prediction)')
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    # JSON
    st.divider()
    st.subheader("📤 API Response (JSON)")
    api_response = {
        "prediction": int(prediction),
        "risk_label": "Default" if prediction == 1 else "No Default",
        "probability_default": round(float(prob_default), 4),
        "probability_no_default": round(float(prob_no_default), 4),
        "risk_category": (
            "Very Low" if prob_default < 0.2 else
            "Low" if prob_default < 0.4 else
            "Medium" if prob_default < 0.6 else "High"
        ),
        "model": "XGBoost (Optuna Tuned)",
        "auc": 0.8734
    }
    st.json(api_response)

# ─── Footer ────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Credit Risk Assessment System | Abu Jafar Sisty | "
    "COMPAS 2026 — IEEE 3rd International Conference | University of Dhaka"
    "</p>",
    unsafe_allow_html=True
)
