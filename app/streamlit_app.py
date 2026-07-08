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
    model_path         = os.path.join(BASE_DIR, 'models', 'best_credit_model.pkl')
    feature_names_path = os.path.join(BASE_DIR, 'models', 'feature_names.json')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    return model, feature_names

model, feature_names = load_model()

# ─── Theme tokens (mirrors the CreditGuard light-mode design) ──
GOLD        = "#A9780F"
GOLD_LIGHT  = "#C9932A"
RISK_HIGH   = "#C7404B"
RISK_MID    = "#B87A1E"
RISK_LOW    = "#2E9C6F"
RISK_HIGH_BG = "rgba(199,64,75,0.08)"
RISK_MID_BG  = "rgba(184,122,30,0.08)"
RISK_LOW_BG  = "rgba(46,156,111,0.08)"
TEXT        = "#1B2436"
TEXT_DIM    = "#5B6478"
TEXT_FAINT  = "#8B93A6"
BORDER      = "#E1E6EF"
SURFACE_2   = "#F1F4F9"

# ─── Global CSS (light mode, card styling, typography) ─────
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background:
            radial-gradient(1200px 600px at 50% -10%, rgba(169,120,15,0.06), transparent 60%),
            #F5F7FB;
    }}
    #MainMenu, footer {{visibility: hidden;}}

    /* Card-style bordered containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #FFFFFF;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 2px rgba(27,36,54,0.04), 0 8px 24px rgba(27,36,54,0.05);
        padding: 6px 6px;
    }}

    h1, h2, h3 {{
        font-family: 'Fraunces', serif !important;
        color: {TEXT} !important;
    }}
    p, span, label, div {{
        color: {TEXT};
    }}
    .cg-card-title {{
        font-family: 'Fraunces', serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 4px 0 2px 0;
        color: {TEXT};
    }}
    .cg-card-sub {{
        color: {TEXT_FAINT};
        font-size: 0.85rem;
        margin: 0 0 18px 0;
    }}
    .cg-section-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {GOLD};
        font-weight: 600;
        margin: 4px 0 10px 0;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 8px;
    }}

    div[data-testid="stNumberInput"] {{ max-width: 220px; }}
    div[data-testid="stNumberInput"] input {{ text-align: left; }}
    div[data-testid="stButton"] {{ display: flex; justify-content: center; margin-top: 12px; }}
    div[data-testid="stButton"] button {{
        max-width: 320px;
        width: 100%;
        background: linear-gradient(180deg, {GOLD_LIGHT}, {GOLD});
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 0;
        box-shadow: 0 8px 20px rgba(169,120,15,0.22);
    }}
    div[data-testid="stButton"] button:hover {{
        color: #FFFFFF;
        box-shadow: 0 10px 24px rgba(169,120,15,0.3);
    }}

    /* Header */
    .cg-header {{ text-align: center; margin-bottom: 8px; }}
    .cg-title {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.4rem;
        letter-spacing: -0.02em;
        margin: 6px 0 6px 0;
        color: {TEXT};
    }}
    .cg-title span {{ color: {GOLD}; }}
    .cg-subtitle {{ color: {TEXT_DIM}; font-size: 1rem; margin: 0 0 16px 0; }}
    .cg-meta-row {{ display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-bottom: 8px;}}
    .cg-pill {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 6px 14px;
        border: 1px solid {BORDER};
        border-radius: 999px;
        color: {TEXT_DIM};
        background: #FFFFFF;
    }}
    .cg-pill b {{ color: {GOLD}; font-weight: 600; }}

    /* Placeholder card */
    .cg-placeholder {{
        border: 1.5px dashed {BORDER};
        border-radius: 14px;
        padding: 56px 28px;
        text-align: center;
        color: {TEXT_FAINT};
        background: #FFFFFF;
    }}
    .cg-placeholder .ph-icon {{ font-size: 1.8rem; margin-bottom: 12px; }}
    .cg-placeholder .ph-title {{
        font-family: 'Fraunces', serif;
        font-size: 1.05rem;
        color: {TEXT_DIM};
        margin: 0 0 6px 0;
        font-weight: 600;
    }}
    .cg-placeholder .ph-sub {{ font-size: 0.85rem; line-height: 1.6; margin: 0; }}

    /* Verdict badge */
    .cg-badge {{
        display:inline-flex; align-items:center; gap:8px;
        padding:8px 16px; border-radius:999px; font-weight:600; font-size:0.9rem;
        margin-bottom: 14px;
    }}

    /* Metric row */
    .cg-metrics {{ display:flex; gap:26px; flex-wrap:wrap; margin-top: 4px;}}
    .cg-metric .m-label {{
        font-size:0.72rem; color:{TEXT_FAINT}; text-transform:uppercase; letter-spacing:0.05em;
    }}
    .cg-metric .m-value {{
        font-family:'JetBrains Mono', monospace; font-size:1.3rem; font-weight:600; color:{TEXT};
    }}

    /* SHAP list items */
    .cg-shap-title {{ font-size:0.8rem; font-weight:600; margin-bottom:10px; }}
    .cg-shap-item {{
        display:flex; justify-content:space-between; align-items:center;
        padding:9px 12px; border-radius:8px; margin-bottom:6px; font-size:0.82rem;
        background:{SURFACE_2};
    }}
    .cg-shap-item .sf-val {{ font-family:'JetBrains Mono', monospace; font-weight:600; font-size:0.78rem; }}

    .cg-footer {{
        text-align:center; margin-top: 36px; color:{TEXT_FAINT}; font-size:0.78rem; line-height:1.7;
    }}
    .cg-footer .author {{ color:{TEXT_DIM}; font-weight:500; }}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────
st.markdown(f"""
<div class="cg-header">
    <div style="font-size:2.2rem;">🛡️</div>
    <div class="cg-title">Credit<span>Guard</span></div>
    <div class="cg-subtitle">Real-time credit default risk, explained — powered by XGBoost + Optuna</div>
    <div class="cg-meta-row">
        <span class="cg-pill">Model: <b>XGBoost (Optuna Tuned)</b></span>
        <span class="cg-pill">Dataset: <b>Give Me Some Credit · 150K records</b></span>
        <span class="cg-pill">AUC: <b>0.8734</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️  About this system"):
    st.markdown(
        "This system predicts the probability of a borrower defaulting on a loan "
        "within two years using machine learning. Enter borrower details on the "
        "left to get an instant risk assessment with a feature-level explanation.\n\n"
        f"**Author:** Abu Jafar Sisty  \n"
        
    )

st.write("")

# ─── Two-column layout: form (left) · result (right) ───────
col_form, col_result = st.columns([1, 1], gap="large")

# ═══════════════════ LEFT: Borrower Information ═════════════
with col_form:
    with st.container(border=True):
        st.markdown('<p class="cg-card-title">📝 Borrower Information</p>', unsafe_allow_html=True)
        st.markdown('<p class="cg-card-sub">All fields feed directly into the risk model</p>', unsafe_allow_html=True)

        sub1, sub2 = st.columns(2)

        with sub1:
            st.markdown('<div class="cg-section-label">Personal</div>', unsafe_allow_html=True)
            age = st.slider("Age", min_value=18, max_value=100, value=40)
            number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

            st.markdown('<div class="cg-section-label">Financial</div>', unsafe_allow_html=True)
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=5000)
            debt_ratio = st.slider("Debt Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            revolving_utilization = st.slider("Revolving Utilization", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

        with sub2:
            st.markdown('<div class="cg-section-label">Credit History</div>', unsafe_allow_html=True)
            num_open_credit = st.number_input("Open Credit Lines & Loans", min_value=0, max_value=50, value=5)
            num_real_estate = st.number_input("Real Estate Loans / Lines", min_value=0, max_value=20, value=1)
            num_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, max_value=10, value=0)
            num_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, max_value=10, value=0)
            num_90_late = st.number_input("Times 90+ Days Late", min_value=0, max_value=10, value=0)

        predict_btn = st.button("🔍 Predict Credit Risk", type="primary", use_container_width=False)

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

# ─── Run prediction on click, persist in session_state so the
#     right-hand panel stays populated even if other widgets rerun ──
if predict_btn:
    features = engineer_features(
        age, monthly_income, debt_ratio, revolving_utilization,
        num_open_credit, num_real_estate, num_30_59, num_60_89,
        num_90_late, number_of_dependents
    )

    input_df = pd.DataFrame([features])[feature_names]
    input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    prob_default    = float(model.predict_proba(input_df)[0][1])
    prob_no_default = 1 - prob_default
    prediction      = int(model.predict(input_df)[0])

    st.session_state['cg_result'] = {
        'input_df': input_df,
        'prob_default': prob_default,
        'prob_no_default': prob_no_default,
        'prediction': prediction,
    }

# ═══════════════════ RIGHT: Prediction Result ════════════════
with col_result:
    if 'cg_result' not in st.session_state:
        st.markdown("""
        <div class="cg-placeholder">
            <div class="ph-icon">📊</div>
            <p class="ph-title">Your prediction will appear here</p>
            <p class="ph-sub">Fill in the borrower information on the left and click<br>
            "Predict Credit Risk" to see the result.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state['cg_result']
        input_df       = r['input_df']
        prob_default   = r['prob_default']
        prob_no_default = r['prob_no_default']
        prediction     = r['prediction']
        pct = prob_default * 100

        cat = ("Very Low" if prob_default < 0.2 else
               "Low" if prob_default < 0.4 else
               "Medium" if prob_default < 0.6 else "High")

        gauge_color = RISK_LOW if pct < 30 else RISK_MID if pct < 60 else RISK_HIGH
        if prediction == 1:
            badge_bg, badge_color, badge_text = RISK_HIGH_BG, RISK_HIGH, "⚠️ HIGH RISK — Likely to Default"
        else:
            badge_bg, badge_color, badge_text = RISK_LOW_BG, RISK_LOW, "✅ LOW RISK — Unlikely to Default"

        # ── Prediction Result card (radial gauge + badge + metrics) ──
        with st.container(border=True):
            st.markdown('<p class="cg-card-title">📊 Prediction Result</p>', unsafe_allow_html=True)

            r_circ = 515
            offset = r_circ - (pct / 100) * r_circ
            gcol1, gcol2 = st.columns([1, 1.3])
            with gcol1:
                st.markdown(f"""
                <div style="position:relative;width:150px;height:150px;">
                  <svg width="150" height="150" viewBox="0 0 190 190">
                    <circle cx="95" cy="95" r="82" fill="none" stroke="#EAEDF3" stroke-width="14"/>
                    <circle cx="95" cy="95" r="82" fill="none" stroke="{gauge_color}"
                            stroke-width="14" stroke-linecap="round"
                            stroke-dasharray="{r_circ}" stroke-dashoffset="{offset}"
                            transform="rotate(-90 95 95)"/>
                  </svg>
                  <div style="position:absolute;inset:0;display:flex;flex-direction:column;
                              align-items:center;justify-content:center;">
                    <div style="font-family:'Fraunces',serif;font-size:1.7rem;font-weight:600;color:{gauge_color};">
                      {pct:.1f}%
                    </div>
                    <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:{TEXT_FAINT};">
                      Default Prob.
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with gcol2:
                st.markdown(f"""
                <div class="cg-badge" style="background:{badge_bg};color:{badge_color};">{badge_text}</div>
                <div class="cg-metrics">
                    <div class="cg-metric">
                        <div class="m-label">Default Probability</div>
                        <div class="m-value">{prob_default*100:.2f}%</div>
                    </div>
                    <div class="cg-metric">
                        <div class="m-label">No-Default Probability</div>
                        <div class="m-value">{prob_no_default*100:.2f}%</div>
                    </div>
                    <div class="cg-metric">
                        <div class="m-label">Risk Category</div>
                        <div class="m-value">{cat}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── SHAP Explanation card ──
        with st.container(border=True):
            st.markdown('<p class="cg-card-title">🔍 SHAP Explanation</p>', unsafe_allow_html=True)
            st.markdown('<p class="cg-card-sub">Top features driving this prediction</p>', unsafe_allow_html=True)
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

                s1, s2 = st.columns(2)
                with s1:
                    st.markdown(f'<p class="cg-shap-title" style="color:{RISK_HIGH};">🔴 Increasing Risk</p>', unsafe_allow_html=True)
                    up_rows = shap_df[shap_df['SHAP Value'] > 0].head(5)
                    if len(up_rows) == 0:
                        st.markdown('<div class="cg-shap-item"><span>—</span></div>', unsafe_allow_html=True)
                    for _, row in up_rows.iterrows():
                        st.markdown(
                            f'<div class="cg-shap-item"><span>{row["Feature"]} = {row["Feature Value"]:.3f}</span>'
                            f'<span class="sf-val" style="color:{RISK_HIGH};">+{row["SHAP Value"]:.4f}</span></div>',
                            unsafe_allow_html=True
                        )
                with s2:
                    st.markdown(f'<p class="cg-shap-title" style="color:{RISK_LOW};">🟢 Decreasing Risk</p>', unsafe_allow_html=True)
                    down_rows = shap_df[shap_df['SHAP Value'] < 0].head(5)
                    if len(down_rows) == 0:
                        st.markdown('<div class="cg-shap-item"><span>—</span></div>', unsafe_allow_html=True)
                    for _, row in down_rows.iterrows():
                        st.markdown(
                            f'<div class="cg-shap-item"><span>{row["Feature"]} = {row["Feature Value"]:.3f}</span>'
                            f'<span class="sf-val" style="color:{RISK_LOW};">{row["SHAP Value"]:.4f}</span></div>',
                            unsafe_allow_html=True
                        )

                fig, ax = plt.subplots(figsize=(7, 4.2))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                colors = [RISK_HIGH if v > 0 else RISK_LOW for v in shap_df['SHAP Value']]
                ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors, edgecolor='none')
                ax.axvline(x=0, color=TEXT_DIM, linewidth=0.8)
                ax.set_title('SHAP Feature Contributions', fontsize=11, fontweight='bold', color=TEXT)
                ax.set_xlabel('SHAP Value (impact on prediction)', color=TEXT_DIM)
                ax.tick_params(colors=TEXT_DIM, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(BORDER)
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        # ── API Response (JSON) card ──
        with st.container(border=True):
            st.markdown('<p class="cg-card-title">📤 API Response</p>', unsafe_allow_html=True)
            api_response = {
                "prediction": prediction,
                "risk_label": "Default" if prediction == 1 else "No Default",
                "probability_default": round(prob_default, 4),
                "probability_no_default": round(prob_no_default, 4),
                "risk_category": cat,
                "model": "XGBoost (Optuna Tuned)",
                "auc": 0.8734
            }
            st.code(json.dumps(api_response, indent=2), language="json")

# ─── Footer ────────────────────────────────────────────────
st.markdown(f"""
<div class="cg-footer">
    <p class="author">Abu Jafar Sisty</p>
</div>
""", unsafe_allow_html=True)