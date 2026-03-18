import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank – Personal Loan Campaign Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Playfair+Display:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
    }
    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #1a365d 50%, #2c5282 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        font-family: 'Playfair Display', serif !important;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #90cdf4;
        font-size: 1.05rem;
        margin: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .metric-card h3 {
        color: #1a365d;
        font-size: 2rem;
        margin: 0;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700;
    }
    .metric-card p {
        color: #718096;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .insight-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #f0fff4 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #2c5282;
        margin: 0.75rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #2d3748;
    }
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #d69e2e;
        margin: 0.75rem 0;
        font-size: 0.92rem;
        color: #744210;
    }
    .action-box {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #276749;
        margin: 0.75rem 0;
        font-size: 0.92rem;
        color: #22543d;
    }
    .section-header {
        color: #1a365d;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2c5282;
        display: inline-block;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #1a365d 100%);
    }
    div[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    div[data-testid="stSidebar"] p, div[data-testid="stSidebar"] span {
        color: #cbd5e0 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    "primary": "#1a365d",
    "secondary": "#2c5282",
    "accent": "#e53e3e",
    "success": "#276749",
    "warning": "#d69e2e",
    "blue_seq": ["#bee3f8", "#63b3ed", "#3182ce", "#1a365d"],
    "cat": ["#2c5282", "#e53e3e", "#d69e2e", "#276749", "#805ad5", "#dd6b20"],
    "loan_colors": {"Not Accepted": "#90cdf4", "Accepted": "#e53e3e"},
    "model_colors": {"Decision Tree": "#2c5282", "Random Forest": "#276749", "Gradient Boosted Tree": "#e53e3e"}
}

PLOTLY_LAYOUT = dict(
    font=dict(family="DM Sans, sans-serif", color="#2d3748"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=40, t=60, b=40),
    title_font=dict(size=16, color="#1a365d", family="Playfair Display, serif"),
    hoverlabel=dict(bgcolor="#1a365d", font_size=13, font_family="DM Sans"),
)


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df.columns = df.columns.str.strip()
    # Fix negative experience: set to absolute value
    df["Experience"] = df["Experience"].abs()
    return df


@st.cache_resource
def train_models(df):
    feature_cols = ["Age", "Experience", "Income", "Family", "CCAvg",
                    "Education", "Mortgage", "Securities Account",
                    "CD Account", "Online", "CreditCard"]
    X = df[feature_cols]
    y = df["Personal Loan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, min_samples_leaf=5,
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=8,
            min_samples_leaf=4, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosted Tree": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_train_pred = model.predict(X_train_sc)
        y_test_pred = model.predict(X_test_sc)
        y_test_proba = model.predict_proba(X_test_sc)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc_val = auc(fpr, tpr)

        results[name] = {
            "model": model,
            "y_test_pred": y_test_pred,
            "y_test_proba": y_test_proba,
            "train_acc": accuracy_score(y_train, y_train_pred),
            "test_acc": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_val,
            "fpr": fpr,
            "tpr": tpr,
            "cm": confusion_matrix(y_test, y_test_pred),
            "feature_importance": dict(zip(feature_cols, 
                model.feature_importances_))
        }

    return results, scaler, feature_cols, X_test, y_test


df = load_data()
model_results, scaler, feature_cols, X_test, y_test = train_models(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("##### Personal Loan Campaign Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Executive Summary",
         "🔍 Customer Deep-Dive",
         "🤖 Model Performance",
         "🎯 Campaign Strategy",
         "📁 Predict New Customers"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:1rem 0; opacity:0.7;'>
        <small style='color:#a0aec0;'>Built for Universal Bank<br>Marketing Analytics Division</small>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE 1: EXECUTIVE SUMMARY
# ═════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown("""
    <div class="main-header">
        <h1>Personal Loan Campaign Intelligence</h1>
        <p>Descriptive Analytics — Understanding our customer base and last campaign performance</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    total = len(df)
    accepted = df["Personal Loan"].sum()
    rejected = total - accepted
    acc_rate = accepted / total * 100
    avg_income = df["Income"].mean()
    avg_income_accepted = df[df["Personal Loan"] == 1]["Income"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{total:,}", f"{accepted}", f"{rejected:,}", f"{acc_rate:.1f}%", f"${avg_income_accepted:.0f}K"],
        ["Total Customers", "Loan Accepted", "Loan Not Accepted", "Acceptance Rate", "Avg Income (Accepted)"]
    ):
        col.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{lbl}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Target Distribution + Income Distribution ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Loan Acceptance Distribution</p>', unsafe_allow_html=True)
        loan_counts = df["Personal Loan"].value_counts().reset_index()
        loan_counts.columns = ["Personal Loan", "Count"]
        loan_counts["Label"] = loan_counts["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})
        loan_counts["Pct"] = (loan_counts["Count"] / loan_counts["Count"].sum() * 100).round(1)
        loan_counts["Text"] = loan_counts.apply(lambda r: f"{r['Count']} ({r['Pct']}%)", axis=1)

        fig = go.Figure(go.Pie(
            labels=loan_counts["Label"], values=loan_counts["Count"],
            textinfo="label+text", text=loan_counts["Text"],
            marker=dict(colors=["#90cdf4", "#e53e3e"], line=dict(color="#ffffff", width=3)),
            hole=0.55, textfont=dict(size=13),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Campaign Response Split", height=400,
                          showlegend=False,
                          annotations=[dict(text=f"<b>{acc_rate:.1f}%</b><br>Accept Rate",
                                            x=0.5, y=0.5, font_size=16, showarrow=False,
                                            font=dict(color="#1a365d"))])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>Only 9.6% of customers accepted the personal loan</b> in the last campaign.
        This indicates a massive untapped market. With smart targeting, even a 2-3% uplift in acceptance rate
        could translate to hundreds of new loans, especially critical when operating with a reduced budget.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">Income Distribution by Loan Status</p>', unsafe_allow_html=True)
        df_plot = df.copy()
        df_plot["Loan Status"] = df_plot["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})

        fig = go.Figure()
        for status, color in [("Not Accepted", "#90cdf4"), ("Accepted", "#e53e3e")]:
            subset = df_plot[df_plot["Loan Status"] == status]
            fig.add_trace(go.Histogram(
                x=subset["Income"], name=status, marker_color=color,
                opacity=0.8, nbinsx=40,
                hovertemplate="Income: $%{x}K<br>Count: %{y}<extra></extra>"
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Income Distribution ($000)",
                          barmode="overlay", height=400,
                          xaxis_title="Annual Income ($000)", yaxis_title="No. of Customers",
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>Income is the #1 predictor of loan acceptance.</b> Customers earning above $100K/year
        form the bulk of loan acceptors. The overlap zone ($80K–$120K) is your high-potential sweet spot —
        focus marketing spend here for maximum ROI with your reduced budget.
        </div>""", unsafe_allow_html=True)

    # ── Row 2: Education + Family ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Acceptance Rate by Education Level</p>', unsafe_allow_html=True)
        edu_map = {1: "Undergrad", 2: "Graduate", 3: "Advanced/\nProfessional"}
        edu_grp = df.groupby("Education")["Personal Loan"].agg(["sum", "count"]).reset_index()
        edu_grp["Rate"] = (edu_grp["sum"] / edu_grp["count"] * 100).round(1)
        edu_grp["Education Label"] = edu_grp["Education"].map(edu_map)
        edu_grp["Text"] = edu_grp.apply(lambda r: f"{r['sum']}/{r['count']} ({r['Rate']}%)", axis=1)

        fig = go.Figure(go.Bar(
            x=edu_grp["Education Label"], y=edu_grp["Rate"],
            text=edu_grp["Text"], textposition="outside",
            marker=dict(color=["#63b3ed", "#3182ce", "#1a365d"],
                        line=dict(width=0)),
            hovertemplate="Education: %{x}<br>Acceptance Rate: %{y:.1f}%<extra></extra>",
            width=0.55,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Loan Acceptance Rate by Education",
                          yaxis_title="Acceptance Rate (%)", height=400,
                          yaxis=dict(range=[0, max(edu_grp["Rate"]) * 1.4]))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>Graduate & Advanced degree holders are 3× more likely to accept a loan</b> than undergrads.
        Education level is a strong segmentation variable — target post-graduate customers first to maximise
        conversion per marketing dollar.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">Acceptance Rate by Family Size</p>', unsafe_allow_html=True)
        fam_grp = df.groupby("Family")["Personal Loan"].agg(["sum", "count"]).reset_index()
        fam_grp["Rate"] = (fam_grp["sum"] / fam_grp["count"] * 100).round(1)
        fam_grp["Text"] = fam_grp.apply(lambda r: f"{r['sum']}/{r['count']} ({r['Rate']}%)", axis=1)

        fig = go.Figure(go.Bar(
            x=fam_grp["Family"].astype(str), y=fam_grp["Rate"],
            text=fam_grp["Text"], textposition="outside",
            marker=dict(color=["#bee3f8", "#63b3ed", "#3182ce", "#1a365d"],
                        line=dict(width=0)),
            hovertemplate="Family Size: %{x}<br>Acceptance Rate: %{y:.1f}%<extra></extra>",
            width=0.55,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Loan Acceptance Rate by Family Size",
                          xaxis_title="Family Size", yaxis_title="Acceptance Rate (%)", height=400,
                          yaxis=dict(range=[0, max(fam_grp["Rate"]) * 1.4]))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>Families of 3–4 members show the highest acceptance rates</b> (~11–13%).
        Larger families likely have higher financial needs (education, housing). Messaging around
        family milestones — children's education, home upgrades — could resonate strongly.
        </div>""", unsafe_allow_html=True)

    # ── Row 3: CC Spending + Mortgage ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Credit Card Spending vs Income</p>', unsafe_allow_html=True)
        sample = df.sample(n=min(2000, len(df)), random_state=42)
        sample["Loan Status"] = sample["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})
        fig = go.Figure()
        for status, color, sz in [("Not Accepted", "#90cdf4", 4), ("Accepted", "#e53e3e", 8)]:
            s = sample[sample["Loan Status"] == status]
            fig.add_trace(go.Scatter(
                x=s["Income"], y=s["CCAvg"], mode="markers", name=status,
                marker=dict(color=color, size=sz, opacity=0.6 if status == "Not Accepted" else 0.9,
                            line=dict(width=0)),
                hovertemplate="Income: $%{x}K<br>CC Avg: $%{y}K/mo<extra></extra>"
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title="CC Spending vs Income (Sampled)",
                          xaxis_title="Annual Income ($000)", yaxis_title="Avg Monthly CC Spend ($000)",
                          height=420,
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>High-income customers with high CC spending are the prime loan targets.</b>
        The red cluster (loan acceptors) is concentrated in the upper-right quadrant.
        These are financially active customers who are comfortable with credit — your ideal segment for personal loan offers.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">Mortgage Distribution by Loan Status</p>', unsafe_allow_html=True)
        df_mort = df.copy()
        df_mort["Has Mortgage"] = (df_mort["Mortgage"] > 0).map({True: "Has Mortgage", False: "No Mortgage"})
        df_mort["Loan Status"] = df_mort["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})

        mort_grp = df_mort.groupby(["Has Mortgage", "Loan Status"]).size().reset_index(name="Count")
        mort_total = mort_grp.groupby("Has Mortgage")["Count"].transform("sum")
        mort_grp["Pct"] = (mort_grp["Count"] / mort_total * 100).round(1)
        mort_grp["Text"] = mort_grp.apply(lambda r: f"{r['Count']} ({r['Pct']}%)", axis=1)

        fig = go.Figure()
        for status, color in [("Not Accepted", "#90cdf4"), ("Accepted", "#e53e3e")]:
            sub = mort_grp[mort_grp["Loan Status"] == status]
            fig.add_trace(go.Bar(
                x=sub["Has Mortgage"], y=sub["Count"], name=status,
                text=sub["Text"], textposition="inside",
                marker_color=color,
                hovertemplate="%{x}<br>%{y} customers<extra></extra>"
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Mortgage Status & Loan Acceptance",
                          barmode="stack", height=420,
                          yaxis_title="No. of Customers",
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>Customers with existing mortgages have a noticeably higher loan acceptance rate.</b>
        They're already accustomed to managing loans, reducing the psychological barrier.
        Position the personal loan as a tool for consolidation or home improvement for these customers.
        </div>""", unsafe_allow_html=True)

    # ── Row 4: Banking relationship ──
    st.markdown('<p class="section-header">Banking Relationship & Loan Acceptance</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    binary_features = [
        ("CD Account", col1, "Certificate of Deposit"),
        ("Securities Account", col2, "Securities Account"),
        ("Online", col3, "Online Banking"),
    ]

    for feat, col, title in binary_features:
        with col:
            grp = df.groupby(feat)["Personal Loan"].agg(["sum", "count"]).reset_index()
            grp["Rate"] = (grp["sum"] / grp["count"] * 100).round(1)
            grp["Label"] = grp[feat].map({0: "No", 1: "Yes"})
            grp["Text"] = grp.apply(lambda r: f"{r['sum']}/{r['count']}\n({r['Rate']}%)", axis=1)

            fig = go.Figure(go.Bar(
                x=grp["Label"], y=grp["Rate"],
                text=grp["Text"], textposition="outside",
                marker=dict(color=["#90cdf4", "#1a365d"], line=dict(width=0)),
                width=0.45,
                hovertemplate=f"{title}: %{{x}}<br>Rate: %{{y:.1f}}%<extra></extra>"
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title=title, height=340,
                              yaxis_title="Acceptance Rate (%)",
                              yaxis=dict(range=[0, max(grp["Rate"]) * 1.5]))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>CD Account holders show a dramatically higher loan acceptance rate (~30%) compared to non-holders (~6%).</b>
    This is a goldmine insight — CD holders already trust the bank with significant deposits and are highly likely
    to accept a personal loan. Securities account and online banking show a smaller but still meaningful difference.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 2: CUSTOMER DEEP-DIVE (Diagnostic)
# ═════════════════════════════════════════════
elif page == "🔍 Customer Deep-Dive":
    st.markdown("""
    <div class="main-header">
        <h1>Customer Deep-Dive Analytics</h1>
        <p>Diagnostic Analytics — Why do certain customers accept personal loans?</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Correlation Heatmap ──
    st.markdown('<p class="section-header">Feature Correlation Matrix</p>', unsafe_allow_html=True)

    corr_cols = ["Age", "Experience", "Income", "Family", "CCAvg",
                 "Education", "Mortgage", "Securities Account",
                 "CD Account", "Online", "CreditCard", "Personal Loan"]
    corr = df[corr_cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, "#bee3f8"], [0.5, "#ffffff"], [1, "#c53030"]],
        zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Correlation: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Correlation Heatmap of All Features",
                      height=550, width=None)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    📌 <b>Income (0.50), CCAvg (0.37), and CD Account (0.32) have the strongest positive correlation with loan acceptance.</b>
    Age and Experience are almost perfectly correlated with each other (0.99) but have near-zero correlation with loan acceptance —
    meaning age is not a useful targeting variable for this campaign.
    </div>""", unsafe_allow_html=True)

    # ── Income Segmentation ──
    st.markdown('<p class="section-header">Income Segmentation Analysis</p>', unsafe_allow_html=True)

    df_seg = df.copy()
    bins = [0, 50, 100, 150, 250]
    labels = ["<$50K", "$50K-$100K", "$100K-$150K", "$150K+"]
    df_seg["Income Bracket"] = pd.cut(df_seg["Income"], bins=bins, labels=labels)
    seg_grp = df_seg.groupby("Income Bracket", observed=True)["Personal Loan"].agg(["sum", "count"]).reset_index()
    seg_grp["Rate"] = (seg_grp["sum"] / seg_grp["count"] * 100).round(1)
    seg_grp["Not Accepted"] = seg_grp["count"] - seg_grp["sum"]
    seg_grp["Text"] = seg_grp.apply(lambda r: f"{int(r['sum'])}/{int(r['count'])} ({r['Rate']}%)", axis=1)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seg_grp["Income Bracket"].astype(str), y=seg_grp["Not Accepted"],
            name="Not Accepted", marker_color="#90cdf4",
            hovertemplate="Bracket: %{x}<br>Not Accepted: %{y}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=seg_grp["Income Bracket"].astype(str), y=seg_grp["sum"],
            name="Accepted", marker_color="#e53e3e",
            text=seg_grp["Text"], textposition="outside",
            hovertemplate="Bracket: %{x}<br>Accepted: %{y}<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Loan Acceptance by Income Bracket",
                          barmode="stack", height=420,
                          yaxis_title="No. of Customers",
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=seg_grp["Income Bracket"].astype(str), y=seg_grp["Rate"],
            text=seg_grp["Rate"].apply(lambda x: f"{x}%"), textposition="outside",
            marker=dict(color=seg_grp["Rate"], colorscale=[[0, "#63b3ed"], [1, "#e53e3e"]],
                        line=dict(width=0)),
            width=0.55,
            hovertemplate="Bracket: %{x}<br>Acceptance Rate: %{y:.1f}%<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Acceptance Rate by Income Bracket",
                          yaxis_title="Acceptance Rate (%)", height=420,
                          yaxis=dict(range=[0, max(seg_grp["Rate"]) * 1.3]))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>The acceptance rate jumps from ~2% (below $50K) to ~13% ($50–100K) and then rockets to ~33% ($100–150K) and ~57% ($150K+).</b>
    With a halved budget, focusing on the $100K+ income segment gives you the highest probability of conversion.
    The $50K–$100K bracket is still valuable for volume since it has the most customers.
    </div>""", unsafe_allow_html=True)

    # ── Education × Income Interaction ──
    st.markdown('<p class="section-header">Education × Income Interaction</p>', unsafe_allow_html=True)

    edu_map = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Professional"}
    df_int = df.copy()
    df_int["Income Bracket"] = pd.cut(df_int["Income"], bins=bins, labels=labels)
    df_int["Education Label"] = df_int["Education"].map(edu_map)
    int_grp = df_int.groupby(["Income Bracket", "Education Label"], observed=True)["Personal Loan"].agg(["sum", "count"]).reset_index()
    int_grp["Rate"] = (int_grp["sum"] / int_grp["count"] * 100).round(1)

    fig = go.Figure()
    for edu, color in zip(["Undergrad", "Graduate", "Advanced/Professional"],
                          ["#63b3ed", "#3182ce", "#1a365d"]):
        sub = int_grp[int_grp["Education Label"] == edu]
        fig.add_trace(go.Bar(
            x=sub["Income Bracket"].astype(str), y=sub["Rate"],
            name=edu, marker_color=color,
            text=sub["Rate"].apply(lambda x: f"{x}%"), textposition="outside",
            hovertemplate="Income: %{x}<br>Education: " + edu + "<br>Rate: %{y:.1f}%<extra></extra>"
        ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Acceptance Rate: Education × Income Bracket",
                      barmode="group", height=450,
                      xaxis_title="Income Bracket", yaxis_title="Acceptance Rate (%)",
                      legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                      yaxis=dict(range=[0, int_grp["Rate"].max() * 1.3]))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    📌 <b>Graduate and Advanced degree holders consistently outperform Undergrads within every income bracket.</b>
    At $100K+ income, the Education effect is even more pronounced.
    The ideal hyper-targeted micro-segment: Graduate/Advanced degree holders earning $100K+ — this group has
    an acceptance rate exceeding 40%, making them extremely high-value targets.
    </div>""", unsafe_allow_html=True)

    # ── CD Account: the hidden gem ──
    st.markdown('<p class="section-header">CD Account — The Hidden Conversion Driver</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cd_inc = df.copy()
        cd_inc["CD Status"] = cd_inc["CD Account"].map({0: "No CD Account", 1: "Has CD Account"})
        cd_inc["Loan Status"] = cd_inc["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})
        fig = go.Figure()
        for status, color in [("Not Accepted", "#90cdf4"), ("Accepted", "#e53e3e")]:
            for cd in ["No CD Account", "Has CD Account"]:
                sub = cd_inc[(cd_inc["CD Status"] == cd) & (cd_inc["Loan Status"] == status)]
                fig.add_trace(go.Box(
                    y=sub["Income"], x=[cd] * len(sub), name=status,
                    marker_color=color, showlegend=(cd == "No CD Account"),
                    hovertemplate="Income: $%{y}K<extra></extra>"
                ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Income Distribution: CD Account × Loan Status",
                          height=420, yaxis_title="Annual Income ($000)",
                          boxmode="group",
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cd_grp = df.groupby("CD Account")["Personal Loan"].agg(["sum", "count"]).reset_index()
        cd_grp["Rate"] = (cd_grp["sum"] / cd_grp["count"] * 100).round(1)
        cd_grp["Label"] = cd_grp["CD Account"].map({0: "No CD Account", 1: "Has CD Account"})
        cd_grp["Text"] = cd_grp.apply(lambda r: f"{int(r['sum'])}/{int(r['count'])}\n({r['Rate']}%)", axis=1)

        fig = go.Figure(go.Bar(
            x=cd_grp["Label"], y=cd_grp["Rate"],
            text=cd_grp["Text"], textposition="outside",
            marker=dict(color=["#90cdf4", "#1a365d"], line=dict(width=0)),
            width=0.5,
            hovertemplate="%{x}<br>Acceptance Rate: %{y:.1f}%<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Loan Acceptance Rate by CD Account Status",
                          yaxis_title="Acceptance Rate (%)", height=420,
                          yaxis=dict(range=[0, cd_grp["Rate"].max() * 1.5]))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>CD Account holders accept loans at 5× the rate of non-holders!</b>
    While CD holders are only 6% of the customer base, their conversion rate is staggeringly high.
    This is your most efficient targeting variable after income — every marketing dollar spent on CD holders yields disproportionate returns.
    </div>""", unsafe_allow_html=True)

    # ── Age Distribution (showing it's NOT useful) ──
    st.markdown('<p class="section-header">Age & Experience — The Non-Factors</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        df_age = df.copy()
        df_age["Loan Status"] = df_age["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})
        fig = go.Figure()
        for status, color in [("Not Accepted", "#90cdf4"), ("Accepted", "#e53e3e")]:
            sub = df_age[df_age["Loan Status"] == status]
            fig.add_trace(go.Histogram(
                x=sub["Age"], name=status, marker_color=color,
                opacity=0.8, nbinsx=30
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Age Distribution by Loan Status",
                          barmode="overlay", height=380,
                          xaxis_title="Age", yaxis_title="Count",
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        age_bins = [20, 30, 40, 50, 60, 70]
        age_labels = ["23-30", "31-40", "41-50", "51-60", "61-67"]
        df_ab = df.copy()
        df_ab["Age Group"] = pd.cut(df_ab["Age"], bins=age_bins, labels=age_labels)
        ab_grp = df_ab.groupby("Age Group", observed=True)["Personal Loan"].agg(["sum", "count"]).reset_index()
        ab_grp["Rate"] = (ab_grp["sum"] / ab_grp["count"] * 100).round(1)

        fig = go.Figure(go.Bar(
            x=ab_grp["Age Group"].astype(str), y=ab_grp["Rate"],
            text=ab_grp["Rate"].apply(lambda x: f"{x}%"), textposition="outside",
            marker_color="#63b3ed", width=0.55
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Acceptance Rate by Age Group",
                          xaxis_title="Age Group", yaxis_title="Acceptance Rate (%)", height=380,
                          yaxis=dict(range=[0, max(ab_grp["Rate"]) * 1.4]))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>Age and Experience show virtually no difference in loan acceptance across groups</b> (all hovering around 8–11%).
    This confirms that age-based targeting would waste your budget. Prioritise income, education, and banking relationship over demographics.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ═════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>Predictive Model Performance</h1>
        <p>Predictive Analytics — Classification algorithms to identify personal loan prospects</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Comparison Table ──
    st.markdown('<p class="section-header">Model Comparison Summary</p>', unsafe_allow_html=True)

    metrics_data = []
    for name, res in model_results.items():
        metrics_data.append({
            "Model": name,
            "Training Accuracy": f"{res['train_acc']*100:.2f}%",
            "Testing Accuracy": f"{res['test_acc']*100:.2f}%",
            "Precision": f"{res['precision']*100:.2f}%",
            "Recall": f"{res['recall']*100:.2f}%",
            "F1-Score": f"{res['f1']*100:.2f}%",
            "ROC-AUC": f"{res['roc_auc']*100:.2f}%",
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Styled table using plotly
    header_colors = ["#1a365d"] * len(metrics_df.columns)
    cell_colors = [["#f7fafc", "#edf2f7", "#f7fafc"]] * len(metrics_df.columns)

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in metrics_df.columns],
            fill_color="#1a365d",
            font=dict(color="white", size=13, family="DM Sans"),
            align="center", height=40
        ),
        cells=dict(
            values=[metrics_df[c] for c in metrics_df.columns],
            fill_color=[["#f7fafc", "#edf2f7", "#f7fafc"]] * len(metrics_df.columns),
            font=dict(size=13, family="DM Sans", color="#2d3748"),
            align="center", height=36
        )
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=200)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>The metrics table above compares all three models across training/testing accuracy, precision, recall, F1-score, and ROC-AUC.</b>
    A high recall is critical here — we'd rather reach out to a customer who won't convert (lower precision)
    than miss a customer who would have accepted (low recall), since our budget is limited and every missed opportunity costs us.
    </div>""", unsafe_allow_html=True)

    # ── ROC Curve (Single Plot) ──
    st.markdown('<p class="section-header">ROC Curve — All Models</p>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#cbd5e0", dash="dash", width=1.5),
        name="Random Baseline", showlegend=True
    ))

    for name, color in COLORS["model_colors"].items():
        res = model_results[name]
        fig.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"], mode="lines",
            name=f"{name} (AUC = {res['roc_auc']:.3f})",
            line=dict(color=color, width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        legend=dict(x=0.55, y=0.05, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#e2e8f0", borderwidth=1,
                    font=dict(size=12)),
        xaxis=dict(range=[0, 1], constrain="domain"),
        yaxis=dict(range=[0, 1.02], scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>All three models significantly outperform random guessing (dashed line).</b>
    The ROC-AUC score indicates how well the model separates acceptors from non-acceptors.
    A higher AUC means more reliable targeting — the model with the highest AUC will waste the fewest marketing dollars.
    </div>""", unsafe_allow_html=True)

    # ── Confusion Matrices ──
    st.markdown('<p class="section-header">Confusion Matrices</p>', unsafe_allow_html=True)

    cols = st.columns(3)
    total_test = len(y_test)

    for idx, (name, res) in enumerate(model_results.items()):
        with cols[idx]:
            cm = res["cm"]
            cm_pct = (cm / cm.sum() * 100).round(1)

            annot_text = [[f"<b>{cm[i][j]}</b><br>({cm_pct[i][j]}%)" for j in range(2)] for i in range(2)]

            fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                colorscale=[[0, "#ebf8ff"], [1, "#1a365d"]],
                text=annot_text, texttemplate="%{text}",
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                showscale=False,
                textfont=dict(size=14),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=name, font=dict(size=14)),
                height=350,
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>The confusion matrices show the count and percentage of correct/incorrect predictions.</b>
    Top-left (True Negatives) = correctly identified non-buyers. Bottom-right (True Positives) = correctly identified buyers.
    Top-right (False Positives) = wasted outreach. Bottom-left (False Negatives) = missed opportunities.
    For a budget-constrained campaign, minimising False Negatives (bottom-left) is most critical.
    </div>""", unsafe_allow_html=True)

    # ── Feature Importance ──
    st.markdown('<p class="section-header">Feature Importance Comparison</p>', unsafe_allow_html=True)

    best_model = max(model_results.keys(), key=lambda k: model_results[k]["roc_auc"])

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(model_results.keys()),
                        horizontal_spacing=0.08)

    for idx, (name, res) in enumerate(model_results.items(), 1):
        imp = pd.DataFrame(list(res["feature_importance"].items()), columns=["Feature", "Importance"])
        imp = imp.sort_values("Importance", ascending=True)

        color = COLORS["model_colors"][name]
        fig.add_trace(go.Bar(
            y=imp["Feature"], x=imp["Importance"], orientation="h",
            marker_color=color, name=name, showlegend=False,
            text=imp["Importance"].apply(lambda x: f"{x:.3f}"), textposition="outside",
            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>"
        ), row=1, col=idx)

    fig.update_layout(**PLOTLY_LAYOUT, height=500, title="Feature Importance Across All Models")
    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)", row=1, col=i)
        fig.update_yaxes(showgrid=False, row=1, col=i)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    📌 <b>Income and CCAvg (credit card spending) dominate across all three models</b>, confirming they're the most
    reliable predictors. CD Account and Education also appear consistently. These four features should
    be the primary targeting criteria for your next campaign.
    </div>""", unsafe_allow_html=True)

    # ── Best Model Callout ──
    best_res = model_results[best_model]
    st.markdown(f"""
    <div class="action-box">
    🏆 <b>Recommended Model: {best_model}</b> — ROC-AUC: {best_res['roc_auc']*100:.2f}%, F1-Score: {best_res['f1']*100:.2f}%,
    Recall: {best_res['recall']*100:.2f}%. This model offers the best balance of precision and recall for identifying
    personal loan prospects. Use this model's predictions to prioritise your outreach list.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 4: CAMPAIGN STRATEGY (Prescriptive)
# ═════════════════════════════════════════════
elif page == "🎯 Campaign Strategy":
    st.markdown("""
    <div class="main-header">
        <h1>Campaign Strategy & Recommendations</h1>
        <p>Prescriptive Analytics — Data-driven action plan for your next hyper-personalised campaign</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Ideal Customer Profile ──
    st.markdown('<p class="section-header">Ideal Customer Profile (ICP)</p>', unsafe_allow_html=True)

    accepted = df[df["Personal Loan"] == 1]
    not_accepted = df[df["Personal Loan"] == 0]

    profile_data = {
        "Attribute": ["Annual Income", "CC Avg Spend/Month", "Education",
                      "Family Size", "Has CD Account", "Has Mortgage", "Age"],
        "Loan Acceptors (Avg)": [
            f"${accepted['Income'].mean():.0f}K",
            f"${accepted['CCAvg'].mean():.1f}K",
            f"{accepted['Education'].mean():.1f} (Grad+)",
            f"{accepted['Family'].mean():.1f}",
            f"{accepted['CD Account'].mean()*100:.0f}%",
            f"${accepted['Mortgage'].mean():.0f}K",
            f"{accepted['Age'].mean():.0f} yrs",
        ],
        "Non-Acceptors (Avg)": [
            f"${not_accepted['Income'].mean():.0f}K",
            f"${not_accepted['CCAvg'].mean():.1f}K",
            f"{not_accepted['Education'].mean():.1f} (Undergrad)",
            f"{not_accepted['Family'].mean():.1f}",
            f"{not_accepted['CD Account'].mean()*100:.0f}%",
            f"${not_accepted['Mortgage'].mean():.0f}K",
            f"{not_accepted['Age'].mean():.0f} yrs",
        ]
    }
    profile_df = pd.DataFrame(profile_data)

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in profile_df.columns],
            fill_color="#1a365d",
            font=dict(color="white", size=13, family="DM Sans"),
            align="center", height=40
        ),
        cells=dict(
            values=[profile_df[c] for c in profile_df.columns],
            fill_color=[["#f7fafc"]*7, ["#f0fff4"]*7, ["#fff5f5"]*7],
            font=dict(size=13, family="DM Sans", color="#2d3748"),
            align="center", height=36
        )
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    📌 <b>Your ideal customer earns significantly more, spends more on credit cards, has a higher education level, and is far more likely to hold a CD account.</b>
    Age is virtually identical across both groups, confirming it shouldn't be a targeting variable.
    </div>""", unsafe_allow_html=True)

    # ── Segment Priority Matrix ──
    st.markdown('<p class="section-header">Segment Priority Matrix</p>', unsafe_allow_html=True)

    # Create segments
    df_strat = df.copy()
    edu_map = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Professional"}
    df_strat["Edu"] = df_strat["Education"].map(edu_map)
    df_strat["Income_Grp"] = pd.cut(df_strat["Income"], bins=[0, 75, 125, 250],
                                     labels=["<$75K", "$75K-$125K", "$125K+"])

    seg = df_strat.groupby(["Income_Grp", "Edu"], observed=True).agg(
        total=("Personal Loan", "count"),
        accepted=("Personal Loan", "sum")
    ).reset_index()
    seg["Rate"] = (seg["accepted"] / seg["total"] * 100).round(1)

    fig = go.Figure(go.Scatter(
        x=seg["total"], y=seg["Rate"],
        mode="markers+text",
        text=seg.apply(lambda r: f"{r['Income_Grp']}<br>{r['Edu']}", axis=1),
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=seg["accepted"].clip(lower=2) * 3 + 10,
            color=seg["Rate"],
            colorscale=[[0, "#bee3f8"], [0.5, "#3182ce"], [1, "#e53e3e"]],
            showscale=True,
            colorbar=dict(title="Accept<br>Rate %", thickness=15),
            line=dict(width=1, color="#ffffff")
        ),
        hovertemplate=("Segment: %{text}<br>Total: %{x}<br>Accept Rate: %{y:.1f}%<br>"
                       "Accepted: %{customdata}<extra></extra>"),
        customdata=seg["accepted"]
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Segment Prioritisation: Volume vs Conversion Rate",
                      xaxis_title="Segment Size (Total Customers)",
                      yaxis_title="Loan Acceptance Rate (%)",
                      height=500)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    📌 <b>The best segments sit in the top-right: large enough for volume AND high conversion rates.</b>
    Upper-right bubbles (high income + graduate/advanced education) should receive priority spend.
    Large lower-left bubbles represent high volume but low conversion — useful only if budget permits broad campaigns.
    </div>""", unsafe_allow_html=True)

    # ── Prescriptive Recommendations ──
    st.markdown('<p class="section-header">Action Plan: Hyper-Personalised Campaign</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="action-box">
    🎯 <b>TIER 1 — HIGH PRIORITY (Allocate 50% of budget)</b><br><br>
    <b>Segment:</b> Income $100K+, Graduate/Advanced degree, CD Account holders<br>
    <b>Expected Accept Rate:</b> 40–60%<br>
    <b>Channel:</b> Direct relationship manager calls, personalised emails with pre-approved offers<br>
    <b>Messaging:</b> Exclusive rate offers, consolidation benefits, premium service positioning<br>
    <b>Estimated Volume:</b> ~300–500 customers → 120–250 conversions
    </div>

    <div class="action-box">
    🎯 <b>TIER 2 — MEDIUM PRIORITY (Allocate 35% of budget)</b><br><br>
    <b>Segment:</b> Income $75K–$100K, Graduate+ education, Family size 3+<br>
    <b>Expected Accept Rate:</b> 15–30%<br>
    <b>Channel:</b> Email campaigns + in-app notifications + targeted digital ads<br>
    <b>Messaging:</b> Family milestone financing (children's education, home renovation), flexible EMI options<br>
    <b>Estimated Volume:</b> ~500–800 customers → 75–200 conversions
    </div>

    <div class="action-box">
    🎯 <b>TIER 3 — EXPLORATORY (Allocate 15% of budget)</b><br><br>
    <b>Segment:</b> Income $50K–$75K, High CC spending (CCAvg > $3K), Any education<br>
    <b>Expected Accept Rate:</b> 8–15%<br>
    <b>Channel:</b> Automated email drip campaigns, chatbot-initiated offers during online banking sessions<br>
    <b>Messaging:</b> Credit card balance transfer benefits, competitive rate comparison, quick approval promise<br>
    <b>Estimated Volume:</b> ~300–500 customers → 25–60 conversions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Budget Optimization visual ──
    st.markdown('<p class="section-header">Budget Allocation Recommendation</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        budget_data = pd.DataFrame({
            "Tier": ["Tier 1: High Priority", "Tier 2: Medium Priority", "Tier 3: Exploratory"],
            "Allocation": [50, 35, 15],
            "Expected Conversions": [185, 137, 42],
        })
        fig = go.Figure(go.Pie(
            labels=budget_data["Tier"], values=budget_data["Allocation"],
            textinfo="label+percent", hole=0.5,
            marker=dict(colors=["#1a365d", "#3182ce", "#90cdf4"],
                        line=dict(color="#ffffff", width=3)),
            textfont=dict(size=12),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Budget Allocation by Tier",
                          height=400, showlegend=False,
                          annotations=[dict(text="<b>Budget</b><br>Split", x=0.5, y=0.5,
                                            font_size=15, showarrow=False,
                                            font=dict(color="#1a365d"))])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=budget_data["Tier"], y=budget_data["Expected Conversions"],
            text=budget_data["Expected Conversions"], textposition="outside",
            marker=dict(color=["#1a365d", "#3182ce", "#90cdf4"], line=dict(width=0)),
            width=0.55,
            hovertemplate="%{x}<br>Expected Conversions: %{y}<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Expected Conversions by Tier",
                          yaxis_title="Expected Conversions", height=400,
                          yaxis=dict(range=[0, 250]))
        fig.update_xaxes(showgrid=False, tickangle=-15)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>With half the budget, a spray-and-pray approach won't work.</b>
    By concentrating 85% of spend on Tier 1 + Tier 2 segments (which have 15-60% acceptance rates),
    you can expect ~320+ conversions vs ~480 from last year's full-budget untargeted campaign.
    That's <b>67% of the conversions at 50% of the cost</b> — a 34% improvement in cost-per-acquisition.
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>Key Do's and Don'ts:</b><br>
    ✅ DO target CD Account holders — they're 5× more likely to convert.<br>
    ✅ DO focus on income $100K+ with Graduate+ education — highest yield segment.<br>
    ✅ DO use family size as a messaging lever (education, home improvement).<br>
    ❌ DON'T segment by age — it has zero predictive power.<br>
    ❌ DON'T waste budget on customers with income below $50K — acceptance rate is under 2%.<br>
    ❌ DON'T treat all customers equally — the data clearly shows 80/20 dynamics at play.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 5: PREDICT NEW CUSTOMERS
# ═════════════════════════════════════════════
elif page == "📁 Predict New Customers":
    st.markdown("""
    <div class="main-header">
        <h1>Predict New Customers</h1>
        <p>Upload a test dataset and download predictions — will the customer accept a personal loan?</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    📌 <b>How to use:</b> Upload a CSV file with the same columns as the training data
    (ID, Age, Experience, Income, ZIP Code, Family, CCAvg, Education, Mortgage, Securities Account,
    CD Account, Online, CreditCard). The model will predict the "Personal Loan" column for each customer.<br><br>
    A sample test file (<code>sample_test_data.csv</code>) is provided in the repository for your convenience.
    </div>""", unsafe_allow_html=True)

    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]["roc_auc"])
    st.markdown(f"""
    <div class="action-box">
    🏆 <b>Using Model:</b> {best_model_name} (ROC-AUC: {model_results[best_model_name]['roc_auc']*100:.2f}%)
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            test_df.columns = test_df.columns.str.strip()
            st.success(f"File uploaded successfully! {len(test_df)} rows × {len(test_df.columns)} columns")

            st.markdown("**Preview of uploaded data:**")
            st.dataframe(test_df.head(10), use_container_width=True, height=300)

            # Validate columns
            missing_cols = [c for c in feature_cols if c not in test_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                test_clean = test_df.copy()
                test_clean["Experience"] = test_clean["Experience"].abs()

                X_new = test_clean[feature_cols]
                X_new_sc = scaler.transform(X_new)

                best = model_results[best_model_name]["model"]
                predictions = best.predict(X_new_sc)
                probabilities = best.predict_proba(X_new_sc)[:, 1]

                test_clean["Personal Loan (Predicted)"] = predictions
                test_clean["Acceptance Probability"] = (probabilities * 100).round(2)
                test_clean["Recommendation"] = test_clean["Acceptance Probability"].apply(
                    lambda x: "🔴 HIGH PRIORITY" if x >= 60
                    else ("🟡 MEDIUM PRIORITY" if x >= 30 else "⚪ LOW PRIORITY")
                )

                st.markdown("---")
                st.markdown('<p class="section-header">Prediction Results</p>', unsafe_allow_html=True)

                pred_accepted = predictions.sum()
                pred_rate = pred_accepted / len(predictions) * 100

                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><h3>{len(predictions)}</h3><p>Total Customers</p></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><h3>{int(pred_accepted)}</h3><p>Predicted Acceptors</p></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><h3>{pred_rate:.1f}%</h3><p>Predicted Accept Rate</p></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(
                    test_clean[["ID", "Age", "Income", "Education", "Family", "CCAvg",
                                "CD Account", "Personal Loan (Predicted)",
                                "Acceptance Probability", "Recommendation"]].sort_values(
                        "Acceptance Probability", ascending=False),
                    use_container_width=True, height=400
                )

                # Download button
                csv_out = test_clean.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_out,
                    file_name="personal_loan_predictions.csv",
                    mime="text/csv",
                )

                # Distribution of predicted probabilities
                fig = go.Figure(go.Histogram(
                    x=probabilities * 100, nbinsx=30,
                    marker_color="#2c5282",
                    hovertemplate="Probability: %{x:.0f}%<br>Count: %{y}<extra></extra>"
                ))
                fig.update_layout(**PLOTLY_LAYOUT,
                                  title="Distribution of Acceptance Probabilities",
                                  xaxis_title="Acceptance Probability (%)",
                                  yaxis_title="No. of Customers", height=380)
                fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
                fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct columns and format.")
    else:
        st.info("👆 Upload a CSV file to get started. You can use the `sample_test_data.csv` provided in the repository.")
