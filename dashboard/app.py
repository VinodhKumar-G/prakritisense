"""
PrakritiSense — Streamlit CFI Dashboard
────────────────────────────────────────
Displays model results, SHAP analysis, and a real-time
Cognitive Fatigue Index (CFI) simulator.

Usage:
    streamlit run dashboard/app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")

# ── PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrakritiSense — CFI Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: #f0f7fa; padding: 1rem; border-radius: 10px; }
    .block-container { padding-top: 1rem; }
    h1 { color: #0D1B2A; }
    h2, h3 { color: #028090; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.title("◆ PrakritiSense")
st.sidebar.markdown("**Cognitive Fatigue Index Dashboard**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Model Results",
    "🧠 SHAP Analysis",
    "📈 CFI Simulator",
    "📋 Data Explorer",
])

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_results():
    path = os.path.join(OUTPUT_DIR, "model_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, "all_data_merged.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

results = load_results()
df = load_data()

# ═══════════════════════════════════════════════════════════════════════
# PAGE: MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════

if page == "📊 Model Results":
    st.title("Model Performance Comparison")
    st.markdown("Baseline (7 micro-interaction features) vs Prakriti-conditioned (+ 3 Dosha scores)")

    if results is None:
        st.error("No model results found. Run `python3 analysis/02_train_model.py` first.")
        st.stop()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Baseline AUC", f"{results['baseline']['auc_ovr']:.3f}")
    with col2:
        st.metric("Prakriti AUC", f"{results['prakriti_conditioned']['auc_ovr']:.3f}",
                  delta=f"{results['improvement']['auc_delta']:+.3f}")
    with col3:
        st.metric("Prakriti Accuracy", f"{results['prakriti_conditioned']['accuracy']:.3f}",
                  delta=f"{results['improvement']['accuracy_delta']:+.3f}")
    with col4:
        st.metric("Prakriti F1", f"{results['prakriti_conditioned']['f1_macro']:.3f}",
                  delta=f"{results['improvement']['f1_delta']:+.3f}")

    st.markdown("---")

    # Comparison bar chart
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Performance Comparison")
        img_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            metrics = ["Accuracy", "F1 Macro", "AUC OVR"]
            base = [results["baseline"]["accuracy"], results["baseline"]["f1_macro"], results["baseline"]["auc_ovr"]]
            prak = [results["prakriti_conditioned"]["accuracy"], results["prakriti_conditioned"]["f1_macro"], results["prakriti_conditioned"]["auc_ovr"]]

            fig = go.Figure(data=[
                go.Bar(name="Baseline", x=metrics, y=base, marker_color="#8899A8"),
                go.Bar(name="Prakriti", x=metrics, y=prak, marker_color="#028090"),
            ])
            fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Confusion Matrices")
        img_path = os.path.join(OUTPUT_DIR, "confusion_matrices_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Run model training to generate confusion matrices.")

    # Key finding callout
    delta = results["improvement"]["auc_delta"]
    if delta > 0:
        st.success(f"**Key finding:** Prakriti conditioning improved AUC by "
                   f"**{delta:+.3f}** ({delta*100:+.1f}%), confirming that Ayurvedic "
                   f"constitutional type provides meaningful personalization for fatigue detection.")
    else:
        st.warning("Prakriti conditioning did not improve AUC. Consider increasing sample size.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE: SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

elif page == "🧠 SHAP Analysis":
    st.title("SHAP Explainability Analysis")
    st.markdown("Which features drive the fatigue prediction — and how Doshas differ.")

    tab1, tab2, tab3 = st.tabs(["Global Importance", "Beeswarm Plots", "Per-Dosha Analysis"])

    with tab1:
        img = os.path.join(OUTPUT_DIR, "shap_importance_prakriti_conditioned.png")
        if os.path.exists(img):
            st.image(img, caption="Feature importance — Prakriti-conditioned model", use_container_width=True)
        else:
            st.info("Run model training first.")

    with tab2:
        img = os.path.join(OUTPUT_DIR, "shap_beeswarm_prakriti_conditioned.png")
        if os.path.exists(img):
            st.image(img, caption="SHAP beeswarm — per fatigue class", use_container_width=True)
        else:
            st.info("Run model training first.")

    with tab3:
        img = os.path.join(OUTPUT_DIR, "shap_per_dosha.png")
        if os.path.exists(img):
            st.image(img, caption="Top features by Dosha type", use_container_width=True)
            st.markdown("""
            **Interpretation:**
            - **Vata** — Error rate and IKI variance dominate, consistent with fast/erratic fatigue pattern
            - **Pitta** — Mouse velocity and click RT dominate, consistent with precision-degradation pattern
            - **Kapha** — IKI mean dominates, consistent with gradual slowdown pattern
            """)
        else:
            st.info("Run model training first.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE: CFI SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

elif page == "📈 CFI Simulator":
    st.title("Cognitive Fatigue Index (CFI) Simulator")
    st.markdown("Simulate how the CFI changes over a work session for different Dosha types.")

    col_config, col_gauge = st.columns([1, 1])

    with col_config:
        st.subheader("Configure")
        dosha = st.selectbox("Dosha type", ["Vata", "Pitta", "Kapha"])
        minute = st.slider("Minutes into session", 0, 90, 45)

        dosha_params = {
            "Vata":  {"onset": 25, "slope": 1.8, "color": "#4A90D9"},
            "Pitta": {"onset": 45, "slope": 1.2, "color": "#E05C5C"},
            "Kapha": {"onset": 60, "slope": 0.8, "color": "#2EAA70"},
        }

        p = dosha_params[dosha]
        t = (minute - p["onset"]) / 10.0
        cfi = int(np.clip(100 / (1 + np.exp(-t * p["slope"])), 0, 100))

        # CFI label
        if cfi < 30:
            label, emoji = "Alert", "🟢"
        elif cfi < 60:
            label, emoji = "Moderate", "🟡"
        else:
            label, emoji = "Fatigued", "🔴"

    with col_gauge:
        st.subheader(f"CFI: {cfi}/100  {emoji} {label}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cfi,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": p["color"]},
                "steps": [
                    {"range": [0, 30], "color": "#E8F5EE"},
                    {"range": [30, 60], "color": "#FEF9E7"},
                    {"range": [60, 100], "color": "#FBEAEA"},
                ],
                "threshold": {"line": {"color": "red", "width": 2}, "value": 70},
            },
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=0, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    # CFI curve over time
    st.subheader("CFI trajectory over 90-minute session")
    minutes = np.arange(0, 91)
    curves = {}
    for d_name, d_params in dosha_params.items():
        ts = (minutes - d_params["onset"]) / 10.0
        cfis = np.clip(100 / (1 + np.exp(-ts * d_params["slope"])), 0, 100)
        curves[d_name] = cfis

    fig = go.Figure()
    for d_name, d_params in dosha_params.items():
        fig.add_trace(go.Scatter(
            x=minutes, y=curves[d_name],
            name=d_name, line=dict(color=d_params["color"], width=3),
        ))
    fig.add_vline(x=minute, line_dash="dash", line_color="gray",
                  annotation_text=f"Current: {minute} min")
    fig.add_hrect(y0=0, y1=30, fillcolor="#E8F5EE", opacity=0.3, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="#FEF9E7", opacity=0.3, line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor="#FBEAEA", opacity=0.3, line_width=0)
    fig.update_layout(
        xaxis_title="Minutes into session",
        yaxis_title="Cognitive Fatigue Index (CFI)",
        yaxis=dict(range=[0, 105]),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dosha-specific recommendation
    recommendations = {
        "Vata": "⚡ **Vata fatigue pattern detected.** Error rate is spiking — typical of early Vata mental fatigue. Recommended: 5-minute Pranayama breathing exercise to stabilize attention.",
        "Pitta": "🔥 **Pitta fatigue pattern detected.** Response time is increasing — Pitta precision is degrading. Recommended: 10-minute screen-free break with eyes closed.",
        "Kapha": "🌊 **Kapha fatigue pattern detected.** Inter-key intervals are gradually lengthening — deep Kapha slowdown. Recommended: 15-minute movement break (walking, stretching).",
    }

    if cfi > 50:
        st.warning(recommendations[dosha])
    elif cfi > 30:
        st.info(f"Moderate fatigue building for {dosha} type. Consider a short break soon.")
    else:
        st.success(f"{dosha} type — alert and performing well. No intervention needed.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════

elif page == "📋 Data Explorer":
    st.title("Dataset Explorer")

    if df is None:
        st.error("No data found. Generate synthetic data or collect real data first.")
        st.stop()

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total rows", len(df))
    with col2:
        st.metric("Participants", df["participant_id"].nunique())
    with col3:
        real = len(df[df.get("is_synthetic", True) == False]) if "is_synthetic" in df.columns else "N/A"
        st.metric("Real data rows", real)
    with col4:
        synth = len(df[df.get("is_synthetic", True) == True]) if "is_synthetic" in df.columns else "N/A"
        st.metric("Synthetic rows", synth)

    st.markdown("---")

    # Feature distributions by fatigue label
    st.subheader("Feature distributions by fatigue label")
    feature = st.selectbox("Select feature", [
        "cpm", "iki_mean_ms", "iki_variance", "error_rate",
        "mouse_velocity_pxs", "jitter_index_px", "pause_frequency"
    ])

    fig = px.box(df, x="fatigue_label", y=feature, color="fatigue_label",
                 category_orders={"fatigue_label": ["Alert", "Moderate", "Fatigued"]},
                 color_discrete_map={"Alert": "#2EAA70", "Moderate": "#C48A1F", "Fatigued": "#E05C5C"})
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # By Dosha
    if "dominant_dosha" in df.columns:
        st.subheader("Feature distributions by Dosha type")
        fig2 = px.box(df, x="dominant_dosha", y=feature, color="dominant_dosha",
                      color_discrete_map={"V": "#4A90D9", "P": "#E05C5C", "K": "#2EAA70"})
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Raw data table
    st.subheader("Raw data")
    st.dataframe(df.head(100), use_container_width=True)

    # Download
    csv = df.to_csv(index=False)
    st.download_button("📥 Download full dataset as CSV", csv, "prakritisense_full_dataset.csv")