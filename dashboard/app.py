"""
PrakritiSense — Streamlit CFI Dashboard (Fixed)
Run: streamlit run dashboard/app.py --server.port 8501 --server.headless true
"""
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")

st.set_page_config(page_title="PrakritiSense", page_icon="◆", layout="wide")

st.sidebar.title("◆ PrakritiSense")
st.sidebar.markdown("**Cognitive Fatigue Index Dashboard**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Model Results","🧠 SHAP Analysis","📈 CFI Simulator","📋 Data Explorer","🔬 EEG Benchmark"])

def find_img(*names):
    for n in names:
        p = os.path.join(OUT_DIR, n)
        if os.path.exists(p): return p
    return None

@st.cache_data
def load_results():
    p = os.path.join(OUT_DIR, "model_results.json")
    return json.load(open(p)) if os.path.exists(p) else None
@st.cache_data
def load_eeg():
    p = os.path.join(OUT_DIR, "eeg_benchmark_results.json")
    return json.load(open(p)) if os.path.exists(p) else None
@st.cache_data
def load_df():
    p = os.path.join(DATA_DIR, "all_data_merged.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

results = load_results()
eeg_res = load_eeg()
df = load_df()

# ═══ MODEL RESULTS ════════════════════════════════════════════════════
if page == "📊 Model Results":
    st.title("Model Performance Comparison")
    if not results:
        st.error("Run `python3 analysis/run_full.py` first."); st.stop()
    bl, pc, imp = results["baseline"], results["prakriti_conditioned"], results.get("improvement",{})
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Baseline AUC", f"{bl['auc_ovr']:.3f}")
    c2.metric("Prakriti AUC", f"{pc['auc_ovr']:.3f}", delta=f"{imp.get('auc_delta',0):+.3f}")
    c3.metric("Prakriti Acc", f"{pc['accuracy']:.3f}", delta=f"{imp.get('accuracy_delta',0):+.3f}")
    c4.metric("Prakriti F1", f"{pc['f1_macro']:.3f}", delta=f"{imp.get('f1_delta',0):+.3f}")
    st.markdown("---")
    a,b = st.columns(2)
    with a:
        img = find_img("model_comparison.png")
        if img: st.image(img, use_container_width=True)
    with b:
        img = find_img("confusion_matrices_comparison.png")
        if img: st.image(img, use_container_width=True)

# ═══ SHAP ═════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Analysis":
    st.title("SHAP Explainability Analysis")
    t1,t2,t3 = st.tabs(["Global Importance","Beeswarm Plots","Per-Dosha Analysis"])
    with t1:
        img = find_img("shap_importance_prakriti.png","shap_importance_prakriti_conditioned.png")
        if img:
            st.image(img, use_container_width=True)
            st.markdown("**Teal** = micro-interaction features | **Gold** = Prakriti features\n\nLonger bar = more important for prediction.")
        else: st.warning("Not found. Run `python3 analysis/run_full.py` first.")
    with t2:
        img = find_img("shap_beeswarm_prakriti.png","shap_beeswarm_prakriti_conditioned.png")
        if img:
            st.image(img, use_container_width=True)
            st.markdown("Each dot = one 60-second window. **Red** = high value, **Blue** = low value. Dots pushed RIGHT = feature pushes toward that class.")
        else: st.warning("Not found. Run `python3 analysis/run_full.py` first.")
    with t3:
        img = find_img("shap_per_dosha.png")
        if img:
            st.image(img, use_container_width=True)
            st.markdown("""**Core finding:** Different Doshas have different top features:
- **Vata** → speed features (CPM, IKI mean) — fatigue = sudden speed collapse
- **Pitta** → precision features (mouse velocity, pause frequency) — fatigue = precision loss
- **Kapha** → stability features (IKI variance, jitter) — fatigue = gradual rhythm disruption""")
        else: st.warning("Not found. Run `python3 analysis/run_full.py` first.")

# ═══ CFI SIMULATOR ════════════════════════════════════════════════════
elif page == "📈 CFI Simulator":
    st.title("Cognitive Fatigue Index (CFI) Simulator")
    dp = {"Vata":{"onset":25,"slope":1.8,"color":"#4A90D9","e":"🌬️"},
          "Pitta":{"onset":45,"slope":1.2,"color":"#E05C5C","e":"🔥"},
          "Kapha":{"onset":60,"slope":0.8,"color":"#2EAA70","e":"🌊"}}
    c1,c2 = st.columns([1,1])
    with c1:
        dosha = st.selectbox("Dosha type",["Vata","Pitta","Kapha"])
        minute = st.slider("Minutes into session",0,90,45)
    p = dp[dosha]
    t = (minute - p["onset"])/10.0
    cfi = int(np.clip(100/(1+np.exp(-t*p["slope"])),0,100))
    label = "Alert" if cfi<30 else "Moderate" if cfi<60 else "Fatigued"
    with c2:
        st.subheader(f"{p['e']} {dosha} — CFI: {cfi}/100 ({label})")
        fig = go.Figure(go.Indicator(mode="gauge+number",value=cfi,
            gauge={"axis":{"range":[0,100]},"bar":{"color":p["color"]},
                   "steps":[{"range":[0,30],"color":"#E8F5EE"},{"range":[30,60],"color":"#FEF9E7"},{"range":[60,100],"color":"#FBEAEA"}],
                   "threshold":{"line":{"color":"red","width":2},"value":70}}))
        fig.update_layout(height=260,margin=dict(t=30,b=0,l=30,r=30))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("CFI trajectory — all three Doshas")
    mins = np.arange(0,91)
    fig = go.Figure()
    for dn,dd in dp.items():
        ts = (mins-dd["onset"])/10.0
        cfis = np.clip(100/(1+np.exp(-ts*dd["slope"])),0,100)
        sel = dn==dosha
        fig.add_trace(go.Scatter(x=mins,y=cfis,name=f"{dd['e']} {dn}",
            line=dict(color=dd["color"],width=4 if sel else 1.5,dash="solid" if sel else "dot"),
            opacity=1.0 if sel else 0.35))
    fig.add_vline(x=minute,line_dash="dash",line_color="gray")
    fig.add_trace(go.Scatter(x=[minute],y=[cfi],mode="markers",
        marker=dict(size=14,color=p["color"],line=dict(width=2,color="white")),showlegend=False))
    fig.add_hrect(y0=0,y1=30,fillcolor="#E8F5EE",opacity=0.25,line_width=0,annotation_text="Alert",annotation_position="top left")
    fig.add_hrect(y0=30,y1=60,fillcolor="#FEF9E7",opacity=0.25,line_width=0,annotation_text="Moderate",annotation_position="top left")
    fig.add_hrect(y0=60,y1=100,fillcolor="#FBEAEA",opacity=0.25,line_width=0,annotation_text="Fatigued",annotation_position="top left")
    fig.update_layout(xaxis_title="Minutes",yaxis_title="CFI",yaxis=dict(range=[0,105]),height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    recs = {"Vata":"Vata fatigues **early (~25 min)**. Error rate spikes sharply. **Break every 25 min.** Try 2-min Pranayama (box breathing 4-4-4-4).",
            "Pitta":"Pitta fatigues **moderately (~45 min)**. Response time increases. **10-min break at 45 min.** Screen-off, eyes closed.",
            "Kapha":"Kapha fatigues **late (~60 min)**. IKI gradually lengthens. **Movement break at 60 min.** Walking or stretching — physical activation works better than breathing."}
    if cfi>50: st.warning(recs[dosha])
    elif cfi>30: st.info(f"Moderate fatigue building. {recs[dosha]}")
    else: st.success(f"**{dosha} type — alert.** No intervention needed yet.")

# ═══ DATA EXPLORER ════════════════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.title("Dataset Explorer")
    if df is None: st.error("Run `python3 analysis/run_full.py` first."); st.stop()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total rows",len(df))
    c2.metric("Participants",df["participant_id"].nunique())
    rn = len(df[df.get("is_synthetic",True)==False]) if "is_synthetic" in df.columns else "N/A"
    sn = len(df[df.get("is_synthetic",True)==True]) if "is_synthetic" in df.columns else "N/A"
    c3.metric("Real rows",rn); c4.metric("Synthetic rows",sn)

    st.markdown("---")
    st.subheader("Understanding Fatigue Labels")
    st.markdown("""Each participant's session is split into **three equal time segments**:
| Label | When | What it means | Signal pattern |
|-------|------|--------------|---------------|
| 🟢 **Alert** | First third | Baseline — participant is fresh | Fast typing, low errors, steady mouse |
| 🟡 **Moderate** | Middle third | Transition — fatigue building | Some features starting to degrade |
| 🔴 **Fatigued** | Final third | Degraded — cognitive resources depleted | Slower, more errors, more pauses, more jitter |

**Why thirds?** Sessions vary in length. Adaptive thirds ensure every participant has balanced labels regardless of session duration.""")

    st.markdown("---")
    st.subheader("Feature distributions by fatigue label")
    feat = st.selectbox("Feature",["cpm","iki_mean_ms","iki_variance","error_rate","mouse_velocity_pxs","jitter_index_px","pause_frequency"])
    expl = {"cpm":"**Typing speed.** Drops with fatigue.","iki_mean_ms":"**Gap between keystrokes.** Increases with fatigue.",
            "iki_variance":"**Typing rhythm consistency.** Gets more irregular with fatigue.","error_rate":"**Backspace ratio.** More errors = more fatigue.",
            "mouse_velocity_pxs":"**Mouse speed.** Slows with fatigue.","jitter_index_px":"**Involuntary mouse tremor.** Increases with fatigue.",
            "pause_frequency":"**>2s gaps.** More pauses = more attention lapses."}
    st.markdown(expl.get(feat,""))
    fig = px.box(df,x="fatigue_label",y=feat,color="fatigue_label",
                 category_orders={"fatigue_label":["Alert","Moderate","Fatigued"]},
                 color_discrete_map={"Alert":"#2EAA70","Moderate":"#C48A1F","Fatigued":"#E05C5C"})
    fig.update_layout(height=400,showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    if "dominant_dosha" in df.columns:
        st.subheader("Same feature by Dosha — shows why personalization matters")
        fig2 = px.box(df,x="dominant_dosha",y=feat,color="dominant_dosha",
                      color_discrete_map={"V":"#4A90D9","P":"#E05C5C","K":"#2EAA70"})
        fig2.update_layout(height=400,showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df.head(100), use_container_width=True)

# ═══ EEG BENCHMARK ════════════════════════════════════════════════════
elif page == "🔬 EEG Benchmark":
    st.title("EEG Hardware Benchmark")
    st.markdown("Real EEG (₹50,000+ hardware) vs PrakritiSense (₹0)")
    if not eeg_res:
        st.warning("Run `python3 analysis/04_eeg_benchmark.py` first.")
        st.stop()
    hw = eeg_res.get("eeg_hardware",{})
    ps = eeg_res.get("prakritisense_prakriti",{})
    c1,c2,c3 = st.columns(3)
    c1.metric("EEG Accuracy (₹50,000+)",f"{hw.get('accuracy',0):.3f}")
    c2.metric("PrakritiSense (₹0)",f"{ps.get('accuracy',0):.3f}")
    if hw.get("accuracy",0)>0 and ps.get("accuracy",0)>0:
        c3.metric("% Retained",f"{ps['accuracy']/hw['accuracy']*100:.0f}%")
    img = find_img("eeg_benchmark_comparison.png")
    if img: st.image(img, use_container_width=True)
    f = eeg_res.get("key_finding","")
    if f: st.success(f"📌 **{f}**")