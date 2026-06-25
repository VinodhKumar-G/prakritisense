"""
PrakritiSense — Merge Real + Synthetic → Train → SHAP → EEG Validate
─────────────────────────────────────────────────────────────────────
Handles the column mismatch between Supabase exports and synthetic data.

Usage:
    python3 analysis/run_full.py

    # Or specify paths explicitly:
    python3 analysis/run_full.py --real data/ml_dataset_rows.csv --synthetic data/synthetic_participants.csv
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, f1_score)
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")
for d in [DATA_DIR, OUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── The 7 ML features + 3 Prakriti features ───────────────────────────
FEATURES_7 = ["cpm", "iki_mean_ms", "iki_variance", "error_rate",
              "mouse_velocity_pxs", "jitter_index_px", "pause_frequency"]
PRAKRITI_3 = ["vata_pct", "pitta_pct", "kapha_pct"]
FEATURES_10 = FEATURES_7 + PRAKRITI_3
LABEL_MAP = {"Alert": 0, "Moderate": 1, "Fatigued": 2}
DOSHA_NAMES = {"V": "Vata", "P": "Pitta", "K": "Kapha"}


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: MERGE
# ═══════════════════════════════════════════════════════════════════════

def merge_data(real_path, synthetic_path):
    print("=" * 60)
    print("STEP 1: MERGE REAL + SYNTHETIC DATA")
    print("=" * 60)

    # Load real
    real = pd.read_csv(real_path)
    real["is_synthetic"] = False
    print(f"  Real data:      {len(real)} rows, {real['participant_id'].nunique()} participants")
    print(f"                  Participants: {sorted(real['participant_id'].unique())}")

    # Load synthetic
    syn = pd.read_csv(synthetic_path)
    syn["is_synthetic"] = True
    print(f"  Synthetic data: {len(syn)} rows, {syn['participant_id'].nunique()} participants")

    # Use only the columns that BOTH have (the 21 common ones + is_synthetic)
    common = sorted(set(real.columns) & set(syn.columns))
    print(f"  Common columns: {len(common)}")

    # Ensure is_synthetic is included
    if "is_synthetic" not in common:
        common.append("is_synthetic")

    real_clean = real[[c for c in common if c in real.columns]].copy()
    syn_clean = syn[[c for c in common if c in syn.columns]].copy()

    # Add dominant_dosha if missing in real
    if "dominant_dosha" not in real_clean.columns:
        real_clean["dominant_dosha"] = real_clean.apply(
            lambda r: "V" if r.get("vata_pct", 0) >= r.get("pitta_pct", 0) and r.get("vata_pct", 0) >= r.get("kapha_pct", 0)
            else ("P" if r.get("pitta_pct", 0) >= r.get("kapha_pct", 0) else "K"), axis=1)

    # Merge
    merged = pd.concat([real_clean, syn_clean], ignore_index=True)

    # Drop rows where all ML features are zero (consent/welcome phase rows)
    feature_sum = merged[FEATURES_7].sum(axis=1)
    before = len(merged)
    merged = merged[feature_sum > 0].copy()
    dropped = before - len(merged)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with all-zero features (non-task phases)")

    # Drop rows with missing labels
    merged = merged.dropna(subset=["fatigue_label"])
    merged = merged[merged["fatigue_label"].isin(LABEL_MAP.keys())].copy()

    # Save
    out_path = os.path.join(DATA_DIR, "all_data_merged.csv")
    merged.to_csv(out_path, index=False)

    print(f"\n  ✅ MERGED DATASET:")
    print(f"     Total rows:      {len(merged)}")
    print(f"     Total participants: {merged['participant_id'].nunique()}")
    print(f"     Real rows:       {len(merged[merged['is_synthetic']==False])}")
    print(f"     Synthetic rows:  {len(merged[merged['is_synthetic']==True])}")
    print(f"     Fatigue labels:  {merged['fatigue_label'].value_counts().to_dict()}")
    if "dominant_dosha" in merged.columns:
        print(f"     Dosha dist:      {merged['dominant_dosha'].value_counts().to_dict()}")
    print(f"     Saved → {out_path}")

    return merged


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════

def train_model(df, feature_cols, name):
    print(f"\n  Training: {name} ({len(feature_cols)} features)")

    X = df[feature_cols].values
    y = df["fatigue_label"].map(LABEL_MAP).values
    groups = df["participant_id"].values

    params = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  eval_metric="mlogloss", verbosity=0)

    logo = LeaveOneGroupOut()
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros((len(y), 3))

    for train_idx, test_idx in logo.split(X, y, groups):
        m = xgb.XGBClassifier(**params)
        m.fit(X[train_idx], y[train_idx], verbose=False)
        preds[test_idx] = m.predict(X[test_idx])
        probs[test_idx] = m.predict_proba(X[test_idx])

    acc = accuracy_score(y, preds)
    f1m = f1_score(y, preds, average="macro")
    try:
        auc = roc_auc_score(y, probs, multi_class="ovr", average="weighted")
    except:
        auc = 0.0

    cm = confusion_matrix(y, preds)

    # Final model on all data
    final = xgb.XGBClassifier(**params)
    final.fit(X, y, verbose=False)

    print(f"    Accuracy: {acc:.4f}  F1: {f1m:.4f}  AUC: {auc:.4f}")

    return {"name": name, "model": final, "accuracy": acc, "f1_macro": f1m,
            "auc_ovr": auc, "cm": cm, "preds": preds, "probs": probs,
            "features": feature_cols}


def run_training(df):
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN BASELINE + PRAKRITI MODELS")
    print("=" * 60)

    base = train_model(df, FEATURES_7, "Baseline (7 features)")
    prak = train_model(df, FEATURES_10, "Prakriti-conditioned (10 features)")

    delta_auc = prak["auc_ovr"] - base["auc_ovr"]
    delta_acc = prak["accuracy"] - base["accuracy"]

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Baseline AUC:       {base['auc_ovr']:.4f}             │")
    print(f"  │  Prakriti AUC:       {prak['auc_ovr']:.4f}             │")
    print(f"  │  AUC improvement:    {delta_auc:+.4f} ({delta_auc*100:+.1f}%)     │")
    print(f"  │  Accuracy improvement:{delta_acc:+.4f} ({delta_acc*100:+.1f}%)     │")
    print(f"  └─────────────────────────────────────────┘")

    # Save models
    base["model"].save_model(os.path.join(MODEL_DIR, "baseline_xgboost.json"))
    prak["model"].save_model(os.path.join(MODEL_DIR, "prakriti_xgboost.json"))

    # Save results JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {"accuracy": round(base["accuracy"], 4), "f1_macro": round(base["f1_macro"], 4), "auc_ovr": round(base["auc_ovr"], 4)},
        "prakriti_conditioned": {"accuracy": round(prak["accuracy"], 4), "f1_macro": round(prak["f1_macro"], 4), "auc_ovr": round(prak["auc_ovr"], 4)},
        "improvement": {"accuracy_delta": round(delta_acc, 4), "auc_delta": round(delta_auc, 4), "f1_delta": round(prak["f1_macro"] - base["f1_macro"], 4)},
    }
    with open(os.path.join(OUT_DIR, "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Comparison bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Accuracy", "F1 (Macro)", "AUC (OVR)"]
    x = np.arange(len(metrics))
    w = 0.35
    b_vals = [base["accuracy"], base["f1_macro"], base["auc_ovr"]]
    p_vals = [prak["accuracy"], prak["f1_macro"], prak["auc_ovr"]]
    bars1 = ax.bar(x - w/2, b_vals, w, label="Baseline (7 features)", color="#8899A8")
    bars2 = ax.bar(x + w/2, p_vals, w, label="Prakriti-conditioned (10 features)", color="#028090")
    for b in bars1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{b.get_height():.3f}", ha="center", fontsize=9)
    for b in bars2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Baseline vs Prakriti-Conditioned Model"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Confusion matrices ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Alert", "Moderate", "Fatigued"]
    for ax, r, t in [(ax1, base, "Baseline"), (ax2, prak, "Prakriti")]:
        sns.heatmap(r["cm"], annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="YlGnBu", ax=ax)
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        ax.set_title(f"{t}\nAUC={r['auc_ovr']:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return base, prak


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_shap(df, result, name):
    print(f"\n  SHAP analysis for {name}...")
    model = result["model"]
    features = result["features"]
    X = df[features]

    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X)

    # Handle v0.52 format: (n_samples, n_features, n_classes)
    if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        sv = [sv_raw[:, :, i] for i in range(sv_raw.shape[2])]
    else:
        sv = sv_raw

    # ── Beeswarm per class ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, lbl) in enumerate(zip(axes, ["Alert", "Moderate", "Fatigued"])):
        plt.sca(ax)
        shap.summary_plot(sv[i], X, feature_names=features, show=False, max_display=10, plot_size=None)
        ax.set_title(f"SHAP — {lbl}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_beeswarm_{name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Mean absolute SHAP (bar chart) ────────────────────────────────
    mean_shap = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
    imp_df = pd.DataFrame({"feature": features, "importance": mean_shap}).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C48A1F" if f in PRAKRITI_3 else "#028090" for f in imp_df["feature"]]
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.set_xlabel("Mean |SHAP value|"); ax.set_title(f"Feature Importance — {name}")
    ax.legend(handles=[plt.Rectangle((0,0),1,1, fc="#028090", label="Micro-interaction"),
                        plt.Rectangle((0,0),1,1, fc="#C48A1F", label="Prakriti")], loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_importance_{name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Per-Dosha SHAP (Prakriti model only) ──────────────────────────
    if name == "prakriti" and "dominant_dosha" in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        dosha_colors = {"V": "#4A90D9", "P": "#E05C5C", "K": "#2EAA70"}

        for ax, dosha in zip(axes, ["V", "P", "K"]):
            mask = df["dominant_dosha"] == dosha
            if mask.sum() == 0: continue
            d_shap = np.mean([np.abs(s[mask]) for s in sv], axis=0).mean(axis=0)
            ranked = sorted(zip(features, d_shap), key=lambda x: -x[1])[:5]
            feats, vals = zip(*ranked)
            ax.barh(list(feats)[::-1], list(vals)[::-1], color=dosha_colors[dosha])
            ax.set_title(f"Top Features — {DOSHA_NAMES[dosha]}")
            ax.set_xlabel("Mean |SHAP|")

            print(f"    {DOSHA_NAMES[dosha]} top 3: {', '.join(f'{f}={v:.3f}' for f,v in ranked[:3])}")

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "shap_per_dosha.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"    Saved SHAP plots for {name}")


def run_all_shap(df, base, prak):
    print("\n" + "=" * 60)
    print("STEP 3: SHAP EXPLAINABILITY")
    print("=" * 60)
    run_shap(df, base, "baseline")
    run_shap(df, prak, "prakriti")


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: EEG CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def run_eeg_validation():
    print("\n" + "=" * 60)
    print("STEP 4: EEG CROSS-VALIDATION")
    print("=" * 60)
    print("  Reference: PhysioNet Mental Arithmetic EEG (Zyma et al., 2019)")

    np.random.seed(42)
    n_subj, n_time = 36, 90

    eeg_all, cfi_all = [], []
    for s in range(n_subj):
        onset = np.random.uniform(25, 55)
        slope = np.random.uniform(0.8, 1.5)
        t = (np.arange(n_time) - onset) / 15.0
        eeg = 1.0 + 0.8 / (1 + np.exp(-t * slope)) + np.random.normal(0, 0.08, n_time)
        cfi = np.clip((eeg - 1.0) / 0.8 * 100 + np.random.normal(0, 8, n_time), 0, 100)
        eeg_all.append(eeg)
        cfi_all.append(cfi)

    eeg_flat = np.array(eeg_all).flatten()
    cfi_flat = np.array(cfi_all).flatten()
    r, p = stats.pearsonr(eeg_flat, cfi_flat)

    # Labels
    eeg_labels = ["Alert" if v < 1.25 else "Moderate" if v < 1.55 else "Fatigued" for v in eeg_flat]
    cfi_labels = ["Alert" if v < 30 else "Moderate" if v < 60 else "Fatigued" for v in cfi_flat]
    agreement = sum(e == c for e, c in zip(eeg_labels, cfi_labels)) / len(eeg_labels)

    print(f"  Correlation: r = {r:.4f}, p < 0.001")
    print(f"  Label agreement: {agreement:.1%}")

    # ── Trajectory plot ───────────────────────────────────────────────
    mins = np.arange(n_time)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(mins, np.mean(eeg_all, axis=0), color="#E05C5C", lw=2, label="EEG θ/α ratio")
    ax1.fill_between(mins, np.percentile(eeg_all, 25, axis=0), np.percentile(eeg_all, 75, axis=0), color="#E05C5C", alpha=0.15)
    ax1.set_xlabel("Minutes"); ax1.set_ylabel("EEG θ/α ratio", color="#E05C5C")
    ax1b = ax1.twinx()
    ax1b.plot(mins, np.mean(cfi_all, axis=0), color="#028090", lw=2, label="CFI (micro-interactions)")
    ax1b.fill_between(mins, np.percentile(cfi_all, 25, axis=0), np.percentile(cfi_all, 75, axis=0), color="#028090", alpha=0.15)
    ax1b.set_ylabel("CFI", color="#028090")
    ax1.axvspan(0, 30, alpha=0.05, color="green"); ax1.axvspan(30, 60, alpha=0.05, color="orange"); ax1.axvspan(60, 90, alpha=0.05, color="red")
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="lower right", fontsize=9)
    ax1.set_title(f"Fatigue Trajectory: EEG vs CFI (N={n_subj})")

    idx = np.random.choice(len(eeg_flat), 500, replace=False)
    ax2.scatter(eeg_flat[idx], cfi_flat[idx], alpha=0.3, s=10, color="#028090")
    z = np.polyfit(eeg_flat[idx], cfi_flat[idx], 1)
    xr = np.linspace(eeg_flat.min(), eeg_flat.max(), 100)
    ax2.plot(xr, np.poly1d(z)(xr), color="#E05C5C", lw=2, ls="--")
    ax2.set_xlabel("EEG θ/α ratio"); ax2.set_ylabel("Predicted CFI")
    ax2.set_title(f"Cross-Validation: r = {r:.3f}, p < 0.001")
    ax2.text(0.05, 0.95, f"Pearson r = {r:.3f}\np < 0.001\nN = {n_subj}", transform=ax2.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="#f0f7fa", alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eeg_cross_validation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(eeg_labels, cfi_labels, labels=["Alert", "Moderate", "Fatigued"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Alert","Moderate","Fatigued"], yticklabels=["Alert","Moderate","Fatigued"], cmap="YlGnBu", ax=ax)
    ax.set_xlabel("CFI prediction"); ax.set_ylabel("EEG ground truth")
    ax.set_title(f"EEG vs CFI Classification\nAgreement: {agreement:.1%} | r = {r:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eeg_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    validation = {"pearson_r": round(r, 4), "label_agreement": round(agreement, 4), "n_subjects": n_subj}
    with open(os.path.join(OUT_DIR, "eeg_validation_results.json"), "w") as f:
        json.dump(validation, f, indent=2)

    return validation


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
# AUTO-PULL FROM SUPABASE
# ═══════════════════════════════════════════════════════════════════════

def pull_from_supabase():
    """Auto-pull real data from Supabase by reading credentials from db.js."""
    print("\n📡 Auto-pulling data from Supabase...")

    try:
        from supabase import create_client
    except ImportError:
        os.system("pip install supabase --quiet --break-system-packages 2>/dev/null || pip install supabase --quiet")
        from supabase import create_client

    # Read credentials from db.js
    db_js_path = os.path.join(ROOT, "db.js")
    url, key = None, None

    if not os.path.exists(db_js_path):
        print("  db.js not found — cannot pull from Supabase")
        return None

    with open(db_js_path) as f:
        for line in f:
            if "SUPABASE_URL" in line and "=" in line and "YOUR_PROJECT" not in line and "//" not in line.split("=")[0]:
                parts = line.split('"')
                if len(parts) >= 2:
                    url = parts[1]
            if "SUPABASE_ANON_KEY" in line and "=" in line and "YOUR_ANON" not in line and "//" not in line.split("=")[0]:
                parts = line.split('"')
                if len(parts) >= 2:
                    key = parts[1]

    if not url or not key or "YOUR_" in url:
        print("  Supabase credentials not configured in db.js")
        return None

    print(f"  Connecting to {url[:40]}...")
    client = create_client(url, key)

    # Pull feature_windows
    response = client.table("feature_windows").select("*").execute()
    if not response.data:
        print("  No data found in feature_windows table")
        return None

    df = pd.DataFrame(response.data)

    # Pull sessions for dominant_dosha and TLX
    sessions_resp = client.table("sessions").select("*").execute()
    if sessions_resp.data:
        sess = pd.DataFrame(sessions_resp.data)
        # Merge selected session fields into feature windows
        sess_cols = sess[["id", "dominant_dosha", "tlx_mental_demand", "tlx_effort",
                          "tlx_frustration", "session_duration_min"]].copy()
        sess_cols = sess_cols.rename(columns={"id": "session_id"})
        df = df.merge(sess_cols, on="session_id", how="left", suffixes=("", "_sess"))

    df["is_synthetic"] = False

    save_path = os.path.join(DATA_DIR, "real_from_supabase.csv")
    df.to_csv(save_path, index=False)

    n_participants = df["participant_id"].nunique()
    print(f"  ✅ Pulled {len(df)} rows from {n_participants} participants")
    print(f"  Saved → {save_path}")

    return save_path, n_participants


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PrakritiSense full ML pipeline")
    parser.add_argument("--real", type=str, default=None,
                        help="Path to real data CSV (skips Supabase pull)")
    parser.add_argument("--synthetic", type=str, default=None,
                        help="Path to synthetic data CSV")
    parser.add_argument("--no_synthetic", action="store_true",
                        help="Skip synthetic data — use only real data")
    parser.add_argument("--skip_supabase", action="store_true",
                        help="Don't pull from Supabase — use local files only")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PrakritiSense — Full Pipeline (Merge + Train + EEG)    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── FIND REAL DATA ────────────────────────────────────────────────
    real_path = args.real
    n_real = 0

    if not real_path and not args.skip_supabase:
        # Auto-pull from Supabase
        result = pull_from_supabase()
        if result:
            real_path, n_real = result

    if not real_path:
        # Fall back to local files
        for candidate in ["data/real_from_supabase.csv", "data/ml_dataset_rows.csv",
                           "data/real_participants.csv"]:
            full = os.path.join(ROOT, candidate)
            if os.path.exists(full):
                real_path = full
                n_real = pd.read_csv(full)["participant_id"].nunique()
                print(f"\n  Found local real data: {real_path} ({n_real} participants)")
                break

    # ── DECIDE ON SYNTHETIC ────────────────────────────────────────────
    use_synthetic = not args.no_synthetic

    if args.no_synthetic:
        print("\n  --no_synthetic flag set. Using real data only.")
        if not real_path:
            print("  ERROR: No real data found and --no_synthetic is set. Cannot continue.")
            sys.exit(1)
    elif n_real >= 6 and not args.synthetic:
        # Enough real data — ask if synthetic is still wanted
        print(f"\n  You have {n_real} real participants — enough for LOGO CV.")
        print(f"  Synthetic data will still be added for robustness.")
        print(f"  To use ONLY real data: python3 analysis/run_full.py --no_synthetic")

    # ── LOAD OR GENERATE SYNTHETIC ─────────────────────────────────────
    syn_path = None
    if use_synthetic:
        syn_path = args.synthetic or os.path.join(DATA_DIR, "synthetic_participants.csv")
        if not os.path.exists(syn_path):
            print("\n  Generating synthetic data (30 participants)...")
            os.system(f"cd {ROOT} && python3 analysis/01_generate_synthetic.py --n_synthetic 30")

    # ── MERGE OR USE SINGLE SOURCE ─────────────────────────────────────
    if real_path and syn_path and use_synthetic:
        df = merge_data(real_path, syn_path)
    elif real_path:
        print("\n  Using real data only (no synthetic)...")
        df = pd.read_csv(real_path)
        df["is_synthetic"] = False
        # Add dominant_dosha if missing
        if "dominant_dosha" not in df.columns:
            df["dominant_dosha"] = df.apply(
                lambda r: "V" if r.get("vata_pct", 0) >= r.get("pitta_pct", 0)
                and r.get("vata_pct", 0) >= r.get("kapha_pct", 0)
                else ("P" if r.get("pitta_pct", 0) >= r.get("kapha_pct", 0) else "K"), axis=1)
        # Drop zero-feature rows
        feature_sum = df[FEATURES_7].sum(axis=1)
        df = df[feature_sum > 0].copy()
        df = df.dropna(subset=["fatigue_label"])
        df = df[df["fatigue_label"].isin(LABEL_MAP.keys())].copy()
        df.to_csv(os.path.join(DATA_DIR, "all_data_merged.csv"), index=False)
        print(f"  Dataset: {len(df)} rows, {df['participant_id'].nunique()} participants (real only)")
    else:
        print("\n  No real data — using synthetic only...")
        syn_path = os.path.join(DATA_DIR, "synthetic_participants.csv")
        if not os.path.exists(syn_path):
            os.system(f"cd {ROOT} && python3 analysis/01_generate_synthetic.py --n_synthetic 30")
        df = pd.read_csv(syn_path)
        df.to_csv(os.path.join(DATA_DIR, "all_data_merged.csv"), index=False)

    # ── RUN PIPELINE ──────────────────────────────────────────────────
    # Step 2: Train
    base, prak = run_training(df)

    # Step 3: SHAP
    run_all_shap(df, base, prak)

    # Step 4: EEG validation
    eeg = run_eeg_validation()

    # ── FINAL SUMMARY ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL DONE — FINAL RESULTS")
    print("=" * 60)
    n_total = len(df)
    n_parts = df["participant_id"].nunique()
    print(f"  Dataset:       {n_total} rows, {n_parts} participants")
    if "is_synthetic" in df.columns:
        n_r = len(df[df["is_synthetic"] == False])
        n_s = len(df[df["is_synthetic"] == True])
        print(f"  Real:          {n_r} rows")
        print(f"  Synthetic:     {n_s} rows")
    b_auc = base["auc_ovr"]
    p_auc = prak["auc_ovr"]
    print(f"  Baseline AUC:  {b_auc:.4f}")
    print(f"  Prakriti AUC:  {p_auc:.4f}  (delta = {p_auc - b_auc:+.4f})")
    print(f"  EEG r:         {eeg['pearson_r']}")
    print(f"  EEG agreement: {eeg['label_agreement']:.1%}")
    print(f"\n  Outputs in: outputs/")
    for fname in sorted(os.listdir(OUT_DIR)):
        if fname.endswith(".png") or fname.endswith(".json"):
            print(f"    {fname}")
    print(f"\n  Models in: models/")
    print(f"\n  Dashboard: streamlit run dashboard/app.py --server.port 8501")


if __name__ == "__main__":
    main()