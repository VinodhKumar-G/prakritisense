"""
PrakritiSense — Real EEG Benchmark Comparison
──────────────────────────────────────────────
Downloads the PhysioNet Mental Arithmetic EEG dataset (Zyma et al., 2019),
extracts EEG band power features, trains XGBoost on REAL EEG data,
and compares accuracy against PrakritiSense (no hardware).

This answers the mentor's question:
  "What accuracy does EEG achieve WITH hardware, and what do you achieve WITHOUT?"

Dataset: https://physionet.org/content/eegmat/1.0.0/
  - 36 healthy subjects
  - EEG during rest (baseline) and mental arithmetic (cognitive load)
  - 19 channels, 500 Hz, EDF format
  - License: Open Data Commons PDDL (fully free)

Usage:
    python3 analysis/04_eeg_benchmark.py

Output:
    outputs/eeg_benchmark_comparison.png    — side-by-side accuracy comparison
    outputs/eeg_benchmark_results.json      — all metrics
"""

import os, sys, json, glob, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data", "eegmat")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD THE DATASET
# ═══════════════════════════════════════════════════════════════════════

def download_eeg_dataset():
    """Download PhysioNet EEGMAT dataset."""
    print("📥 Step 1: Downloading PhysioNet Mental Arithmetic EEG dataset...")

    check_file = os.path.join(DATA_DIR, "subject-info.csv")
    if os.path.exists(check_file):
        print("  Already downloaded. Skipping.")
        return True

    # Use wget to download the full dataset
    url = "https://physionet.org/static/published-projects/eegmat/eeg-during-mental-arithmetic-tasks-1.0.0.zip"
    zip_path = os.path.join(DATA_DIR, "eegmat.zip")

    print(f"  Downloading from PhysioNet (~25 MB)...")
    ret = os.system(f'wget -q --show-progress -O "{zip_path}" "{url}" 2>&1')

    if ret != 0:
        # Try with curl if wget fails
        print("  wget failed, trying curl...")
        ret = os.system(f'curl -L -o "{zip_path}" "{url}" 2>&1')

    if ret != 0 or not os.path.exists(zip_path):
        print("  ❌ Download failed. Please download manually from:")
        print(f"     {url}")
        print(f"     Extract to: {DATA_DIR}")
        return False

    # Extract
    print("  Extracting...")
    os.system(f'unzip -q -o "{zip_path}" -d "{DATA_DIR}"')

    # The zip extracts to a subdirectory — move files up if needed
    sub = os.path.join(DATA_DIR, "eeg-during-mental-arithmetic-tasks-1.0.0")
    if os.path.exists(sub):
        os.system(f'mv "{sub}"/* "{DATA_DIR}"/ 2>/dev/null')
        os.system(f'rmdir "{sub}" 2>/dev/null')

    os.remove(zip_path) if os.path.exists(zip_path) else None

    if os.path.exists(check_file):
        print("  ✅ Downloaded and extracted successfully")
        return True
    else:
        # Check alternative paths
        edfs = glob.glob(os.path.join(DATA_DIR, "**/*.edf"), recursive=True)
        if edfs:
            print(f"  ✅ Found {len(edfs)} EDF files")
            return True
        print("  ❌ Extraction may have failed. Check data/eegmat/ directory.")
        return False


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: EXTRACT EEG FEATURES
# ═══════════════════════════════════════════════════════════════════════

def extract_eeg_features():
    """Extract band power features from EDF files."""
    print("\n🧠 Step 2: Extracting EEG band power features...")

    try:
        import pyedflib
    except ImportError:
        print("  Installing pyedflib...")
        os.system("pip install pyedflib --quiet --break-system-packages 2>/dev/null || pip install pyedflib --quiet")
        import pyedflib

    from scipy.signal import welch

    # Find all EDF files
    edf_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.edf"), recursive=True))
    if not edf_files:
        print(f"  No EDF files found in {DATA_DIR}")
        return None

    print(f"  Found {len(edf_files)} EDF files")

    # EEG frequency bands
    BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    all_rows = []

    for edf_path in edf_files:
        filename = os.path.basename(edf_path)
        # Parse subject and condition from filename
        # Format: Subject00_1.edf (before) or Subject00_2.edf (during task)
        parts = filename.replace(".edf", "").split("_")
        if len(parts) < 2:
            continue

        subject = parts[0]
        condition_code = parts[1]

        # 1 = before (rest/baseline), 2 = during task (cognitive load)
        if condition_code == "1":
            condition = "Rest"
        elif condition_code == "2":
            condition = "CognitiveLoad"
        else:
            continue

        try:
            f = pyedflib.EdfReader(edf_path)
            n_channels = f.signals_in_file
            fs = int(f.getSampleFrequency(0))

            # Read all channels
            signals = []
            for ch in range(min(n_channels, 19)):  # Max 19 channels
                signals.append(f.readSignal(ch))
            f.close()

            if not signals:
                continue

            # Segment into 5-second windows
            window_samples = 5 * fs
            n_windows = min(len(signals[0]) // window_samples, 20)  # Max 20 windows per recording

            for w in range(n_windows):
                start = w * window_samples
                end = start + window_samples

                row = {"subject": subject, "condition": condition, "window": w}

                # Compute band powers for each channel, then average across channels
                for band_name, (low, high) in BANDS.items():
                    powers = []
                    for ch_signal in signals:
                        segment = ch_signal[start:end]
                        if len(segment) < window_samples:
                            continue
                        freqs, psd = welch(segment, fs=fs, nperseg=min(256, len(segment)))
                        band_mask = (freqs >= low) & (freqs <= high)
                        if band_mask.sum() > 0:
                            powers.append(np.mean(psd[band_mask]))

                    row[f"band_{band_name}"] = np.mean(powers) if powers else 0

                # Theta/Alpha ratio (key fatigue biomarker)
                if row["band_alpha"] > 0:
                    row["theta_alpha_ratio"] = row["band_theta"] / row["band_alpha"]
                else:
                    row["theta_alpha_ratio"] = 0

                # Beta/Alpha ratio (arousal indicator)
                if row["band_alpha"] > 0:
                    row["beta_alpha_ratio"] = row["band_beta"] / row["band_alpha"]
                else:
                    row["beta_alpha_ratio"] = 0

                all_rows.append(row)

        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

    if not all_rows:
        print("  No features extracted!")
        return None

    df = pd.DataFrame(all_rows)

    # Map conditions to fatigue labels
    # Rest = Alert (no cognitive load), CognitiveLoad = Fatigued (under mental strain)
    df["fatigue_label"] = df["condition"].map({"Rest": "Alert", "CognitiveLoad": "Fatigued"})

    print(f"  ✅ Extracted {len(df)} feature windows from {df['subject'].nunique()} subjects")
    print(f"     Conditions: {df['condition'].value_counts().to_dict()}")

    # Save
    save_path = os.path.join(DATA_DIR, "eeg_features.csv")
    df.to_csv(save_path, index=False)
    print(f"     Saved → {save_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN EEG MODEL
# ═══════════════════════════════════════════════════════════════════════

def train_eeg_model(df):
    """Train XGBoost on EEG features — this is the hardware benchmark."""
    print("\n🤖 Step 3: Training EEG-based fatigue model (hardware benchmark)...")

    feature_cols = ["band_delta", "band_theta", "band_alpha", "band_beta",
                    "band_gamma", "theta_alpha_ratio", "beta_alpha_ratio"]

    X = df[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["fatigue_label"].values)
    groups = df["subject"].values
    class_names = list(le.classes_)
    n_classes = len(class_names)

    print(f"  Features: {feature_cols}")
    print(f"  Classes: {class_names}")
    print(f"  Samples: {len(y)}, Subjects: {len(np.unique(groups))}")

    params = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  eval_metric="mlogloss" if n_classes > 2 else "logloss",
                  verbosity=0)
    if n_classes > 2:
        params["num_class"] = n_classes
        params["objective"] = "multi:softprob"

    # LOGO CV
    logo = LeaveOneGroupOut()
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros((len(y), n_classes)) if n_classes > 2 else np.zeros(len(y))

    for train_idx, test_idx in logo.split(X, y, groups):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < n_classes:
            from collections import Counter
            most_common = Counter(y_train).most_common(1)[0][0]
            preds[test_idx] = most_common
            if n_classes > 2:
                probs[test_idx, most_common] = 1.0
            else:
                probs[test_idx] = 0.5
            continue

        m = xgb.XGBClassifier(**params)
        m.fit(X[train_idx], y_train, verbose=False)
        preds[test_idx] = m.predict(X[test_idx])
        if n_classes > 2:
            probs[test_idx] = m.predict_proba(X[test_idx])
        else:
            probs[test_idx] = m.predict_proba(X[test_idx])[:, 1]

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro" if n_classes > 2 else "binary", zero_division=0)
    try:
        if n_classes > 2:
            auc = roc_auc_score(y, probs, multi_class="ovr", average="weighted")
        else:
            auc = roc_auc_score(y, probs)
    except:
        auc = 0.0

    print(f"\n  EEG Model Results (WITH hardware):")
    print(f"  ─────────────────────────────────")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"  AUC:        {auc:.4f}")

    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc": round(auc, 4),
            "n_subjects": int(df["subject"].nunique()), "n_samples": len(y),
            "n_features": len(feature_cols), "features": feature_cols,
            "class_names": class_names, "hardware": "EEG (19 channels, 500Hz)"}


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: COMPARE WITH PRAKRITISENSE
# ═══════════════════════════════════════════════════════════════════════

def compare_results(eeg_results):
    """Compare EEG benchmark with PrakritiSense results."""
    print("\n📊 Step 4: Comparing EEG (hardware) vs PrakritiSense (no hardware)...")

    # Load PrakritiSense results
    ps_results_path = os.path.join(OUT_DIR, "model_results.json")
    if os.path.exists(ps_results_path):
        with open(ps_results_path) as f:
            ps = json.load(f)
        ps_baseline = ps.get("baseline", {})
        ps_prakriti = ps.get("prakriti_conditioned", {})
    else:
        print("  ⚠️ No PrakritiSense results found. Run run_full.py first.")
        ps_baseline = {"accuracy": 0, "f1_macro": 0, "auc_ovr": 0}
        ps_prakriti = {"accuracy": 0, "f1_macro": 0, "auc_ovr": 0}

    # ── Comparison figure ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Bar chart comparison
    models = ["EEG\n(19-channel hardware)", "PrakritiSense\nBaseline (7 features)", "PrakritiSense\nPrakriti (10 features)"]
    accs = [eeg_results["accuracy"], ps_baseline.get("accuracy", 0), ps_prakriti.get("accuracy", 0)]
    aucs = [eeg_results["auc"], ps_baseline.get("auc_ovr", 0), ps_prakriti.get("auc_ovr", 0)]
    colors = ["#E05C5C", "#8899A8", "#028090"]

    x = np.arange(len(models))
    w = 0.35
    bars1 = ax1.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.8)
    bars2 = ax1.bar(x + w/2, aucs, w, label="AUC", color=colors, alpha=0.5, hatch="//")

    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
    for b in bars2:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{b.get_height():.3f}", ha="center", fontsize=10)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score")
    ax1.set_title("Fatigue Detection: EEG Hardware vs PrakritiSense (No Hardware)")
    ax1.legend(loc="upper right")
    ax1.axhline(y=eeg_results["accuracy"], color="#E05C5C", linestyle="--", alpha=0.3, linewidth=1)

    # Right: Cost vs Accuracy scatter
    hardware_costs = [50000, 0, 0]  # EEG ~₹50,000, PrakritiSense = ₹0
    acc_values = accs

    ax2.scatter(hardware_costs[0], acc_values[0], s=200, color="#E05C5C", zorder=5,
                label=f"EEG (₹50,000+)", edgecolors="white", linewidth=2)
    ax2.scatter(hardware_costs[1], acc_values[1], s=200, color="#8899A8", zorder=5,
                label=f"PS Baseline (₹0)", edgecolors="white", linewidth=2)
    ax2.scatter(hardware_costs[2], acc_values[2], s=200, color="#028090", zorder=5,
                label=f"PS Prakriti (₹0)", edgecolors="white", linewidth=2)

    # Annotate
    ax2.annotate(f"EEG\nAcc={acc_values[0]:.3f}", (hardware_costs[0], acc_values[0]),
                 textcoords="offset points", xytext=(15, -15), fontsize=9, color="#E05C5C")
    ax2.annotate(f"PrakritiSense\nAcc={acc_values[2]:.3f}", (hardware_costs[2]+500, acc_values[2]),
                 textcoords="offset points", xytext=(15, 15), fontsize=9, color="#028090")

    # Highlight the cost gap
    if acc_values[2] > 0 and acc_values[0] > 0:
        pct_of_eeg = acc_values[2] / acc_values[0] * 100
        ax2.text(25000, max(acc_values) * 0.5,
                 f"PrakritiSense achieves\n{pct_of_eeg:.0f}% of EEG accuracy\nat ₹0 hardware cost",
                 fontsize=11, ha="center", fontweight="bold", color="#028090",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#E1F2F4", alpha=0.8))

    ax2.set_xlabel("Hardware Cost (₹)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Hardware Cost")
    ax2.legend(loc="lower right")
    ax2.set_xlim(-2000, 55000)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eeg_benchmark_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved comparison figure → {path}")

    # ── Save results ──────────────────────────────────────────────────
    comparison = {
        "eeg_hardware": {
            "accuracy": eeg_results["accuracy"],
            "f1": eeg_results["f1"],
            "auc": eeg_results["auc"],
            "hardware": eeg_results["hardware"],
            "cost_inr": "50,000+",
            "n_subjects": eeg_results["n_subjects"],
            "dataset": "PhysioNet EEGMAT (Zyma et al., 2019)"
        },
        "prakritisense_baseline": {
            "accuracy": ps_baseline.get("accuracy", 0),
            "f1": ps_baseline.get("f1_macro", 0),
            "auc": ps_baseline.get("auc_ovr", 0),
            "hardware": "None (browser JavaScript only)",
            "cost_inr": "0",
        },
        "prakritisense_prakriti": {
            "accuracy": ps_prakriti.get("accuracy", 0),
            "f1": ps_prakriti.get("f1_macro", 0),
            "auc": ps_prakriti.get("auc_ovr", 0),
            "hardware": "None (browser JavaScript only)",
            "cost_inr": "0",
        },
        "key_finding": f"PrakritiSense achieves {acc_values[2]/acc_values[0]*100:.0f}% of EEG accuracy at zero hardware cost." if acc_values[0] > 0 and acc_values[2] > 0 else "Comparison pending"
    }

    results_path = os.path.join(OUT_DIR, "eeg_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved results → {results_path}")

    # ── Print final comparison ────────────────────────────────────────
    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  EEG (19ch, ₹50,000+)   Acc: {eeg_results['accuracy']:.4f}  AUC: {eeg_results['auc']:.4f}  │")
    print(f"  │  PS Baseline (₹0)        Acc: {ps_baseline.get('accuracy',0):.4f}  AUC: {ps_baseline.get('auc_ovr',0):.4f}  │")
    print(f"  │  PS Prakriti (₹0)        Acc: {ps_prakriti.get('accuracy',0):.4f}  AUC: {ps_prakriti.get('auc_ovr',0):.4f}  │")
    print(f"  └────────────────────────────────────────────────────────┘")

    if acc_values[0] > 0 and acc_values[2] > 0:
        pct = acc_values[2] / acc_values[0] * 100
        print(f"\n  📌 KEY FINDING FOR IEEE PAPER:")
        print(f"     PrakritiSense achieves {pct:.0f}% of EEG-grade accuracy")
        print(f"     with ZERO hardware cost — only a standard web browser needed.")

    return comparison


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PrakritiSense — Real EEG Benchmark Comparison          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("  Dataset: PhysioNet EEGMAT (Zyma et al., 2019, N=36)")
    print("  License: Open Data Commons PDDL (free for research)")

    # Step 1: Download
    if not download_eeg_dataset():
        print("\n  Cannot proceed without EEG dataset. Exiting.")
        sys.exit(1)

    # Step 2: Extract features
    df = extract_eeg_features()
    if df is None:
        print("\n  Feature extraction failed. Exiting.")
        sys.exit(1)

    # Step 3: Train EEG model
    eeg_results = train_eeg_model(df)

    # Step 4: Compare
    comparison = compare_results(eeg_results)

    print("\n" + "=" * 60)
    print("  DONE — EEG benchmark complete")
    print("=" * 60)
    print(f"  Figures: outputs/eeg_benchmark_comparison.png")
    print(f"  Results: outputs/eeg_benchmark_results.json")


if __name__ == "__main__":
    main()