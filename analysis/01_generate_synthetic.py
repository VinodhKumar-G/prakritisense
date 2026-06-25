"""
PrakritiSense — Synthetic Data Generator
─────────────────────────────────────────
Generates realistic synthetic participants based on Dosha-specific
behavioral profiles grounded in Ayurvedic literature.

The synthetic data follows the same schema as real collected data
and is used ONLY to augment the 5 real participants for model training.

Usage:
    python3 analysis/01_generate_synthetic.py
    python3 analysis/01_generate_synthetic.py --n_synthetic 30
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# DOSHA-SPECIFIC BEHAVIORAL PROFILES
# ─────────────────────────────────────────────────────────────────────
# These are grounded in:
#   - Charaka Samhita Sharira Sthana 6.10 (cognitive-behavioral traits)
#   - CCRAS-PAS Manual 2023 (behavioral predictors)
#   - Acien et al. 2022 (keystroke fatigue ranges)
#   - Banholzer et al. 2021 (mouse movement ranges)
#
# Each profile defines the ALERT baseline and how fatigue changes it.
# ═══════════════════════════════════════════════════════════════════════

DOSHA_PROFILES = {
    "V": {  # VATA — fast, erratic, fatigues early
        "name": "Vata",
        "prakriti": {"vata_pct": (55, 80), "pitta_pct": (10, 25), "kapha_pct": (5, 20)},
        "alert": {
            "cpm": (280, 360),          # Fast typist
            "iki_mean_ms": (140, 200),   # Short intervals
            "iki_variance": (3000, 8000), # High variability (erratic)
            "error_rate": (0.06, 0.12),  # More errors naturally
            "mouse_velocity_pxs": (400, 650), # Fast mouse
            "jitter_index_px": (6, 12),  # More jitter naturally
            "pause_frequency": (0, 2),   # Few pauses when alert
        },
        "fatigue_delta": {  # How features CHANGE when fatigued
            "cpm": (-120, -60),          # Speed DROPS sharply
            "iki_mean_ms": (60, 140),    # Intervals INCREASE
            "iki_variance": (2000, 6000), # Even more erratic
            "error_rate": (0.06, 0.14),  # Errors SPIKE (key Vata signal)
            "mouse_velocity_pxs": (-200, -80),
            "jitter_index_px": (4, 10),  # Jitter increases
            "pause_frequency": (3, 7),   # Many pauses
        },
        "fatigue_onset_window": 6,  # Fatigue starts around window 6 (~30 min)
    },
    "P": {  # PITTA — precise, focused, moderate fatigue resistance
        "name": "Pitta",
        "prakriti": {"vata_pct": (10, 25), "pitta_pct": (55, 80), "kapha_pct": (5, 20)},
        "alert": {
            "cpm": (220, 300),
            "iki_mean_ms": (180, 250),
            "iki_variance": (1500, 4000),  # Consistent rhythm
            "error_rate": (0.02, 0.06),    # Low errors
            "mouse_velocity_pxs": (300, 500),
            "jitter_index_px": (3, 7),
            "pause_frequency": (0, 1),
        },
        "fatigue_delta": {
            "cpm": (-60, -30),             # Moderate speed drop
            "iki_mean_ms": (30, 80),
            "iki_variance": (1000, 3000),
            "error_rate": (0.02, 0.06),    # Small error increase
            "mouse_velocity_pxs": (-120, -40),
            "jitter_index_px": (2, 6),
            "pause_frequency": (1, 4),
        },
        "fatigue_onset_window": 10,  # Fatigue starts ~50 min
    },
    "K": {  # KAPHA — slow, steady, late fatigue
        "name": "Kapha",
        "prakriti": {"vata_pct": (5, 20), "pitta_pct": (10, 25), "kapha_pct": (55, 80)},
        "alert": {
            "cpm": (140, 220),           # Slow typist
            "iki_mean_ms": (250, 380),   # Long intervals
            "iki_variance": (800, 2500), # Very consistent
            "error_rate": (0.01, 0.04),  # Very few errors
            "mouse_velocity_pxs": (180, 350),
            "jitter_index_px": (2, 5),   # Minimal jitter
            "pause_frequency": (0, 1),
        },
        "fatigue_delta": {
            "cpm": (-40, -15),           # Gradual slowdown
            "iki_mean_ms": (40, 100),    # KEY Kapha signal: IKI lengthening
            "iki_variance": (500, 2000),
            "error_rate": (0.01, 0.03),  # Minimal error increase
            "mouse_velocity_pxs": (-80, -20),
            "jitter_index_px": (1, 4),
            "pause_frequency": (2, 5),   # Gradual pause increase
        },
        "fatigue_onset_window": 14,  # Fatigue starts ~70 min
    },
}

PHASES = ["typing-copy", "typing-free", "typing-free", "stroop", "mouse"]


def generate_one_participant(participant_id, dosha_key, n_windows=90):
    """Generate a full session for one synthetic participant."""
    profile = DOSHA_PROFILES[dosha_key]

    # Generate Prakriti percentages
    v = np.random.randint(*profile["prakriti"]["vata_pct"])
    p = np.random.randint(*profile["prakriti"]["pitta_pct"])
    k = 100 - v - p
    if k < 0:
        k = np.random.randint(5, 15)
        total = v + p + k
        v = int(v / total * 100)
        p = int(p / total * 100)
        k = 100 - v - p

    onset = profile["fatigue_onset_window"]
    rows = []

    for w in range(n_windows):
        # Fatigue progression: sigmoid curve starting at onset
        t = (w - onset) / 4.0
        fatigue_factor = 1.0 / (1.0 + np.exp(-t))  # 0→1 smooth

        # Phase based on window number
        if w < 15:
            phase = "typing-copy"
        elif w < 45:
            phase = "typing-free"
        elif w < 60:
            phase = "stroop"
        else:
            phase = "mouse"

        # Generate features: alert baseline + fatigue_factor * delta
        row = {"participant_id": participant_id, "phase": phase}

        for feat in ["cpm", "iki_mean_ms", "iki_variance", "error_rate",
                      "mouse_velocity_pxs", "jitter_index_px", "pause_frequency"]:
            base_lo, base_hi = profile["alert"][feat]
            # Individual baseline (consistent within a participant)
            np.random.seed(hash(participant_id + feat) % 2**31)
            personal_base = np.random.uniform(base_lo, base_hi)
            np.random.seed(None)

            delta_lo, delta_hi = profile["fatigue_delta"][feat]
            delta = np.random.uniform(delta_lo, delta_hi) * fatigue_factor

            # Add noise (natural variation ~5-10%)
            noise = np.random.normal(0, abs(personal_base) * 0.07)

            value = personal_base + delta + noise

            # Clamp
            if feat == "error_rate":
                value = np.clip(value, 0.0, 0.5)
            elif feat == "pause_frequency":
                value = max(0, int(round(value)))
            elif feat in ["cpm", "iki_mean_ms", "mouse_velocity_pxs"]:
                value = max(10, value)
            else:
                value = max(0, value)

            row[feat] = round(value, 4)

        # Metadata
        row["window_start_ms"] = w * 60000
        row["duration_sec"] = 60.0
        row["time_on_task_minutes"] = round(w * 1.0, 2)

        # Fatigue label (ground truth)
        if w < 30:
            row["fatigue_label"] = "Alert"
        elif w < 60:
            row["fatigue_label"] = "Moderate"
        else:
            row["fatigue_label"] = "Fatigued"

        # Prakriti scores
        row["vata_pct"] = v
        row["pitta_pct"] = p
        row["kapha_pct"] = k
        row["dominant_dosha"] = dosha_key

        # Raw counts (approximate, for realism)
        row["keystroke_count"] = int(row["cpm"] * 1.0)  # ~1 min window
        row["backspace_count"] = int(row["keystroke_count"] * row["error_rate"])
        row["mouse_move_count"] = np.random.randint(40, 120)
        row["click_count"] = np.random.randint(2, 15)
        row["is_synthetic"] = True

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_synthetic", type=int, default=30,
                        help="Number of synthetic participants to generate")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--real_data", type=str, default=None,
                        help="Path to real data CSV (to merge with)")
    args = parser.parse_args()

    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PrakritiSense Synthetic Data Generator")
    print("=" * 60)

    # Distribute doshas: roughly equal with some variation
    dosha_keys = ["V", "P", "K"]
    all_dfs = []

    for i in range(args.n_synthetic):
        dosha = dosha_keys[i % 3]  # Cycle through V, P, K
        pid = f"SYN-{dosha}-{i+1:03d}"

        # Vary session length slightly (70-90 windows)
        n_windows = np.random.randint(70, 91)

        df = generate_one_participant(pid, dosha, n_windows)
        all_dfs.append(df)
        print(f"  Generated {pid}: {len(df)} windows, "
              f"Dosha={DOSHA_PROFILES[dosha]['name']}, "
              f"V={df['vata_pct'].iloc[0]}% P={df['pitta_pct'].iloc[0]}% K={df['kapha_pct'].iloc[0]}%")

    synthetic_df = pd.concat(all_dfs, ignore_index=True)

    # Summary stats
    print(f"\nSynthetic dataset: {len(synthetic_df)} total rows")
    print(f"  Participants: {synthetic_df['participant_id'].nunique()}")
    print(f"  Dosha distribution:")
    print(f"    Vata:  {(synthetic_df.groupby('participant_id')['dominant_dosha'].first() == 'V').sum()}")
    print(f"    Pitta: {(synthetic_df.groupby('participant_id')['dominant_dosha'].first() == 'P').sum()}")
    print(f"    Kapha: {(synthetic_df.groupby('participant_id')['dominant_dosha'].first() == 'K').sum()}")
    print(f"\n  Feature means by fatigue label:")
    print(synthetic_df.groupby("fatigue_label")[
        ["cpm", "iki_mean_ms", "error_rate", "mouse_velocity_pxs", "jitter_index_px"]
    ].mean().round(2).to_string())

    # Save synthetic data
    syn_path = os.path.join(OUT_DIR, "synthetic_participants.csv")
    synthetic_df.to_csv(syn_path, index=False)
    print(f"\nSaved synthetic data → {syn_path}")

    # If real data provided, merge
    if args.real_data and os.path.exists(args.real_data):
        print(f"\nMerging with real data from {args.real_data}...")
        real_df = pd.read_csv(args.real_data)
        real_df["is_synthetic"] = False
        real_df["dominant_dosha"] = real_df.apply(
            lambda r: "V" if r.get("vata_pct", 0) >= r.get("pitta_pct", 0) and r.get("vata_pct", 0) >= r.get("kapha_pct", 0)
            else ("P" if r.get("pitta_pct", 0) >= r.get("kapha_pct", 0) else "K"), axis=1
        )
        merged = pd.concat([real_df, synthetic_df], ignore_index=True)
        merged_path = os.path.join(OUT_DIR, "all_data_merged.csv")
        merged.to_csv(merged_path, index=False)
        print(f"Merged dataset: {len(merged)} rows ({len(real_df)} real + {len(synthetic_df)} synthetic)")
        print(f"Saved → {merged_path}")
    else:
        # Just save synthetic as the training set
        merged_path = os.path.join(OUT_DIR, "all_data_merged.csv")
        synthetic_df.to_csv(merged_path, index=False)
        print(f"\nNo real data provided — using synthetic only for training.")
        print(f"Saved → {merged_path}")
        print(f"\nTo merge with real data later:")
        print(f"  python3 analysis/01_generate_synthetic.py --real_data data/all_participants.csv")


if __name__ == "__main__":
    main()