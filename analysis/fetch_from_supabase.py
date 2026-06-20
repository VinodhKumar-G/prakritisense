"""
PrakritiSense — Pull data from Supabase and save as ML-ready CSV
─────────────────────────────────────────────────────────────────
Usage:
    python3 analysis/fetch_from_supabase.py

Requires:
    pip install supabase pandas

Outputs:
    data/all_participants.csv  — full ML dataset, one row per 60s window
    data/sessions_summary.csv  — one row per participant session
"""

import os
import pandas as pd
from supabase import create_client

# ── CONFIGURE ─────────────────────────────────────────────────────────
SUPABASE_URL      = "https://gxodetmbxbqlzkpwyktm.supabase.co/rest/v1/"
SUPABASE_ANON_KEY = "sb_publishable_Sb-z0yjKTN8nEJE07Ky2UA_4Y41TvOu"
# ──────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_all():
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    # ── Feature windows (ML dataset) ──────────────────────────────────
    print("Fetching feature_windows...")
    response = client.table("feature_windows").select("*").execute()
    windows_df = pd.DataFrame(response.data)
    print(f"  {len(windows_df)} rows fetched")

    # ── Sessions ──────────────────────────────────────────────────────
    print("Fetching sessions...")
    response = client.table("sessions").select("*").execute()
    sessions_df = pd.DataFrame(response.data)
    print(f"  {len(sessions_df)} sessions fetched")

    # ── Merge (join windows with session metadata) ────────────────────
    if not windows_df.empty and not sessions_df.empty:
        sessions_slim = sessions_df[[
            "id", "dominant_dosha",
            "tlx_mental_demand", "tlx_effort", "tlx_frustration",
            "session_duration_min"
        ]].rename(columns={"id": "session_id"})

        merged = windows_df.merge(sessions_slim, on="session_id", how="left")
    else:
        merged = windows_df

    # ── Save ──────────────────────────────────────────────────────────
    ml_path  = os.path.join(OUT_DIR, "all_participants.csv")
    ses_path = os.path.join(OUT_DIR, "sessions_summary.csv")

    merged.to_csv(ml_path, index=False)
    sessions_df.to_csv(ses_path, index=False)

    print(f"\nSaved:")
    print(f"  ML dataset    → {ml_path}  ({len(merged)} rows, {len(merged.columns)} cols)")
    print(f"  Sessions      → {ses_path}  ({len(sessions_df)} sessions)")

    # ── Quick sanity report ───────────────────────────────────────────
    if not merged.empty:
        print(f"\nParticipants: {merged['participant_id'].nunique()}")
        print(f"Fatigue label distribution:\n{merged['fatigue_label'].value_counts().to_string()}")
        if "dominant_dosha" in merged.columns:
            print(f"\nDosha distribution:\n{merged['dominant_dosha'].value_counts().to_string()}")
        print(f"\nFeature means:\n{merged[['cpm','iki_mean_ms','error_rate','mouse_velocity_pxs','jitter_index_px']].mean().round(2).to_string()}")

    return merged, sessions_df


if __name__ == "__main__":
    if "YOUR_PROJECT_ID" in SUPABASE_URL:
        print("ERROR: Please update SUPABASE_URL and SUPABASE_ANON_KEY in this script.")
    else:
        fetch_all()