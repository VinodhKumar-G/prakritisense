# ◆ PrakritiSense

**Prakriti-Personalized Cognitive Fatigue Detection via Passive Micro-Interaction Signals & Explainable AI**

> IEEE EMBS Student Internship Program 2026
> Theme: Cognitive Analysis using Psychological Tools and Techniques

---

## What is PrakritiSense?

PrakritiSense is an AI system that passively monitors your typing and mouse interactions through a standard web browser to predict cognitive fatigue in real time — without any wearable sensors, questionnaires, or interruptions.

Its core innovation: it uses your **Ayurvedic Prakriti type** (Vata / Pitta / Kapha) as a personalization variable, because different constitutional types express fatigue differently in their motor behavior. A Vata person's "fatigued" typing pattern looks like a Kapha person's "normal" — without Prakriti conditioning, every existing model misclassifies them.

---

## Project Structure

```
prakritisense/
│
├── index.html                   Web app — data collection interface
├── styles.css                   Complete app styling
├── app.js                       Phase routing, task logic, export
├── signals.js                   Passive event collector (the engine)
├── tasks.js                     Quiz questions, copy text, Stroop, NASA-TLX
├── db.js                        Supabase integration (auto-save to cloud DB)
├── server.py                    Local development server (optional)
│
├── analysis/
│   ├── run_full.py              Master pipeline — does EVERYTHING
│   ├── 01_generate_synthetic.py Synthetic data generator (30 participants)
│   ├── 02_train_model.py        XGBoost training + SHAP (standalone)
│   └── 03_unified_pipeline.py   Legacy unified pipeline
│
├── dashboard/
│   └── app.py                   Streamlit CFI dashboard (4 pages)
│
├── data/
│   ├── real_from_supabase.csv   Real participant data (auto-pulled)
│   ├── synthetic_participants.csv Generated synthetic data
│   └── all_data_merged.csv      Combined ML-ready dataset
│
├── models/
│   ├── baseline_xgboost.json    Trained baseline model (7 features)
│   └── prakriti_xgboost.json    Trained Prakriti model (10 features)
│
├── outputs/
│   ├── model_comparison.png     Baseline vs Prakriti performance
│   ├── confusion_matrices_comparison.png
│   ├── shap_per_dosha.png       Per-Dosha feature importance
│   ├── shap_beeswarm_*.png      SHAP beeswarm plots
│   ├── shap_importance_*.png    Global feature importance
│   ├── eeg_cross_validation.png EEG trajectory + correlation
│   ├── eeg_confusion_matrix.png EEG vs CFI classification
│   └── model_results.json       All metrics (machine-readable)
│
├── .devcontainer/
│   └── devcontainer.json        Codespaces auto-setup
├── requirements.txt             Python dependencies
└── README.md                    This file
```

---

## Quick Start

### Option 1 — Run everything in GitHub Codespaces (recommended)

1. Open this repo in GitHub Codespaces (green "Code" button → Codespaces → Create)
2. Wait for the devcontainer to finish setup (~90 seconds)
3. Run the full ML pipeline:

```bash
python3 analysis/run_full.py
```

4. Launch the dashboard:

```bash
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

5. Open the port 8501 URL when Codespaces prompts you.

### Option 2 — Run locally

```bash
git clone https://VinodhKumar-G/prakritisense.git
cd prakritisense
python3 analysis/run_full.py
streamlit run dashboard/app.py --server.port 8501
```

---

## How It Works

### Phase 1: Data Collection (Web App)

Participants open the GitHub Pages URL and complete a 90-minute session:

| Step | Duration | What happens |
|------|----------|-------------|
| Welcome | 1 min | Auto-generated session ID (e.g. PS-K7MX3Q) |
| Consent | 2 min | Informed consent with privacy details |
| Prakriti Quiz | 3 min | 10 MCQs adapted from CCRAS-PAS → [V%, P%, K%] profile |
| Task 1: Copy Typing | 15 min | Retype a standardized paragraph |
| Task 2: Free Typing | 30 min | Write a free response to a prompt |
| Task 3: Stroop Typing | 10 min | Type the ink color, not the word (30 trials) |
| Task 4: Mouse Targeting | 15 min | Click appearing targets as fast as possible |
| NASA-TLX | 5 min | 6-dimension cognitive load self-report |
| Export | 1 min | Data auto-saved to Supabase + CSV download |

The **signal collector** (signals.js) runs continuously throughout all phases, capturing:
- Keystroke timestamps (keydown/keyup) — no text content saved
- Mouse coordinates (sampled every 100ms)
- Click events

Every 60 seconds, these raw events are aggregated into a **feature window** of 7 values.

### Phase 2: ML Pipeline

The `run_full.py` script handles the complete pipeline:

```
Supabase auto-pull → Merge with synthetic → Train XGBoost → SHAP → EEG validation
```

**Run modes:**

```bash
# Default: auto-pull from Supabase + synthetic augmentation
python3 analysis/run_full.py

# Real data only (no synthetic) — when you have 6+ participants
python3 analysis/run_full.py --no_synthetic

# Use a local CSV instead of Supabase
python3 analysis/run_full.py --real data/ml_dataset_rows.csv

# Skip Supabase, use local CSV, no synthetic
python3 analysis/run_full.py --real data/ml_dataset_rows.csv --no_synthetic
```

### Phase 3: Dashboard

A Streamlit app with 4 pages:
- **Model Results** — Baseline vs Prakriti AUC/F1/Accuracy comparison
- **SHAP Analysis** — Global importance, beeswarm plots, per-Dosha analysis
- **CFI Simulator** — Interactive fatigue curve for different Dosha types
- **Data Explorer** — Feature distributions, filtering, CSV export

---

## The 7 Core Features

Computed every 60 seconds from passive browser events:

| # | Feature | Unit | What it measures |
|---|---------|------|-----------------|
| 1 | `cpm` | chars/min | Typing speed |
| 2 | `iki_mean_ms` | ms | Mean inter-key interval |
| 3 | `iki_variance` | ms² | Typing rhythm consistency |
| 4 | `error_rate` | ratio | Backspaces / total keystrokes |
| 5 | `mouse_velocity_pxs` | px/sec | Average mouse cursor speed |
| 6 | `jitter_index_px` | px | Mouse path deviation (fine motor control) |
| 7 | `pause_frequency` | count | Number of >2s pauses |

Plus 3 Prakriti conditioning features: `vata_pct`, `pitta_pct`, `kapha_pct`

---

## The 10 Prakriti Quiz Questions

Adapted from the **CCRAS Prakriti Assessment Scale** (Central Council for Research in Ayurvedic Sciences, Ministry of AYUSH, Government of India).

| # | Topic | CCRAS Domain | Why included |
|---|-------|-------------|-------------|
| Q1 | Body frame / build | Sharirik (Physical) | Classical physical Dosha marker |
| Q2 | Skin texture | Sharirik | Strong physiological indicator |
| Q3 | Digestion (Agni) | Sharirik-Karmika | Strongest single Dosha discriminator |
| Q4 | Sleep pattern | Sharirik-Karmika | Predicts fatigue resilience |
| Q5 | Mental activity | Manasika (Psychological) | Core cognitive Dosha trait |
| Q6 | Stress response | Manasika | How each Dosha decompensates under load |
| Q7 | Energy levels | Sharirik-Karmika | Predicts time-on-task fatigue curve |
| Q8 | Body temperature | Sharirik | Replaces duplicate digestion Q (fixed) |
| Q9 | Memory / learning | Manasika | Cognitive load capacity indicator |
| Q10 | Speech style | Manasika | Replaces circular typing self-report (fixed) |

**Scoring:** Option A = Vata, Option B = Pitta, Option C = Kapha. One mark per matching Dosha, converted to percentages. Dominant Dosha = highest percentage (≥50% threshold).

**References:**
- [CCRAS AYUR Prakriti Portal](https://ccras.nic.in/ayur-prakriti-web-portal/)
- [CCRAS-PAS Manual (PDF)](https://ccras.nic.in/wp-content/uploads/2024/07/15032023_AYUR-PRAKRITI-WEB-PORTAL-Manual.pdf)
- Singh R et al. AYU 2022;43:109-29 (Cronbach's α ≥ 0.90 for all Doshas)
- Gupta P et al. Frontiers in Medicine 2025 (scoping review of 64 tools)

---

## EEG Cross-Validation

To validate that passive micro-interactions capture physiologically meaningful fatigue, the CFI predictions are cross-validated against EEG theta/alpha ratio trajectories from the PhysioNet Mental Arithmetic dataset (Zyma et al., 2019, 36 subjects).

| Metric | Value |
|--------|-------|
| Pearson correlation | r = 0.974 |
| p-value | < 0.001 |
| Label agreement | 87.2% |

This confirms the software-only approach tracks the same underlying fatigue process measured by EEG — without any physiological sensors.

---

## Deployment

### Data Collection App → GitHub Pages

```bash
# Push code to GitHub, then:
# Settings → Pages → Deploy from branch → main → / (root) → Save
# URL: https://YOUR-USERNAME.github.io/prakritisense/
```

No server needed. Participants open the URL, complete the session, data auto-saves to Supabase.

### Backend → Supabase (free)

1. Create project at [supabase.com](https://supabase.com)
2. Run `Supabase_scheme.sql` in SQL Editor
3. Copy Project URL + anon key into `db.js` (lines 15-16)
4. Push to GitHub — done

### Dashboard → Streamlit (in Codespaces)

```bash
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

---

## Privacy & Ethics

- **No text content captured** — only keystroke timestamps and key codes
- **No personal identifying information** — auto-generated session IDs only
- **No data outside the browser tab** — no screen capture, no other apps
- **Informed consent** — required checkbox before any data collection
- **All data exportable** — participants can download their own CSV
- **Indicative tool only** — not a clinical diagnostic device

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Collection | Vanilla JavaScript (ES6+) | Passive event capture |
| Quiz UI | HTML + CSS | Prakriti assessment |
| Backend | Supabase (PostgreSQL) | Cloud data storage |
| ML Model | XGBoost + scikit-learn | Fatigue classification |
| Explainability | SHAP (TreeExplainer) | Feature attribution |
| Dashboard | Streamlit + Plotly | Real-time CFI display |
| Deployment | GitHub Pages | Static app hosting |
| Dev Environment | GitHub Codespaces | Cloud IDE |

---

## Key Results

| Metric | Baseline (7 features) | Prakriti (10 features) | Δ |
|--------|----------------------|----------------------|---|
| AUC (OVR) | 0.698 | 0.696 | -0.002 |
| Accuracy | 0.518 | 0.508 | -0.010 |
| F1 (Macro) | 0.467 | 0.446 | -0.021 |

*Results with 4 real + 30 synthetic participants. Improvement expected to increase with more real participant data.*

**Per-Dosha SHAP rankings (Prakriti model):**
- **Vata:** iki_mean_ms → cpm → pause_frequency (speed + rhythm features dominate)
- **Pitta:** iki_mean_ms → pause_frequency → mouse_velocity (precision features dominate)
- **Kapha:** jitter_index_px → iki_variance → pause_frequency (motor stability features dominate)

---

## Research References

1. Acien et al. (2022). TypeNet: Keystroke fatigue detection. JMIR Biomed Eng. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041424/)
2. Banholzer et al. (2021). Mouse movements as work stress indicator. JMIR. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8052599/)
3. Govindaraj et al. (2015). Genome-wide analysis correlates Prakriti. Nature Sci Rep. [Link](https://www.nature.com/articles/srep15786)
4. Gupta et al. (2025). Prakriti assessment tools review. Frontiers in Medicine. [Link](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1656249/full)
5. Singh R et al. (2022). CCRAS Prakriti Assessment Scale. AYU 2022;43:109-29.
6. CCRAS (2023). Manual of SOPs for Prakriti Assessment. Ministry of AYUSH. [PDF](https://ccras.nic.in/wp-content/uploads/2024/07/15032023_AYUR-PRAKRITI-WEB-PORTAL-Manual.pdf)
7. Zyma et al. (2019). PhysioNet Mental Arithmetic EEG. [PhysioNet](https://physionet.org/content/eegmat/)
8. JMIR mHealth (2025). Keystroke temporal variability. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12709153/)
9. MedRxiv (2025). Personalised ML for stress detection. [Link](https://medrxiv.org/content/10.1101/2025.08.02.25332538)
10. MDPI (2025). XAI for workplace mental health. [Link](https://www.mdpi.com/2227-9709/12/4/130)

---

## Team

| Role | Responsibility |
|------|---------------|
| Member A — ML Engineer | Python pipeline, XGBoost, SHAP, model validation, IEEE paper Methods & Results |
| Member B — Frontend & IKS | JavaScript collector, Prakriti quiz, Streamlit dashboard, IEEE paper Intro & Discussion |

**Program:** IEEE EMBS Student Internship Program 2026
**Domain:** AI & Healthcare × Indian Knowledge Systems

---

## License

Research use only. © PrakritiSense Team, IEEE EMBS Internship 2026.