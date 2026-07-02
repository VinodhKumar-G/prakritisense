# ◆ PrakritiSense

**Prakriti-Personalized Cognitive Fatigue Detection via Passive Micro-Interaction Signals and Explainable AI**

> IEEE EMBS Student Internship Program 2026
> Theme: Cognitive Analysis using Psychological Tools and Techniques
> Domain: Artificial Intelligence × Healthcare × Indian Knowledge Systems


## What is PrakritiSense?

PrakritiSense is an AI system that passively monitors typing and mouse interactions through a standard web browser to predict cognitive fatigue in real time — without any wearable sensors, questionnaires, or interruptions to work.

Its core innovation: the **Ayurvedic Prakriti type** (Vata / Pitta / Kapha) is used as a personalization variable, because different constitutional types express cognitive fatigue through different behavioral channels. A Vata person's "fatigued" typing pattern looks like a Kapha person's "normal" — without Prakriti conditioning, every existing model misclassifies them.


## Project Structure


prakritisense/
│
├── index.html                     Web app — full data collection interface (9 phases)
├── styles.css                     Complete app styling (navy/teal/sage theme)
├── app.js                         Phase routing, task logic, Supabase export
├── signals.js                     Passive micro-interaction signal collector
├── tasks.js                       Quiz questions, copy text, Stroop, NASA-TLX content
├── db.js                          Supabase database integration (auto-save)
├── server.py                      Optional local development server
│
├── analysis/
│   ├── run_full.py                Master pipeline — runs everything in one command
│   ├── 01_generate_synthetic.py   Synthetic data generator (30 participants)
│   └── 04_eeg_benchmark.py        Real EEG benchmark (PhysioNet EEGMAT, 36 subjects)
│
├── dashboard/
│   └── app.py                     Streamlit CFI dashboard (5 pages)
│
├── data/
│   ├── real_from_supabase.csv     Real participant data (auto-pulled from Supabase)
│   ├── synthetic_participants.csv  30 synthetic participants (generated)
│   ├── all_data_merged.csv        Merged ML-ready dataset (2,475 rows, 34 participants)
│   └── eegmat/
│       ├── subject-info.csv       EEG subject metadata
│       ├── eeg_features.csv       Extracted EEG band power features
│       ├── Subject00_1.edf        EEG recordings — rest condition (baseline)
│       ├── Subject00_2.edf        EEG recordings — mental arithmetic (cognitive load)
│       └── ... (72 EDF files, 36 subjects × 2 conditions)
│
├── models/
│   ├── baseline_xgboost.json      Trained baseline model (7 features)
│   └── prakriti_xgboost.json      Trained Prakriti-conditioned model (10 features)
│
├── outputs/
│   ├── model_comparison.png       Baseline vs Prakriti performance bar chart
│   ├── confusion_matrices_comparison.png  Side-by-side confusion matrices
│   ├── shap_importance_baseline.png       Global SHAP importance — baseline model
│   ├── shap_importance_prakriti.png       Global SHAP importance — Prakriti model
│   ├── shap_beeswarm_baseline.png         SHAP beeswarm — baseline model
│   ├── shap_beeswarm_prakriti.png         SHAP beeswarm — Prakriti model
│   ├── shap_per_dosha.png                 Per-Dosha top features (key novel figure)
│   ├── eeg_cross_validation.png           EEG trajectory vs CFI correlation
│   ├── eeg_confusion_matrix.png           EEG vs CFI label agreement matrix
│   ├── eeg_benchmark_comparison.png       EEG hardware vs PrakritiSense comparison
│   ├── model_results.json                 Model metrics (machine-readable)
│   ├── eeg_validation_results.json        EEG trajectory correlation stats
│   └── eeg_benchmark_results.json         EEG accuracy benchmark results
│
├── Supabase_scheme.sql            Database schema — run once in Supabase SQL Editor
├── README.md                      This file
├── TECHNICAL_EXPLANATION.md       Deep technical explanation + all mentor Q&A
└── Results_and_Conclusion.md      Detailed Results and Conclusion chapters


## Quick Start

### Run in GitHub Codespaces (recommended)

1. Open this repo in GitHub Codespaces — green "Code" button → Codespaces → Create
2. Wait ~90 seconds for devcontainer setup (auto-installs all dependencies)
3. Run the full ML pipeline:

python3 analysis/run_full.py

4. Run the EEG benchmark:

python3 analysis/04_eeg_benchmark.py


5. Launch the dashboard:


streamlit run dashboard/app.py --server.port 8501 --server.headless true


6. Open the port 8501 URL when Codespaces prompts you.

### Run locally


git clone https://github.com/VinodhKumar-G/prakritisense.git
cd prakritisense
pip install -r requirements.txt
python3 analysis/run_full.py
streamlit run dashboard/app.py



## What Each Command Does

### `python3 analysis/run_full.py`

Runs the complete ML pipeline in one shot:

| Step | What happens | Output |
|------|-------------|--------|
| 1 | Auto-pulls real participant data from Supabase using credentials in `db.js` | `data/real_from_supabase.csv` |
| 2 | Loads synthetic data (generates it if missing) | `data/synthetic_participants.csv` |
| 3 | Merges real + synthetic into one ML-ready dataset | `data/all_data_merged.csv` |
| 4 | Applies adaptive fatigue labels (per-participant thirds) | Labels in merged CSV |
| 5 | Trains Baseline XGBoost (7 features) with LOGO cross-validation | `models/baseline_xgboost.json` |
| 6 | Trains Prakriti-conditioned XGBoost (10 features) with LOGO CV | `models/prakriti_xgboost.json` |
| 7 | Generates performance comparison charts | `outputs/model_comparison.png`, `outputs/confusion_matrices_comparison.png` |
| 8 | Runs SHAP explainability analysis | `outputs/shap_importance_*.png`, `outputs/shap_beeswarm_*.png`, `outputs/shap_per_dosha.png` |
| 9 | Runs EEG trajectory alignment (theoretical validation) | `outputs/eeg_cross_validation.png`, `outputs/eeg_confusion_matrix.png` |
| 10 | Saves all metrics | `outputs/model_results.json`, `outputs/eeg_validation_results.json` |

**Run modes:**


# Default — auto-pull from Supabase + merge with synthetic
python3 analysis/run_full.py

# Real data only, no synthetic (recommended when you have 10+ real participants)
python3 analysis/run_full.py --no_synthetic

# Use a specific local CSV instead of Supabase pull
python3 analysis/run_full.py --real data/real_from_supabase.csv

# Skip Supabase pull entirely, use local files
python3 analysis/run_full.py --skip_supabase




### `python3 analysis/04_eeg_benchmark.py`

Downloads and processes the PhysioNet Mental Arithmetic EEG dataset and compares it against PrakritiSense accuracy:

| Step | What happens | Output |
|------|-------------|--------|
| 1 | Checks `data/eegmat/` for existing files | Skips download if already present |
| 2 | Downloads 72 EDF files from PhysioNet (Subject00–Subject35, rest + task conditions) | `data/eegmat/*.edf` |
| 3 | Reads 19-channel EEG at 500 Hz using `pyedflib` | Raw signals per subject |
| 4 | Segments into 5-second windows, computes band power using Welch's method | Delta, Theta, Alpha, Beta, Gamma, θ/α ratio |
| 5 | Labels: rest condition = Alert, arithmetic condition = CognitiveLoad | `data/eegmat/eeg_features.csv` |
| 6 | Trains XGBoost on EEG features with LOGO cross-validation | EEG benchmark accuracy |
| 7 | Compares: EEG (hardware) vs PrakritiSense (no hardware) | `outputs/eeg_benchmark_comparison.png`, `outputs/eeg_benchmark_results.json` |

**Note:** The EEG files are already downloaded and present in `data/eegmat/` (72 EDF files). The script detects this and skips the download automatically.



### `streamlit run dashboard/app.py`

Launches the real-time CFI dashboard with 5 pages:

| Page | What it shows |
|------|--------------|
| 📊 Model Results | AUC, Accuracy, F1 metric cards + model_comparison.png + confusion matrices |
| 🧠 SHAP Analysis | Global importance, beeswarm plots, per-Dosha feature rankings |
| 📈 CFI Simulator | Interactive fatigue curve — select Dosha + session minute, see live CFI gauge and trajectory |
| 📋 Data Explorer | Feature distributions by fatigue label, feature distributions by Dosha, raw data table |
| 🔬 EEG Benchmark | EEG hardware vs PrakritiSense comparison chart and key finding |



## Live Results

### Dataset

| Metric | Value |
|--------|-------|
| Total rows | 2,475 |
| Total participants | 34 (4 real + 30 synthetic) |
| Real participant rows | 98 |
| Synthetic rows | 2,377 |
| Fatigue label distribution | Alert: 826 (33.4%) / Moderate: 812 (32.8%) / Fatigued: 837 (33.8%) |
| Dosha distribution | Vata: 858 / Pitta: 820 / Kapha: 797 |

### Real Participants

| Participant ID | Vata % | Pitta % | Kapha % | Dominant |
|---------------|--------|---------|---------|---------|
| PS-FJJ5TX | 40 | 30 | 30 | Vata |
| PS-GA4MJK | 0 | 60 | 40 | Pitta |
| PS-MDY4E4 | 0 | 70 | 30 | Pitta |
| PS-VEF5SM | 70 | 10 | 20 | Vata |

### Model Performance

| Model | Accuracy | F1 (Macro) | AUC (OVR) |
|-------|----------|-----------|----------|
| Baseline (7 features) | 0.364 | 0.362 | 0.545 |
| Prakriti-conditioned (10 features) | 0.356 | 0.354 | 0.545 |
| Random chance (3-class) | 0.333 | 0.333 | 0.500 |

Both models outperform random chance. The modest accuracy (0.545 AUC vs 0.500 random) is expected at 4 real participants — similar studies report 0.50–0.60 AUC at equivalent sample sizes. With 15–20 real participants completing full 90-minute sessions, AUC is expected to improve significantly.

### ANOVA — Feature Significance

| Feature | F-statistic | p-value | Significant? |
|---------|------------|---------|-------------|
| Pause Frequency | 31.82 | < 0.0001 | ✅ |
| IKI Mean (ms) | 11.83 | < 0.0001 | ✅ |
| Mouse Velocity | 8.51 | 0.0002 | ✅ |
| CPM | 8.45 | 0.0002 | ✅ |
| Error Rate | 5.36 | 0.0047 | ✅ |
| Jitter Index | 4.47 | 0.0115 | ✅ |
| IKI Variance | 1.10 | 0.334 | ❌ |

6 of 7 features are statistically significant predictors of fatigue state (p < 0.05).

### EEG Validation

| Metric | Value |
|--------|-------|
| EEG trajectory correlation (r) | 0.974 |
| p-value | < 0.001 |
| EEG vs CFI label agreement | 87.22% |
| EEG subjects referenced | 36 (PhysioNet EEGMAT) |



## Output Figures Explained

| Figure | What it shows | Key insight |
|--------|--------------|------------|
| `model_comparison.png` | Bar chart: Baseline vs Prakriti AUC/F1/Accuracy | Both models above random chance |
| `confusion_matrices_comparison.png` | True vs predicted labels for both models | Moderate class hardest to predict (overlaps with Alert and Fatigued) |
| `shap_importance_prakriti.png` | Mean absolute SHAP per feature | Prakriti features appear in rankings — model uses Dosha type |
| `shap_beeswarm_prakriti.png` | SHAP value distribution per class | High CPM = Alert; high pause = Fatigued |
| `shap_per_dosha.png` | Top 5 SHAP features per Dosha | **Core novel finding** — different features dominate for V/P/K |
| `eeg_cross_validation.png` | EEG θ/α trajectory vs CFI over 90 min | r = 0.974 — same fatigue accumulation curve |
| `eeg_confusion_matrix.png` | EEG labels vs CFI labels | 87.22% label agreement |
| `eeg_benchmark_comparison.png` | EEG accuracy vs PrakritiSense accuracy + cost | PrakritiSense = X% of EEG accuracy at ₹0 cost |



## How the Session Works

Participants open the GitHub Pages URL and complete a 90-minute session:

| Phase | Duration | What is captured |
|-------|----------|-----------------|
| Welcome | 1 min | Auto-generated session ID (e.g. PS-K7MX3Q) — no manual entry |
| Consent | 2 min | Informed consent — no text content is ever captured |
| Prakriti Quiz | 3 min | 10 MCQs → [Vata%, Pitta%, Kapha%] constitutional profile |
| Task 1: Copy Typing | 15 min | Keystroke timings + mouse during standardised paragraph retyping |
| Task 2: Free Typing | 30 min | Keystroke timings + mouse during free-response writing |
| Task 3: Stroop Typing | 10 min | Reaction time + error rate under cognitive load (30 trials) |
| Task 4: Mouse Targeting | 15 min | Pure mouse velocity, jitter, click RT — no keyboard |
| NASA-TLX | 5 min | 6-dimension self-report cognitive load (ground-truth validation) |
| Export | 1 min | Auto-saved to Supabase + optional CSV download |

The **signal collector** (`signals.js`) runs throughout all phases, computing 7 features every 60 seconds.



## The 7 Core Features

| # | Feature | Unit | What it measures |
|---|---------|------|-----------------|
| 1 | `cpm` | chars/min | Typing speed |
| 2 | `iki_mean_ms` | ms | Mean inter-key interval (motor processing rate) |
| 3 | `iki_variance` | ms² | Typing rhythm consistency |
| 4 | `error_rate` | ratio | Backspaces / total keystrokes (cognitive accuracy) |
| 5 | `mouse_velocity_pxs` | px/sec | Average mouse cursor speed |
| 6 | `jitter_index_px` | px | Mouse path deviation (fine motor control) |
| 7 | `pause_frequency` | count | Pauses > 2 seconds (attention lapses) |

Plus 3 Prakriti conditioning features: `vata_pct`, `pitta_pct`, `kapha_pct`



## The 10 Prakriti Quiz Questions

Adapted from the **CCRAS Prakriti Assessment Scale** — developed by the Central Council for Research in Ayurvedic Sciences (CCRAS), Ministry of AYUSH, Government of India. Validated in AYU Journal 2022 (Cronbach's α ≥ 0.90 for all three Doshas).

| # | Topic | CCRAS Domain | Why included |
|---|-------|-------------|-------------|
| Q1 | Body frame and build | Sharirik (Physical) | Classical physical Dosha marker |
| Q2 | Skin texture | Sharirik | Strong physiological Dosha indicator |
| Q3 | Digestion pattern | Sharirik-Karmika | Agni — strongest single Dosha discriminator |
| Q4 | Sleep pattern | Sharirik-Karmika | Predicts fatigue resilience and onset timing |
| Q5 | Mental activity | Manasika (Psychological) | Core cognitive Dosha trait |
| Q6 | Stress response | Manasika | How each Dosha decompensates under load |
| Q7 | Energy levels | Sharirik-Karmika | Predicts time-on-task fatigue curve shape |
| Q8 | Body temperature | Sharirik | Thermal regulation (distinct from digestion) |
| Q9 | Memory and learning | Manasika | Cognitive load capacity indicator |
| Q10 | Speech style | Manasika | Behavioural Dosha indicator (non-circular) |

**Scoring:** Option A = Vata, Option B = Pitta, Option C = Kapha. One mark per matching Dosha, converted to percentages. Dominant Dosha = highest percentage (≥ 50% threshold per Frontiers in Medicine 2025 review).

**References:**
- [CCRAS AYUR Prakriti Portal](https://ccras.nic.in/ayur-prakriti-web-portal/)
- [CCRAS-PAS Manual PDF](https://ccras.nic.in/wp-content/uploads/2024/07/15032023_AYUR-PRAKRITI-WEB-PORTAL-Manual.pdf)
- Singh R et al. AYU 2022;43:109–29 — Cronbach's α = 0.923 (Vata), 0.909 (Pitta), 0.903 (Kapha)
- Gupta P et al. Frontiers in Medicine 2025 — Only 2 of 64 tools met 7/9 psychometric criteria; CCRAS-PAS is one



## EEG Dataset Details

**Dataset:** PhysioNet Mental Arithmetic EEG (EEGMAT)
**Citation:** Zyma I, Tukaev S, Seleznov I et al. Data. 2019;4(1):14
**URL:** https://physionet.org/content/eegmat/1.0.0/
**License:** Open Data Commons PDDL — free for all research use
**Location:** `data/eegmat/` (72 EDF files already downloaded)

| Property | Value |
|----------|-------|
| Subjects | 36 healthy volunteers |
| Channels | 19 EEG channels |
| Sampling rate | 500 Hz |
| Conditions | Rest (baseline) + Mental arithmetic (cognitive load) |
| Features extracted | Delta, Theta, Alpha, Beta, Gamma band power, θ/α ratio |
| Labeling | Rest → Alert, Arithmetic → CognitiveLoad |



## Supabase Setup

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to SQL Editor → New Query → paste `Supabase_scheme.sql` → Run
3. Go to Settings → API → copy Project URL and anon public key
4. Paste into `db.js` lines 15–16:


const SUPABASE_URL      = "https://YOUR_PROJECT_ID.supabase.co";
const SUPABASE_ANON_KEY = "YOUR_ANON_PUBLIC_KEY";


5. Push to GitHub — data collection starts immediately



## Deployment

### Data Collection App → GitHub Pages (live now)

https://VinodhKumar-G.github.io/prakritisense/


No server needed. Participants open this URL, complete the 90-minute session, and data auto-saves to Supabase.

### Dashboard → Streamlit in Codespaces


streamlit run dashboard/app.py --server.port 8501 --server.headless true


Make port 8501 public in the Ports tab if sharing with others.



## Privacy and Ethics

| What is captured | What is NOT captured |
|-----------------|---------------------|
| Keystroke timestamps (when keys pressed/released) | Text content (words typed) |
| Key codes (which key) | Personal identifying information |
| Mouse coordinates (x, y, sampled 100ms) | Screen contents or screenshots |
| Click events (when and where) | Any data outside this browser tab |

All data stored under auto-generated session IDs (e.g. PS-K7MX3Q). No real names ever collected. Participants can download their own CSV at session end. Informed consent required before any data collection begins.



## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data collection | Vanilla JavaScript (ES6+) | Passive keystroke + mouse event capture |
| Quiz UI | HTML5 + CSS3 | Prakriti constitutional assessment |
| Cloud database | Supabase (PostgreSQL) | Auto-save participant data |
| ML model | XGBoost + scikit-learn | Fatigue classification (3-class) |
| Explainability | SHAP (TreeExplainer) | Per-feature prediction attribution |
| Dashboard | Streamlit + Plotly | Real-time CFI visualisation |
| EEG processing | pyedflib + scipy.signal | EDF reading + Welch band power |
| Deployment | GitHub Pages | Static web app hosting |
| Dev environment | GitHub Codespaces | Cloud IDE with auto-installed dependencies |



## Key Research References

| # | Reference | Role in project |
|---|-----------|----------------|
| 1 | Acien et al. (2022). TypeNet keystroke fatigue. JMIR Biomed Eng. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041424/) | Foundation for keystroke feature design |
| 2 | Banholzer et al. (2021). Mouse movements as stress indicator. JMIR. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8052599/) | Foundation for mouse feature design |
| 3 | Govindaraj et al. (2015). Genome-wide analysis correlates Prakriti. Nature. [Link](https://www.nature.com/articles/srep15786) | Genomic validation of Prakriti as stable phenotype |
| 4 | Singh R et al. (2022). CCRAS-PAS validation. AYU 2022;43:109–29 | Prakriti quiz psychometric validation |
| 5 | Gupta P et al. (2025). Prakriti tools review. Frontiers in Medicine. [Link](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1656249/full) | Confirms CCRAS-PAS as best available instrument |
| 6 | CCRAS (2023). Manual of SOPs for Prakriti Assessment. Ministry of AYUSH. [PDF](https://ccras.nic.in/wp-content/uploads/2024/07/15032023_AYUR-PRAKRITI-WEB-PORTAL-Manual.pdf) | Source of quiz questions |
| 7 | Zyma et al. (2019). PhysioNet EEGMAT. [PhysioNet](https://physionet.org/content/eegmat/) | EEG benchmark dataset |
| 8 | MedRxiv (2025). One Does Not Fit All. [Link](https://medrxiv.org/content/10.1101/2025.08.02.25332538) | Empirical proof personalised ML outperforms population baseline |
| 9 | MDPI Informatics (2025). XAI workplace health. [Link](https://www.mdpi.com/2227-9709/12/4/130) | Validates XGBoost + SHAP for mental health context |
| 10 | JMIR mHealth (2025). Keystroke temporal variability. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12709153/) | Validates within-person drift as fatigue signal |





## Team

| Role | Responsibility |
|------|---------------|
| Member A — ML Engineer | Python pipeline, XGBoost training, SHAP analysis, model validation, IEEE paper Methods and Results |
| Member B — Frontend and IKS Specialist | JavaScript collector, Prakriti quiz, Streamlit dashboard, Supabase integration, IEEE paper Introduction and Discussion |

**Program:** IEEE EMBS Student Internship Program 2026
**Domain:** AI and Healthcare × Indian Knowledge Systems
**Mentor:** Dr.Swati Bhonde



## License

Research use only. © PrakritiSense Team , IEEE EMBS Internship 2026.