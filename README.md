# PrakritiSense — Data Collection App

Browser-based data collection tool for the PrakritiSense cognitive fatigue study.

Captures keystroke, mouse, and click micro-interactions during a structured 90-minute
session, conditioned on the user's Ayurvedic Prakriti type.

---

## Project structure

```
prakritisense/
├── index.html           Main app shell (welcome → consent → quiz → 4 tasks → TLX → export)
├── styles.css           All styling
├── signals.js           Passive signal collector (the engine — captures all events)
├── tasks.js             Task content (quiz questions, copy text, prompts, Stroop, TLX)
├── app.js               Application logic, phase routing, task implementations
└── README.md            This file
```

---

## Running in GitHub Codespaces

### Option 1 — Quickest (no setup)
1. Open this repo in GitHub Codespaces.
2. In the Codespaces terminal, run:
   ```
   python3 -m http.server 8000
   ```
3. Click "Open in Browser" when Codespaces prompts you, OR open the Ports tab and click the globe icon next to port 8000.
4. The app is live.

### Option 2 — Live reload (recommended for development)
1. Install Live Server: `npm install -g live-server`
2. Run: `live-server --port=8000`
3. Browser opens automatically. File changes auto-reload.

---

## Running locally

Any static HTTP server works. The simplest:

```bash
cd prakritisense
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

> The app must be served over HTTP, not opened directly as `file://`, because some browsers restrict clipboard and event APIs on file URLs.

---

## What gets captured

### Per-window features (7 core features, computed every 60 seconds)

| Feature              | Unit        | What it measures                          |
|----------------------|-------------|-------------------------------------------|
| `cpm`                | chars/min   | Typing speed                              |
| `iki_mean_ms`        | ms          | Mean inter-key interval                   |
| `iki_variance`       | ms²         | Rhythm consistency                        |
| `error_rate`         | ratio       | Backspaces / total keystrokes             |
| `mouse_velocity_pxs` | px/sec      | Average mouse cursor speed                |
| `jitter_index_px`    | px          | Average small movement displacement       |
| `pause_frequency`    | count       | Number of >2s pauses                      |

Plus contextual fields: participant ID, phase (which task), window start time,
Vata/Pitta/Kapha scores from the quiz, and the fatigue label (Alert / Moderate / Fatigued)
derived from time-on-task.

### What is NEVER captured

- Typed text content (only keystroke timings and key codes)
- Personal identifying information
- Anything outside the browser tab
- Screen contents

---

## Session flow (90 minutes total)

| Phase             | Duration | Captures                                |
|-------------------|----------|-----------------------------------------|
| Welcome           | 1 min    | Participant ID entry                    |
| Consent           | 2 min    | Informed consent checkbox               |
| Prakriti quiz     | 3 min    | 10 MCQs → [V%, P%, K%] profile          |
| Task 1: Copy typing | 15 min | Keystroke + mouse during transcription |
| Task 2: Free typing | 30 min | Keystroke + mouse during free writing  |
| Task 3: Stroop typing | 15 min | Reaction time + error rate under load |
| Task 4: Mouse targeting | 15 min | Pure mouse signals (click + jitter)  |
| NASA-TLX          | 5 min    | 6-dim self-report cognitive load        |
| Export            | 1 min    | Download CSV + JSON                     |

---

## Exports

Two files per participant:

### `prakritisense_P001_<timestamp>.csv`
One row per 60-second feature window. Direct input to the ML pipeline.

### `prakritisense_P001_<timestamp>.json`
Complete session: all feature windows, Prakriti result, quiz answers, Stroop trial-level
results, mouse task summary, NASA-TLX values, raw event counts.

---

## Prakriti quiz — the 10 questions

The quiz is adapted from CCRAS-PAS (Central Council for Research in Ayurvedic Sciences
Prakriti Assessment Scale, validated 2025).

| # | Topic                          | Why included                                       |
|---|--------------------------------|----------------------------------------------------|
| 1 | Body frame and build           | Classical physical Dosha marker                    |
| 2 | Skin texture                   | Strong physiological Dosha indicator               |
| 3 | Digestion pattern              | Agni — strongest Dosha discriminator               |
| 4 | Sleep pattern                  | Predicts fatigue resilience                        |
| 5 | Mental activity pattern        | Cognitive Dosha trait                              |
| 6 | Stress response style          | How each Dosha decompensates under load            |
| 7 | Energy pattern through the day | Predicts time-on-task fatigue curve                |
| 8 | Body temperature regulation    | (Replaced Q3 duplicate)                            |
| 9 | Memory and learning style      | Cognitive load capacity indicator                  |
| 10| Speech and conversation style  | (Replaced circular self-report about typing)       |

---

## Day-by-day plan (Week 2 → Week 3)

### Today (setup)
- [x] Deploy this app in GitHub Codespaces
- [ ] Test on yourself end-to-end (run a full 90-min session as P000)
- [ ] Verify CSV exports correctly
- [ ] Share session link with first 2 participants

### Tomorrow (data collection starts)
- [ ] Run 3 participants
- [ ] Inspect each CSV — check for missing windows, zero-value features
- [ ] Begin Python feature analysis pipeline (see `/analysis/` to be created)

### Day after (modelling begins)
- [ ] 5–8 participant CSVs collected
- [ ] Run baseline XGBoost on combined CSV (no Prakriti features)
- [ ] Run Prakriti-conditioned XGBoost
- [ ] Compare AUC

### By end of Week 3
- [ ] SHAP analysis complete
- [ ] Streamlit dashboard with CFI gauge
- [ ] Ready for Week 4 paper writing

---

## Next files to create (in `/analysis/`)

You'll build these tomorrow after collecting first 3 sessions:

- `analysis/01_load_data.py` — merge all participant CSVs
- `analysis/02_eda.py` — ANOVA across Prakriti groups
- `analysis/03_train_baseline.py` — XGBoost without Prakriti
- `analysis/04_train_prakriti.py` — XGBoost with Prakriti
- `analysis/05_shap.py` — SHAP explainability
- `analysis/06_streamlit_dashboard.py` — real-time CFI display

---

## Troubleshooting

**Keystrokes not being captured?**
Open browser DevTools → Console. You should see `[SignalCollector] Started for participant P001`.
If not, the script failed to load — check the network tab for 404s.

**Feature counter stuck at 0?**
The collector flushes a window every 60 seconds. If you're still in the welcome/consent phase,
or just started a task, no windows have been flushed yet. Wait 60 seconds into a typing task.

**CSV download blocked?**
Some browsers block downloads from non-HTTPS local origins. Use HTTPS (`live-server --https`)
or run in Codespaces, which serves over HTTPS by default.

---

## License

Research use only. © PrakritiSense team, IEEE EMBS Internship 2026.
