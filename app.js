/* ═══════════════════════════════════════════════════════════════════════
   PrakritiSense Main App
   ──────────────────────
   Phase routing, task implementations, export logic.
   ═══════════════════════════════════════════════════════════════════════ */

// ──────── STATE ──────────────────────────────────────────────────────

const state = {
  participantId: null,
  currentPhase: "welcome",
  phaseSequence: [
    "welcome", "consent", "quiz", "quiz-result",
    "typing-copy", "typing-free", "stroop", "mouse",
    "tlx", "done"
  ],
  quizCurrent: 0,
  quizAnswers: new Array(10).fill(null),
  prakritiResult: null,
  copyTaskData: { keystrokes: 0, chars: 0, errors: 0, startTime: null },
  freeTaskData: { keystrokes: 0, chars: 0, errors: 0, startTime: null },
  stroopTrials: [],
  stroopCurrent: 0,
  stroopResults: [],
  stroopStimulusTime: 0,
  mouseTaskData: { hits: 0, misses: 0, rtSum: 0, jitterSum: 0, samples: 0, startTime: null, lastPos: null },
  mouseTargetTime: 0,
  tlxValues: {},
  taskTimers: {},
  sessionStartTime: null,
  globalElapsedInterval: null,
};

// ──────── PHASE NAVIGATION ────────────────────────────────────────────

function goToPhase(phase) {
  document.querySelectorAll(".phase").forEach(el => el.classList.remove("active"));
  const target = document.getElementById(`phase-${phase}`);
  if (target) target.classList.add("active");

  state.currentPhase = phase;
  if (collector.isCollecting) collector.setPhase(phase);

  // Update header phase indicator
  document.querySelectorAll(".phase-dot").forEach(dot => {
    const dotPhase = dot.dataset.phase;
    const idx = state.phaseSequence.indexOf(dotPhase);
    const currentIdx = state.phaseSequence.indexOf(phase);
    dot.classList.remove("active", "complete");
    if (idx < currentIdx) dot.classList.add("complete");
    else if (idx === currentIdx) dot.classList.add("active");
  });

  // Update footer phase label
  const phaseLabel = document.getElementById("phaseLabel");
  if (phaseLabel) {
    phaseLabel.textContent = {
      "welcome": "Welcome",
      "consent": "Consent",
      "quiz": "Prakriti quiz",
      "quiz-result": "Quiz result",
      "typing-copy": "Task 1 — Copy typing",
      "typing-free": "Task 2 — Free typing",
      "stroop": "Task 3 — Stroop task",
      "mouse": "Task 4 — Mouse targeting",
      "tlx": "NASA-TLX",
      "done": "Session complete"
    }[phase] || phase;
  }

  // Phase initialization
  if (phase === "quiz") initQuiz();
  if (phase === "typing-copy") initCopyTask();
  if (phase === "typing-free") initFreeTask();
  if (phase === "stroop") initStroopTask();
  if (phase === "mouse") initMouseTask();
  if (phase === "tlx") initTLX();
  if (phase === "done") finalizeSession();

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ──────── WELCOME → SESSION START ────────────────────────────────────

function startSession() {
  const idInput = document.getElementById("participantId");
  const id = idInput.value.trim();
  if (!id) {
    alert("Please enter your participant ID before starting.");
    idInput.focus();
    return;
  }
  state.participantId = id;
  state.sessionStartTime = Date.now();
  collector.start(id, "welcome");

  // Global elapsed time counter
  state.globalElapsedInterval = setInterval(() => {
    const el = document.getElementById("elapsedTime");
    if (el && state.sessionStartTime) {
      const sec = Math.floor((Date.now() - state.sessionStartTime) / 1000);
      const mm = String(Math.floor(sec / 60)).padStart(2, "0");
      const ss = String(sec % 60).padStart(2, "0");
      el.textContent = `${mm}:${ss}`;
    }
  }, 1000);

  goToPhase("consent");
}

// ──────── CONSENT ────────────────────────────────────────────────────

function updateConsentBtn() {
  const checked = document.getElementById("consentCheck").checked;
  document.getElementById("consentNextBtn").disabled = !checked;
}

// ──────── PRAKRITI QUIZ ──────────────────────────────────────────────

function initQuiz() {
  state.quizCurrent = 0;
  renderQuiz();
}

function renderQuiz() {
  const q = PRAKRITI_QUESTIONS[state.quizCurrent];
  document.getElementById("qCounter").textContent = `Question ${state.quizCurrent + 1} of 10`;
  document.getElementById("quizProgress").style.width = `${((state.quizCurrent + 1) / 10) * 100}%`;
  document.getElementById("qText").textContent = q.text;

  const grid = document.getElementById("optionsGrid");
  grid.innerHTML = "";
  const labels = ["A", "B", "C"];
  q.options.forEach((opt, i) => {
    const btn = document.createElement("button");
    btn.className = "opt-btn" + (state.quizAnswers[state.quizCurrent] === i ? " selected" : "");
    btn.innerHTML = `<span class="opt-label">${labels[i]}</span><span>${opt}</span>`;
    btn.onclick = () => selectQuizOption(i);
    grid.appendChild(btn);
  });

  document.getElementById("quizBackBtn").style.visibility = state.quizCurrent === 0 ? "hidden" : "visible";
  const nextBtn = document.getElementById("quizNextBtn");
  nextBtn.textContent = state.quizCurrent === 9 ? "See result →" : "Next →";
  nextBtn.disabled = state.quizAnswers[state.quizCurrent] === null;
}

function selectQuizOption(i) {
  state.quizAnswers[state.quizCurrent] = i;
  renderQuiz();
}

function goQuizNext() {
  if (state.quizAnswers[state.quizCurrent] === null) return;
  if (state.quizCurrent === 9) {
    state.prakritiResult = scorePrakriti(state.quizAnswers);
    collector.injectPrakriti(
      state.prakritiResult.vataPct,
      state.prakritiResult.pittaPct,
      state.prakritiResult.kaphaPct
    );
    renderQuizResult();
    goToPhase("quiz-result");
    return;
  }
  state.quizCurrent++;
  renderQuiz();
}

function goQuizBack() {
  if (state.quizCurrent === 0) return;
  state.quizCurrent--;
  renderQuiz();
}

function renderQuizResult() {
  const r = state.prakritiResult;
  const dominantInfo = DOSHA_INFO[r.dominant];

  const html = `
    <span class="dominant-badge" style="background:${dominantInfo.bg};color:${dominantInfo.dark};">
      Dominant: ${dominantInfo.name.split(" ")[0]}
    </span>
    ${["V", "P", "K"].map(d => {
      const pct = d === "V" ? r.vataPct : d === "P" ? r.pittaPct : r.kaphaPct;
      const info = DOSHA_INFO[d];
      return `
        <div class="dosha-card">
          <div class="dosha-header">
            <span class="dosha-name">${info.name}</span>
            <span class="dosha-pct" style="color:${info.color}">${pct}%</span>
          </div>
          <div class="dosha-bar-bg">
            <div class="dosha-bar-fill" style="width:${pct}%;background:${info.color}"></div>
          </div>
          <div class="dosha-desc">${info.desc}</div>
        </div>
      `;
    }).join("")}
  `;
  document.getElementById("doshaResults").innerHTML = html;
}

// ──────── TASK 1: COPY TYPING ────────────────────────────────────────

function initCopyTask() {
  document.getElementById("copySourceText").textContent = COPY_TEXT;
  const area = document.getElementById("copyTypingArea");
  area.value = "";
  state.copyTaskData = { keystrokes: 0, chars: 0, errors: 0, startTime: Date.now() };

  area.addEventListener("input", updateCopyStats);
  area.addEventListener("keydown", e => {
    state.copyTaskData.keystrokes++;
    if (e.key === "Backspace") state.copyTaskData.errors++;
  });

  startTaskTimer("timer1", 15 * 60, () => {
    document.getElementById("copyNextBtn").disabled = false;
  });

  setTimeout(() => area.focus(), 300);
}

function updateCopyStats() {
  const area = document.getElementById("copyTypingArea");
  const elapsed = (Date.now() - state.copyTaskData.startTime) / 60000;
  state.copyTaskData.chars = area.value.length;
  document.getElementById("copyKeystrokes").textContent = state.copyTaskData.keystrokes;
  document.getElementById("copyChars").textContent = state.copyTaskData.chars;
  document.getElementById("copyErrors").textContent = state.copyTaskData.errors;
  document.getElementById("copyCPM").textContent = elapsed > 0
    ? Math.round(state.copyTaskData.chars / elapsed)
    : 0;
}

function finishCopyTask() {
  clearTaskTimer("timer1");
  goToPhase("typing-free");
}

// ──────── TASK 2: FREE TYPING ────────────────────────────────────────

function initFreeTask() {
  document.getElementById("freePromptText").textContent = FREE_PROMPT;
  const area = document.getElementById("freeTypingArea");
  area.value = "";
  state.freeTaskData = { keystrokes: 0, chars: 0, errors: 0, startTime: Date.now() };

  area.addEventListener("input", updateFreeStats);
  area.addEventListener("keydown", e => {
    state.freeTaskData.keystrokes++;
    if (e.key === "Backspace") state.freeTaskData.errors++;
  });

  startTaskTimer("timer2", 30 * 60);

  setTimeout(() => area.focus(), 300);
}

function updateFreeStats() {
  const area = document.getElementById("freeTypingArea");
  const elapsed = (Date.now() - state.freeTaskData.startTime) / 60000;
  state.freeTaskData.chars = area.value.length;
  document.getElementById("freeKeystrokes").textContent = state.freeTaskData.keystrokes;
  document.getElementById("freeChars").textContent = state.freeTaskData.chars;
  document.getElementById("freeErrors").textContent = state.freeTaskData.errors;
  document.getElementById("freeCPM").textContent = elapsed > 0
    ? Math.round(state.freeTaskData.chars / elapsed)
    : 0;
}

function finishFreeTask() {
  clearTaskTimer("timer2");
  goToPhase("stroop");
}

// ──────── TASK 3: STROOP TASK ────────────────────────────────────────

function initStroopTask() {
  state.stroopTrials = generateStroopTrials(60);
  state.stroopCurrent = 0;
  state.stroopResults = [];
  document.getElementById("stroopTotalTrials").textContent = state.stroopTrials.length;
  startTaskTimer("timer3", 15 * 60);

  // Keyboard listener for Stroop
  document.addEventListener("keydown", handleStroopKey);

  setTimeout(showNextStroopTrial, 800);
}

function showNextStroopTrial() {
  if (state.stroopCurrent >= state.stroopTrials.length) {
    finishStroopTask();
    return;
  }
  const trial = state.stroopTrials[state.stroopCurrent];
  const stim = document.getElementById("stroopStimulus");
  stim.textContent = trial.word;
  stim.style.color = trial.inkHex;
  document.getElementById("stroopTrialNum").textContent = state.stroopCurrent + 1;
  document.getElementById("stroopFeedback").textContent = "";
  document.getElementById("stroopFeedback").className = "stroop-feedback";
  state.stroopStimulusTime = Date.now();
}

function handleStroopKey(e) {
  if (state.currentPhase !== "stroop") return;
  if (state.stroopCurrent >= state.stroopTrials.length) return;
  if (state.stroopStimulusTime === 0) return;

  const key = e.key.toLowerCase();
  if (!["r","g","b","y","p","o"].includes(key)) return;

  const trial = state.stroopTrials[state.stroopCurrent];
  const rt = Date.now() - state.stroopStimulusTime;
  const correct = key === trial.correctKey;

  state.stroopResults.push({
    trial: state.stroopCurrent + 1,
    word: trial.word,
    inkColor: trial.inkColor,
    congruent: trial.congruent,
    response: key,
    correct,
    rt_ms: rt,
  });

  const feedback = document.getElementById("stroopFeedback");
  feedback.textContent = correct ? "✓ Correct" : `✗ Should be ${trial.correctKey.toUpperCase()}`;
  feedback.className = "stroop-feedback " + (correct ? "correct" : "wrong");

  const correctCount = state.stroopResults.filter(r => r.correct).length;
  const wrongCount = state.stroopResults.filter(r => !r.correct).length;
  const avgRT = Math.round(state.stroopResults.reduce((s, r) => s + r.rt_ms, 0) / state.stroopResults.length);
  document.getElementById("stroopCorrect").textContent = correctCount;
  document.getElementById("stroopWrong").textContent = wrongCount;
  document.getElementById("stroopAvgRT").textContent = avgRT;

  state.stroopCurrent++;
  if (state.stroopCurrent >= state.stroopTrials.length) {
    document.getElementById("stroopNextBtn").disabled = false;
    document.getElementById("stroopStimulus").textContent = "DONE";
    document.getElementById("stroopStimulus").style.color = "var(--sage)";
  } else {
    setTimeout(showNextStroopTrial, 500);
  }
}

function finishStroopTask() {
  clearTaskTimer("timer3");
  document.removeEventListener("keydown", handleStroopKey);
  goToPhase("mouse");
}

// ──────── TASK 4: MOUSE TARGETING ────────────────────────────────────

function initMouseTask() {
  state.mouseTaskData = {
    hits: 0, misses: 0, rtSum: 0,
    jitterSum: 0, samples: 0,
    startTime: Date.now(), lastPos: null
  };
  startTaskTimer("timer4", 15 * 60, finishMouseTask);

  const arena = document.getElementById("mouseArena");
  arena.addEventListener("click", handleArenaClick);
  arena.addEventListener("mousemove", handleArenaMouseMove);

  spawnTarget();
}

function spawnTarget() {
  const arena = document.getElementById("mouseArena");
  const target = document.getElementById("mouseTarget");
  const rect = arena.getBoundingClientRect();
  const x = Math.random() * (rect.width - 60) + 5;
  const y = Math.random() * (rect.height - 60) + 5;
  target.style.left = `${x}px`;
  target.style.top = `${y}px`;
  target.style.display = "block";
  state.mouseTargetTime = Date.now();
}

function handleArenaClick(e) {
  const target = document.getElementById("mouseTarget");
  if (e.target === target) {
    const rt = Date.now() - state.mouseTargetTime;
    state.mouseTaskData.hits++;
    state.mouseTaskData.rtSum += rt;
    target.style.display = "none";
    setTimeout(spawnTarget, 400);
  } else if (e.target.id === "mouseArena") {
    state.mouseTaskData.misses++;
  }
  updateMouseStats();
}

function handleArenaMouseMove(e) {
  const arena = document.getElementById("mouseArena");
  const rect = arena.getBoundingClientRect();
  const pos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  if (state.mouseTaskData.lastPos) {
    const dx = pos.x - state.mouseTaskData.lastPos.x;
    const dy = pos.y - state.mouseTaskData.lastPos.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist > 0 && dist < 30) {
      state.mouseTaskData.jitterSum += dist;
      state.mouseTaskData.samples++;
    }
  }
  state.mouseTaskData.lastPos = pos;
}

function updateMouseStats() {
  const d = state.mouseTaskData;
  document.getElementById("mouseHits").textContent = d.hits;
  document.getElementById("mouseMisses").textContent = d.misses;
  document.getElementById("mouseAvgRT").textContent = d.hits > 0 ? Math.round(d.rtSum / d.hits) : 0;
  document.getElementById("mouseJitter").textContent = d.samples > 0 ? (d.jitterSum / d.samples).toFixed(1) : 0;
}

function finishMouseTask() {
  clearTaskTimer("timer4");
  const arena = document.getElementById("mouseArena");
  arena.removeEventListener("click", handleArenaClick);
  arena.removeEventListener("mousemove", handleArenaMouseMove);
  goToPhase("tlx");
}

// ──────── NASA-TLX ───────────────────────────────────────────────────

function initTLX() {
  const grid = document.getElementById("tlxGrid");
  grid.innerHTML = "";
  NASA_TLX_DIMENSIONS.forEach(dim => {
    state.tlxValues[dim.key] = 50;
    const div = document.createElement("div");
    div.className = "tlx-item";
    div.innerHTML = `
      <div class="tlx-label">
        <span class="tlx-name">${dim.name}</span>
        <span class="tlx-value" id="val_${dim.key}">50</span>
      </div>
      <div class="tlx-desc">${dim.desc}</div>
      <input type="range" class="tlx-slider" id="slider_${dim.key}"
             min="0" max="100" value="50"
             oninput="updateTLX('${dim.key}', this.value)">
      <div class="tlx-scale">
        <span>${dim.low}</span>
        <span>${dim.high}</span>
      </div>
    `;
    grid.appendChild(div);
  });
}

function updateTLX(key, val) {
  state.tlxValues[key] = parseInt(val);
  document.getElementById(`val_${key}`).textContent = val;
}

function finishTLX() {
  goToPhase("done");
}

// ──────── FINALIZE & EXPORT ─────────────────────────────────────────

function finalizeSession() {
  collector.injectLabels();
  collector.stop();
  if (state.globalElapsedInterval) clearInterval(state.globalElapsedInterval);

  const stats = collector.getStats();
  const r = state.prakritiResult;

  const summary = `
Participant ID:           ${state.participantId}
Session duration:         ${stats.elapsedMin} minutes
Feature windows captured: ${stats.windows}
Total keystrokes:         ${stats.keystrokes}
Total mouse moves:        ${stats.mouseMoves}
Total clicks:             ${stats.clicks}

Prakriti profile:
  Vata:  ${r.vataPct}%
  Pitta: ${r.pittaPct}%
  Kapha: ${r.kaphaPct}%
  Dominant: ${DOSHA_INFO[r.dominant].name}

Stroop task:
  Correct: ${state.stroopResults.filter(s => s.correct).length} / ${state.stroopResults.length}
  Avg RT:  ${state.stroopResults.length > 0 ? Math.round(state.stroopResults.reduce((s,r)=>s+r.rt_ms,0) / state.stroopResults.length) : 0} ms

Mouse task:
  Hits:    ${state.mouseTaskData.hits}
  Misses:  ${state.mouseTaskData.misses}
  Avg RT:  ${state.mouseTaskData.hits > 0 ? Math.round(state.mouseTaskData.rtSum / state.mouseTaskData.hits) : 0} ms

NASA-TLX:
  Mental Demand:   ${state.tlxValues.mental_demand}/100
  Physical Demand: ${state.tlxValues.physical_demand}/100
  Temporal Demand: ${state.tlxValues.temporal_demand}/100
  Performance:     ${state.tlxValues.performance}/100
  Effort:          ${state.tlxValues.effort}/100
  Frustration:     ${state.tlxValues.frustration}/100
`;
  document.getElementById("sessionSummary").textContent = summary.trim();
}

function exportCSV() {
  const csv = collector.exportCSV();
  const filename = `prakritisense_${state.participantId}_${Date.now()}.csv`;
  saveToServer(filename, csv, "/save-csv");
  download(filename, csv, "text/csv");
}

function exportJSON() {
  const data = collector.exportJSON({
    prakriti_result: state.prakritiResult,
    quiz_answers: state.quizAnswers.map((a, i) => ({
      qid: PRAKRITI_QUESTIONS[i].id,
      answer_index: a,
      answer_dosha: a !== null ? PRAKRITI_QUESTIONS[i].dosha[a] : null,
    })),
    stroop_trials: state.stroopResults,
    mouse_task: {
      hits: state.mouseTaskData.hits,
      misses: state.mouseTaskData.misses,
      avg_rt_ms: state.mouseTaskData.hits > 0 ? state.mouseTaskData.rtSum / state.mouseTaskData.hits : 0,
      avg_jitter_px: state.mouseTaskData.samples > 0 ? state.mouseTaskData.jitterSum / state.mouseTaskData.samples : 0,
    },
    nasa_tlx: state.tlxValues,
  });
  const filename = `prakritisense_${state.participantId}_${Date.now()}.json`;
  saveToServer(filename, data, "/save-json");
  download(filename, data, "application/json");
}

function saveToServer(filename, content, endpoint) {
  fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, content }),
  })
    .then(res => res.json())
    .then(result => {
      console.log("[saveToServer] Saved into Codespace:", result.path);
      const note = document.querySelector(".export-note");
      if (note) {
        note.textContent = `Saved to ${result.path} inside this Codespace, and downloaded to your computer.`;
        note.style.color = "var(--sage)";
      }
    })
    .catch(err => {
      console.warn("[saveToServer] Could not reach server.py — file was still downloaded locally.", err);
      const note = document.querySelector(".export-note");
      if (note) {
        note.textContent = "Could not save into Codespace automatically — file was downloaded to your computer. Drag it into the Codespace file explorer manually, or check that server.py is running.";
        note.style.color = "var(--coral)";
      }
    });
}

function download(filename, content, mimetype) {
  const blob = new Blob([content], { type: mimetype });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function copySessionSummary() {
  const summary = document.getElementById("sessionSummary").textContent;
  navigator.clipboard.writeText(summary).then(() => {
    alert("Summary copied to clipboard.");
  });
}

// ──────── TIMER HELPERS ──────────────────────────────────────────────

function startTaskTimer(timerId, totalSeconds, onComplete) {
  let remaining = totalSeconds;
  const update = () => {
    const el = document.getElementById(timerId);
    if (!el) return;
    const mm = String(Math.floor(remaining / 60)).padStart(2, "0");
    const ss = String(remaining % 60).padStart(2, "0");
    el.textContent = `${mm}:${ss}`;
    if (remaining <= 60) el.classList.add("warning");
    if (remaining <= 0) {
      clearInterval(state.taskTimers[timerId]);
      if (onComplete) onComplete();
      return;
    }
    remaining--;
  };
  update();
  state.taskTimers[timerId] = setInterval(update, 1000);
}

function clearTaskTimer(timerId) {
  if (state.taskTimers[timerId]) {
    clearInterval(state.taskTimers[timerId]);
    delete state.taskTimers[timerId];
  }
}

// ──────── INITIALIZE ON LOAD ─────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
  goToPhase("welcome");
});