/* ═══════════════════════════════════════════════════════════════════════
   PrakritiSense Signal Collector
   ─────────────────────────────
   Captures keystroke + mouse + click events globally during ALL tasks.
   Aggregates into 60-second feature windows.
   Privacy-safe: NO text content captured — only event metadata.
   ═══════════════════════════════════════════════════════════════════════ */

class SignalCollector {
  constructor() {
    this.sessionStart = null;
    this.windowSize = 60000;  // 60 seconds per feature window
    this.windows = [];        // Array of computed feature windows
    this.currentWindow = this.newEmptyWindow();
    this.windowStartTime = null;

    // Raw event buffers (cleared after each window flush)
    this.keystrokes = [];     // {type, key, time}
    this.mouseMoves = [];     // {x, y, time}
    this.clicks = [];         // {x, y, time}
    this.lastEventTime = 0;
    this.lastMouseSample = 0;

    // State
    this.isCollecting = false;
    this.currentPhase = "welcome";
    this.participantId = "UNKNOWN";

    // For mouse jitter calculation
    this.lastMousePos = null;
  }

  newEmptyWindow() {
    return {
      keydownCount: 0,
      keyupCount: 0,
      backspaceCount: 0,
      enterCount: 0,
      spaceCount: 0,
      mouseMoveCount: 0,
      clickCount: 0,
      ikiSum: 0,           // For inter-key interval mean
      ikiSqSum: 0,         // For variance computation
      ikiCount: 0,
      mouseVelocitySum: 0,
      mouseVelocityCount: 0,
      mouseJitterSum: 0,
      mouseJitterCount: 0,
      pauseCount: 0,       // Number of >2s pauses
      lastKeydownTime: 0,
    };
  }

  start(participantId, phase) {
    this.participantId = participantId;
    this.currentPhase = phase;
    this.sessionStart = Date.now();
    this.windowStartTime = Date.now();
    this.isCollecting = true;

    document.addEventListener("keydown", this.handleKeydown.bind(this), true);
    document.addEventListener("keyup", this.handleKeyup.bind(this), true);
    document.addEventListener("mousemove", this.handleMouseMove.bind(this), true);
    document.addEventListener("click", this.handleClick.bind(this), true);

    // Auto-flush window every 60s
    this.windowInterval = setInterval(() => this.flushWindow(), this.windowSize);

    console.log("[SignalCollector] Started for participant", participantId);
  }

  setPhase(phase) {
    if (this.currentPhase !== phase) {
      this.flushWindow();  // Flush current window before phase change
      this.currentPhase = phase;
      console.log("[SignalCollector] Phase →", phase);
    }
  }

  stop() {
    this.flushWindow();
    this.isCollecting = false;
    clearInterval(this.windowInterval);
    document.removeEventListener("keydown", this.handleKeydown.bind(this), true);
    document.removeEventListener("keyup", this.handleKeyup.bind(this), true);
    document.removeEventListener("mousemove", this.handleMouseMove.bind(this), true);
    document.removeEventListener("click", this.handleClick.bind(this), true);
    console.log("[SignalCollector] Stopped. Total windows:", this.windows.length);
  }

  /* ──────── EVENT HANDLERS ──────── */

  handleKeydown(e) {
    if (!this.isCollecting) return;
    const now = Date.now();

    // Detect pause (>2s since last event)
    if (this.lastEventTime > 0 && (now - this.lastEventTime) > 2000) {
      this.currentWindow.pauseCount++;
    }

    // Inter-key interval
    if (this.currentWindow.lastKeydownTime > 0) {
      const iki = now - this.currentWindow.lastKeydownTime;
      if (iki < 5000) {  // Ignore long pauses for IKI calculation
        this.currentWindow.ikiSum += iki;
        this.currentWindow.ikiSqSum += iki * iki;
        this.currentWindow.ikiCount++;
      }
    }
    this.currentWindow.lastKeydownTime = now;
    this.currentWindow.keydownCount++;

    // Special keys
    if (e.key === "Backspace") this.currentWindow.backspaceCount++;
    else if (e.key === "Enter") this.currentWindow.enterCount++;
    else if (e.key === " ") this.currentWindow.spaceCount++;

    this.keystrokes.push({
      type: "keydown",
      keyCode: e.keyCode,
      isSpecial: ["Backspace","Enter","Tab","Shift","Control","Alt"].includes(e.key),
      time: now - this.sessionStart,
    });

    this.lastEventTime = now;
  }

  handleKeyup(e) {
    if (!this.isCollecting) return;
    this.currentWindow.keyupCount++;
    this.keystrokes.push({
      type: "keyup",
      keyCode: e.keyCode,
      time: Date.now() - this.sessionStart,
    });
  }

  handleMouseMove(e) {
    if (!this.isCollecting) return;
    const now = Date.now();

    // Sample at most every 100ms (mouse moves can fire 100s/sec)
    if (now - this.lastMouseSample < 100) return;
    this.lastMouseSample = now;

    this.currentWindow.mouseMoveCount++;

    if (this.lastMousePos) {
      const dx = e.clientX - this.lastMousePos.x;
      const dy = e.clientY - this.lastMousePos.y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      const dt = (now - this.lastMousePos.time) / 1000; // seconds

      if (dt > 0 && dist > 2) {  // Ignore tiny movements
        const velocity = dist / dt;  // px/sec
        this.currentWindow.mouseVelocitySum += velocity;
        this.currentWindow.mouseVelocityCount++;

        // Jitter = average per-sample displacement (smaller dist = more jittery)
        // Better measure: angular variability — but for simplicity use displacement
        if (dist < 30) {  // Small movements = potential jitter
          this.currentWindow.mouseJitterSum += dist;
          this.currentWindow.mouseJitterCount++;
        }
      }
    }

    this.lastMousePos = { x: e.clientX, y: e.clientY, time: now };

    this.mouseMoves.push({
      x: e.clientX,
      y: e.clientY,
      time: now - this.sessionStart,
    });

    this.lastEventTime = now;
  }

  handleClick(e) {
    if (!this.isCollecting) return;
    const now = Date.now();
    this.currentWindow.clickCount++;
    this.clicks.push({
      x: e.clientX,
      y: e.clientY,
      time: now - this.sessionStart,
      target: e.target.tagName || "UNKNOWN",
    });
    this.lastEventTime = now;
  }

  /* ──────── FEATURE COMPUTATION ──────── */

  flushWindow() {
    if (!this.isCollecting) return;
    const now = Date.now();
    const duration = (now - this.windowStartTime) / 1000;  // seconds
    if (duration < 5) return; // Skip very short windows

    const w = this.currentWindow;

    // 7 CORE FEATURES
    const cpm = (w.keydownCount / duration) * 60;
    const ikiMean = w.ikiCount > 0 ? w.ikiSum / w.ikiCount : 0;
    const ikiVar = w.ikiCount > 1
      ? (w.ikiSqSum - (w.ikiSum * w.ikiSum / w.ikiCount)) / (w.ikiCount - 1)
      : 0;
    const errorRate = w.keydownCount > 0
      ? w.backspaceCount / w.keydownCount
      : 0;
    const mouseVelocity = w.mouseVelocityCount > 0
      ? w.mouseVelocitySum / w.mouseVelocityCount
      : 0;
    const jitterIndex = w.mouseJitterCount > 0
      ? w.mouseJitterSum / w.mouseJitterCount
      : 0;
    const pauseFrequency = w.pauseCount;

    const featureWindow = {
      participantId: this.participantId,
      windowStartMs: this.windowStartTime - this.sessionStart,
      duration_sec: parseFloat(duration.toFixed(2)),
      phase: this.currentPhase,
      // 7 features
      cpm: parseFloat(cpm.toFixed(2)),
      iki_mean_ms: parseFloat(ikiMean.toFixed(2)),
      iki_variance: parseFloat(ikiVar.toFixed(2)),
      error_rate: parseFloat(errorRate.toFixed(4)),
      mouse_velocity_pxs: parseFloat(mouseVelocity.toFixed(2)),
      jitter_index_px: parseFloat(jitterIndex.toFixed(2)),
      pause_frequency: pauseFrequency,
      // Raw counts (for verification)
      keystroke_count: w.keydownCount,
      backspace_count: w.backspaceCount,
      mouse_move_count: w.mouseMoveCount,
      click_count: w.clickCount,
      // Will be filled later
      vata_pct: 0,
      pitta_pct: 0,
      kapha_pct: 0,
      time_on_task_minutes: 0,
      fatigue_label: "",
    };

    this.windows.push(featureWindow);

    // Reset for next window
    this.currentWindow = this.newEmptyWindow();
    this.windowStartTime = now;

    // Update UI counter if exists
    const counter = document.getElementById("featureCount");
    if (counter) {
      counter.textContent = `${this.windows.length} windows captured`;
    }
  }

  /* ──────── PRAKRITI INJECTION ──────── */

  injectPrakriti(vataPct, pittaPct, kaphaPct) {
    this.windows.forEach(w => {
      w.vata_pct = vataPct;
      w.pitta_pct = pittaPct;
      w.kapha_pct = kaphaPct;
    });
  }

  /* ──────── LABEL INJECTION (time-on-task) ──────── */

  injectLabels() {
    // First task window = time 0 = Alert
    // Use session start as t=0
    this.windows.forEach(w => {
      const minutesIntoSession = (w.windowStartMs / 1000) / 60;
      w.time_on_task_minutes = parseFloat(minutesIntoSession.toFixed(2));

      // Time-on-task label: based on minutes into typing tasks
      if (minutesIntoSession < 30) w.fatigue_label = "Alert";
      else if (minutesIntoSession < 60) w.fatigue_label = "Moderate";
      else w.fatigue_label = "Fatigued";
    });
  }

  /* ──────── EXPORT ──────── */

  exportCSV() {
    if (this.windows.length === 0) return "";
    const headers = Object.keys(this.windows[0]);
    const rows = [headers.join(",")];
    this.windows.forEach(w => {
      const row = headers.map(h => {
        const val = w[h];
        if (typeof val === "string") return `"${val.replace(/"/g, '""')}"`;
        return val;
      });
      rows.push(row.join(","));
    });
    return rows.join("\n");
  }

  exportJSON(extra = {}) {
    return JSON.stringify({
      meta: {
        participant_id: this.participantId,
        session_start: new Date(this.sessionStart).toISOString(),
        session_duration_min: ((Date.now() - this.sessionStart) / 60000).toFixed(2),
        total_windows: this.windows.length,
        feature_collector_version: "1.0.0",
        ...extra,
      },
      feature_windows: this.windows,
      raw_event_summary: {
        total_keystrokes: this.keystrokes.length,
        total_mouse_moves: this.mouseMoves.length,
        total_clicks: this.clicks.length,
      },
    }, null, 2);
  }

  getStats() {
    return {
      windows: this.windows.length,
      keystrokes: this.keystrokes.length,
      mouseMoves: this.mouseMoves.length,
      clicks: this.clicks.length,
      elapsedMin: ((Date.now() - this.sessionStart) / 60000).toFixed(1),
    };
  }
}

const collector = new SignalCollector();
