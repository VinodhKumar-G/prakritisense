/* ═══════════════════════════════════════════════════════════════════════
   PrakritiSense — Supabase Database Layer (db.js)
   ────────────────────────────────────────────────
   Saves session + all feature windows directly to Supabase PostgreSQL.
   No backend server needed — Supabase client runs in the browser.

   SETUP (one time):
   1. Create a project at supabase.com (free)
   2. Run supabase_schema.sql in Supabase SQL Editor
   3. Get your project URL + anon key from Settings → API
   4. Paste them in SUPABASE_URL and SUPABASE_ANON_KEY below
   ═══════════════════════════════════════════════════════════════════════ */

// ── CONFIGURE THESE TWO VALUES ────────────────────────────────────────
const SUPABASE_URL      = "https://gxodetmbxbqlzkpwyktm.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_Sb-z0yjKTN8nEJE07Ky2UA_4Y41TvOu";
// ──────────────────────────────────────────────────────────────────────

class PrakritiDB {
  constructor() {
    this.client = null;
    this.ready  = false;
    this.sessionId = null;
  }

  /* ── INITIALISE ──────────────────────────────────────────────────── */

  async init() {
    if (this.ready) return true;

    // Guard: don't run if placeholders are still in place
    if (SUPABASE_URL.includes("YOUR_PROJECT_ID")) {
      console.warn("[PrakritiDB] Supabase not configured — running in offline mode. Data will only download locally.");
      return false;
    }

    try {
      const { createClient } = supabase; // loaded from CDN in index.html
      this.client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
      this.ready  = true;
      console.log("[PrakritiDB] Connected to Supabase ✓");
      return true;
    } catch (err) {
      console.error("[PrakritiDB] Failed to init Supabase:", err);
      return false;
    }
  }

  /* ── SAVE SESSION METADATA ───────────────────────────────────────── */

  async saveSession(participantId, prakritiResult, tlxValues, stroopResults, mouseData, collectorStats) {
    if (!this.ready) return null;

    const stroopCorrect = stroopResults.filter(r => r.correct).length;
    const stroopWrong   = stroopResults.filter(r => !r.correct).length;
    const stroopAvgRT   = stroopResults.length > 0
      ? stroopResults.reduce((s, r) => s + r.rt_ms, 0) / stroopResults.length
      : 0;

    const sessionRow = {
      participant_id:      participantId,
      vata_pct:            prakritiResult.vataPct,
      pitta_pct:           prakritiResult.pittaPct,
      kapha_pct:           prakritiResult.kaphaPct,
      dominant_dosha:      prakritiResult.dominant,

      tlx_mental_demand:   tlxValues.mental_demand   || 0,
      tlx_physical_demand: tlxValues.physical_demand || 0,
      tlx_temporal_demand: tlxValues.temporal_demand || 0,
      tlx_performance:     tlxValues.performance      || 0,
      tlx_effort:          tlxValues.effort           || 0,
      tlx_frustration:     tlxValues.frustration      || 0,

      stroop_correct:      stroopCorrect,
      stroop_wrong:        stroopWrong,
      stroop_avg_rt_ms:    Math.round(stroopAvgRT),

      mouse_hits:          mouseData.hits || 0,
      mouse_misses:        mouseData.misses || 0,
      mouse_avg_rt_ms:     mouseData.hits > 0 ? mouseData.rtSum / mouseData.hits : 0,
      mouse_avg_jitter_px: mouseData.samples > 0 ? mouseData.jitterSum / mouseData.samples : 0,

      session_duration_min: parseFloat(collectorStats.elapsedMin) || 0,
      total_windows:        collectorStats.windows || 0,
    };

    const { data, error } = await this.client
      .from("sessions")
      .insert(sessionRow)
      .select("id")
      .single();

    if (error) {
      console.error("[PrakritiDB] saveSession error:", error.message);
      return null;
    }

    this.sessionId = data.id;
    console.log("[PrakritiDB] Session saved ✓  id:", this.sessionId);
    return this.sessionId;
  }

  /* ── SAVE FEATURE WINDOWS (batch) ────────────────────────────────── */

  async saveFeatureWindows(windows) {
    if (!this.ready || !this.sessionId) return false;
    if (!windows || windows.length === 0) return true;

    // Map collector output fields → database columns
    const rows = windows.map(w => ({
      session_id:           this.sessionId,
      participant_id:       w.participantId,
      window_start_ms:      w.windowStartMs,
      duration_sec:         w.duration_sec,
      phase:                w.phase,
      time_on_task_minutes: w.time_on_task_minutes,
      fatigue_label:        w.fatigue_label,

      cpm:                  w.cpm,
      iki_mean_ms:          w.iki_mean_ms,
      iki_variance:         w.iki_variance,
      error_rate:           w.error_rate,
      mouse_velocity_pxs:   w.mouse_velocity_pxs,
      jitter_index_px:      w.jitter_index_px,
      pause_frequency:      w.pause_frequency,

      vata_pct:             w.vata_pct,
      pitta_pct:            w.pitta_pct,
      kapha_pct:            w.kapha_pct,

      keystroke_count:      w.keystroke_count,
      backspace_count:      w.backspace_count,
      mouse_move_count:     w.mouse_move_count,
      click_count:          w.click_count,
    }));

    // Batch insert in chunks of 50 (Supabase recommends <500 rows/call)
    const CHUNK = 50;
    for (let i = 0; i < rows.length; i += CHUNK) {
      const chunk = rows.slice(i, i + CHUNK);
      const { error } = await this.client
        .from("feature_windows")
        .insert(chunk);

      if (error) {
        console.error(`[PrakritiDB] saveFeatureWindows error (chunk ${i}):`, error.message);
        return false;
      }
    }

    console.log(`[PrakritiDB] ${rows.length} feature windows saved ✓`);
    return true;
  }

  /* ── FULL SAVE (call this once at session end) ────────────────────── */

  async saveAll(participantId, prakritiResult, tlxValues, stroopResults, mouseData, collectorStats, featureWindows) {
    if (!this.ready) {
      console.warn("[PrakritiDB] Not connected — skipping cloud save.");
      return { success: false, reason: "not_connected" };
    }

    const sessionId = await this.saveSession(
      participantId, prakritiResult, tlxValues,
      stroopResults, mouseData, collectorStats
    );

    if (!sessionId) {
      return { success: false, reason: "session_save_failed" };
    }

    const windowsOk = await this.saveFeatureWindows(featureWindows);

    return {
      success: windowsOk,
      sessionId,
      windowsSaved: featureWindows.length,
    };
  }

  /* ── STATUS CHECK (for UI feedback) ─────────────────────────────── */

  getStatus() {
    if (SUPABASE_URL.includes("YOUR_PROJECT_ID")) return "not_configured";
    if (!this.ready)     return "not_connected";
    if (!this.sessionId) return "connected_no_session";
    return "saved";
  }
}

// Singleton — used by app.js
const db = new PrakritiDB();