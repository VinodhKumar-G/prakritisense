-- ═══════════════════════════════════════════════════════════════════
-- PrakritiSense — Supabase schema
-- Run this entire block in Supabase → SQL Editor → New Query → Run
-- ═══════════════════════════════════════════════════════════════════

-- Table 1: One row per participant session (Prakriti + NASA-TLX)
create table if not exists sessions (
  id                  uuid default gen_random_uuid() primary key,
  participant_id      text not null,
  created_at          timestamptz default now(),

  -- Prakriti profile
  vata_pct            integer,
  pitta_pct           integer,
  kapha_pct           integer,
  dominant_dosha      text,

  -- NASA-TLX subscales (0–100)
  tlx_mental_demand   integer,
  tlx_physical_demand integer,
  tlx_temporal_demand integer,
  tlx_performance     integer,
  tlx_effort          integer,
  tlx_frustration     integer,

  -- Stroop task summary
  stroop_correct      integer,
  stroop_wrong        integer,
  stroop_avg_rt_ms    numeric,

  -- Mouse task summary
  mouse_hits          integer,
  mouse_misses        integer,
  mouse_avg_rt_ms     numeric,
  mouse_avg_jitter_px numeric,

  -- Session metadata
  session_duration_min numeric,
  total_windows        integer
);

-- Table 2: One row per 60-second feature window (ML input data)
create table if not exists feature_windows (
  id                   bigserial primary key,
  session_id           uuid references sessions(id) on delete cascade,
  participant_id       text not null,
  created_at           timestamptz default now(),

  -- Timing
  window_start_ms      bigint,
  duration_sec         numeric,
  phase                text,
  time_on_task_minutes numeric,
  fatigue_label        text,      -- Alert / Moderate / Fatigued

  -- The 7 core ML features
  cpm                  numeric,   -- Typing speed (chars/min)
  iki_mean_ms          numeric,   -- Inter-key interval mean
  iki_variance         numeric,   -- Inter-key interval variance
  error_rate           numeric,   -- Backspaces / total keystrokes
  mouse_velocity_pxs   numeric,   -- Mouse speed (px/sec)
  jitter_index_px      numeric,   -- Mouse jitter
  pause_frequency      integer,   -- Pauses > 2s

  -- Prakriti conditioning features
  vata_pct             integer,
  pitta_pct            integer,
  kapha_pct            integer,

  -- Raw counts (for QA / verification)
  keystroke_count      integer,
  backspace_count      integer,
  mouse_move_count     integer,
  click_count          integer
);

-- ── Allow browser to read/write without auth (for research use) ──────
-- Go to Supabase → Authentication → Policies and enable RLS,
-- then add these policies so your app can insert without login:

alter table sessions       enable row level security;
alter table feature_windows enable row level security;

-- Insert policy: anyone can insert (participants saving their data)
create policy "Allow public insert on sessions"
  on sessions for insert
  with check (true);

create policy "Allow public insert on feature_windows"
  on feature_windows for insert
  with check (true);

-- Select policy: anyone can read (so you can query from Python too)
create policy "Allow public select on sessions"
  on sessions for select
  using (true);

create policy "Allow public select on feature_windows"
  on feature_windows for select
  using (true);

-- ── Handy views for your Python analysis ────────────────────────────

-- Full ML dataset: feature windows joined with session metadata
create or replace view ml_dataset as
select
  fw.*,
  s.dominant_dosha,
  s.tlx_mental_demand,
  s.tlx_effort,
  s.tlx_frustration,
  s.session_duration_min
from feature_windows fw
left join sessions s on fw.session_id = s.id;

-- Quick session summary
create or replace view session_summary as
select
  participant_id,
  dominant_dosha,
  vata_pct, pitta_pct, kapha_pct,
  tlx_mental_demand, tlx_effort, tlx_frustration,
  total_windows,
  session_duration_min,
  created_at
from sessions
order by created_at desc;