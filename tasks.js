/* ═══════════════════════════════════════════════════════════════════════
   PrakritiSense Tasks Module
   ──────────────────────────
   Contains:
   - 10 Prakriti quiz questions (with the 3 fixes applied)
   - Copy typing source text (Wikipedia neutral content)
   - Free typing prompts
   - Stroop word list with colour mapping
   - NASA-TLX dimensions
   ═══════════════════════════════════════════════════════════════════════ */

/* ──────────────────────────────────────────────────────────────────────
   PRAKRITI QUIZ — 10 questions, with the 3 fixes applied
   Fixed: Q8 (was duplicate of Q3), Q9 (option C shortened), Q10 (was circular)
   ────────────────────────────────────────────────────────────────────── */

const PRAKRITI_QUESTIONS = [
  {
    id: 1,
    text: "How would you describe your body frame and physical build?",
    dosha: ["V", "P", "K"],
    options: [
      "Thin, light frame — I find it hard to gain weight",
      "Medium build — fairly athletic, moderate muscle",
      "Broad, sturdy frame — I gain weight easily"
    ]
  },
  {
    id: 2,
    text: "How would you describe your skin texture and complexion?",
    dosha: ["V", "P", "K"],
    options: [
      "Dry, rough, or flaky — prone to cracking",
      "Warm, reddish, or prone to rashes and acne",
      "Smooth, oily, pale, or cool to the touch"
    ]
  },
  {
    id: 3,
    text: "How is your digestion typically?",
    dosha: ["V", "P", "K"],
    options: [
      "Irregular — sometimes strong, sometimes poor",
      "Strong — I get very hungry and irritable if I skip meals",
      "Slow but steady — I can skip meals without much discomfort"
    ]
  },
  {
    id: 4,
    text: "How do you typically sleep?",
    dosha: ["V", "P", "K"],
    options: [
      "Light sleeper — I wake easily and often feel under-rested",
      "Moderate — I fall asleep quickly but can be disturbed",
      "Deep, heavy sleeper — I love long sleep and feel groggy when woken"
    ]
  },
  {
    id: 5,
    text: "How would you describe your mental activity and thought patterns?",
    dosha: ["V", "P", "K"],
    options: [
      "Fast, creative, restless — many ideas, easily distracted",
      "Focused, sharp, and goal-oriented — can be intense or critical",
      "Calm, steady, methodical — slow to decide but loyal to decisions"
    ]
  },
  {
    id: 6,
    text: "How do you typically respond when under stress or pressure?",
    dosha: ["V", "P", "K"],
    options: [
      "Anxiety, worry, or feeling scattered and overwhelmed",
      "Anger, frustration, or strong criticism — I get irritable",
      "Withdrawal, stubbornness, or emotional eating — I go quiet"
    ]
  },
  {
    id: 7,
    text: "How would you describe your energy levels throughout the day?",
    dosha: ["V", "P", "K"],
    options: [
      "Bursts of energy followed by sudden fatigue — variable",
      "Strong and sustained energy — I push hard until I crash",
      "Slow to start but steady once going — consistent but low-key"
    ]
  },
  {
    /* FIXED Q8 — was duplicate of Q3 (digestion). Now: body temperature. */
    id: 8,
    text: "How does your body usually feel temperature-wise?",
    dosha: ["V", "P", "K"],
    options: [
      "Often cold — cold hands and feet, prefer warm environments",
      "Warm or hot — I run hot, dislike heat, sweat easily",
      "Cool and comfortable — adapt easily, rarely too hot or cold"
    ]
  },
  {
    /* FIXED Q9 — option C shortened for parity with A/B */
    id: 9,
    text: "How would you describe your memory and learning style?",
    dosha: ["V", "P", "K"],
    options: [
      "Quick to learn, quick to forget — grasp fast but lose detail",
      "Sharp memory with strong focus — I remember what matters",
      "Slow to learn but very strong long-term memory"
    ]
  },
  {
    /* FIXED Q10 — was circular (asked about typing while measuring typing).
       Now: speech style, a classical Dosha indicator independent of typing. */
    id: 10,
    text: "How do you usually speak and converse with others?",
    dosha: ["V", "P", "K"],
    options: [
      "Fast and talkative — I jump between topics, lots of words",
      "Clear and articulate — precise, to the point, persuasive",
      "Slow and measured — I say less, but with weight and warmth"
    ]
  }
];

/* ──────────────────────────────────────────────────────────────────────
   COPY TYPING TEXT
   Neutral, factual, ~250 words from Wikipedia. Same for every participant.
   ────────────────────────────────────────────────────────────────────── */

const COPY_TEXT = `The history of typewriters spans nearly two centuries, beginning with experimental machines in the eighteenth century and reaching their commercial peak in the early twentieth. The first practical typewriter was developed by Christopher Latham Sholes in eighteen sixty-eight, working with collaborators Carlos Glidden and Samuel Soule in Milwaukee, Wisconsin. Their machine became the basis for the Remington Number One, sold by the firearms manufacturer E. Remington and Sons starting in eighteen seventy-four.

Early typewriters were mechanical marvels, using levers and a complex linkage to strike inked ribbons against paper. The arrangement of keys, which we now call the QWERTY layout, was designed not for efficiency but to prevent the mechanical arms from jamming when adjacent keys were struck in rapid succession. The layout has persisted for more than one hundred and fifty years, surviving multiple attempts to replace it with theoretically superior designs.

Throughout the twentieth century, typewriters became essential tools of office work, journalism, and literature. Many famous authors, including Ernest Hemingway, Agatha Christie, and Mark Twain, composed their works on typewriters. The introduction of electric typewriters in the nineteen sixties improved typing speed and reduced fatigue, but the fundamental design remained largely unchanged until the personal computer began displacing typewriters in the nineteen eighties.

Today, the typewriter is largely a nostalgic artifact, but its influence persists. The QWERTY layout dominates digital keyboards worldwide, and the rhythmic clack of typing remains a familiar sound in offices and homes everywhere.`;

/* ──────────────────────────────────────────────────────────────────────
   FREE TYPING PROMPT
   Open-ended, neutral, encourages sustained typing.
   ────────────────────────────────────────────────────────────────────── */

const FREE_PROMPT = `Please write a response to the following prompt. Aim for at least 300 words. Write naturally — there are no right or wrong answers, and your text content will not be saved.

Prompt: Describe a typical workday or study day in your life. Include details about how you spend your morning, your most productive time, what challenges you face, what helps you focus, and how you wind down. Reflect on whether your routine has changed in recent months and what you would change about it if you could.`;

/* ──────────────────────────────────────────────────────────────────────
   STROOP TASK
   Word and ink colour combinations — incongruent trials increase load.
   ────────────────────────────────────────────────────────────────────── */

const STROOP_COLORS = {
  RED:    { hex: "#E24B4A", key: "r" },
  GREEN:  { hex: "#1D9E75", key: "g" },
  BLUE:   { hex: "#378ADD", key: "b" },
  YELLOW: { hex: "#BA7517", key: "y" },
  PURPLE: { hex: "#7F77DD", key: "p" },
  ORANGE: { hex: "#D85A30", key: "o" },
};

const STROOP_WORDS = ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE", "ORANGE"];

function generateStroopTrials(count = 30) {
  const trials = [];
  for (let i = 0; i < count; i++) {
    const word = STROOP_WORDS[Math.floor(Math.random() * STROOP_WORDS.length)];
    let inkColor;
    if (Math.random() < 0.3) {
      inkColor = word;  // 30% congruent (easy)
    } else {
      do {
        inkColor = STROOP_WORDS[Math.floor(Math.random() * STROOP_WORDS.length)];
      } while (inkColor === word);  // 70% incongruent (hard)
    }
    trials.push({
      word,
      inkColor,
      correctKey: STROOP_COLORS[inkColor].key,
      inkHex: STROOP_COLORS[inkColor].hex,
      congruent: word === inkColor,
    });
  }
  return trials;
}

/* ──────────────────────────────────────────────────────────────────────
   NASA-TLX DIMENSIONS (6 standard subscales)
   ────────────────────────────────────────────────────────────────────── */

const NASA_TLX_DIMENSIONS = [
  { key: "mental_demand",   name: "Mental Demand",   desc: "How mentally demanding was the session?",            low: "Very low", high: "Very high" },
  { key: "physical_demand", name: "Physical Demand", desc: "How physically demanding was the session?",          low: "Very low", high: "Very high" },
  { key: "temporal_demand", name: "Temporal Demand", desc: "How hurried or rushed was the pace?",                low: "Very low", high: "Very high" },
  { key: "performance",     name: "Performance",     desc: "How successful were you in accomplishing the tasks?", low: "Perfect",  high: "Failed" },
  { key: "effort",          name: "Effort",          desc: "How hard did you have to work?",                     low: "Very low", high: "Very high" },
  { key: "frustration",     name: "Frustration",     desc: "How insecure, stressed, or annoyed were you?",        low: "Very low", high: "Very high" },
];

/* ──────────────────────────────────────────────────────────────────────
   DOSHA INFORMATION (displayed in result page)
   ────────────────────────────────────────────────────────────────────── */

const DOSHA_INFO = {
  V: {
    name: "Vata (Air + Space)",
    color: "#4A90D9",
    bg: "#E8F1FB",
    dark: "#0C447C",
    desc: "Fast, creative, and scattered. Typing tends to be quick with variable rhythm and higher error rates. Fatigue sets in earliest — typically within 60–90 minutes."
  },
  P: {
    name: "Pitta (Fire + Water)",
    color: "#E05C5C",
    bg: "#FBEAEA",
    dark: "#4A1B0C",
    desc: "Focused, intense, and precise. Typing is moderate-speed with low errors and consistency. Fatigue appears in 90–150 minutes as increased response time and micro-corrections."
  },
  K: {
    name: "Kapha (Earth + Water)",
    color: "#2EAA70",
    bg: "#E5F5ED",
    dark: "#04342C",
    desc: "Calm, steady, and methodical. Typing is deliberate and slow with very low errors. Fatigue shows as gradual inter-key interval lengthening after 150+ minutes."
  }
};

/* ──────────────────────────────────────────────────────────────────────
   PRAKRITI SCORING FUNCTION
   ────────────────────────────────────────────────────────────────────── */

function scorePrakriti(answers) {
  const scores = { V: 0, P: 0, K: 0 };
  answers.forEach((ans, qi) => {
    if (ans === null) return;
    const dosha = PRAKRITI_QUESTIONS[qi].dosha[ans];
    scores[dosha]++;
  });

  const total = 10;
  return {
    vataPct: Math.round((scores.V / total) * 100),
    pittaPct: Math.round((scores.P / total) * 100),
    kaphaPct: Math.round((scores.K / total) * 100),
    dominant: scores.V >= scores.P && scores.V >= scores.K ? "V" :
              scores.P >= scores.K ? "P" : "K",
    rawScores: scores,
  };
}