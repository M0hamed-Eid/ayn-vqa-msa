# M3 -- First real pipeline: design walkthrough

The "why", file by file, for M3: the first end-to-end cascade that
actually looks at the image and audio to answer the question. Companion
to [`docs/M0_DATA_AUDIT.md`](M0_DATA_AUDIT.md),
[`docs/M1_BASELINES.md`](M1_BASELINES.md),
[`docs/M2_ASR_BENCH.md`](M2_ASR_BENCH.md), and
[`../../AynVQA-ArabicNLP26/project_analysis_and_plan.md`](../../AynVQA-ArabicNLP26/project_analysis_and_plan.md)
(§11, M3). **Task 1a only** -- no Task 1b/1c anywhere.

## Model choice: local Qwen2.5-VL-7B via Ollama, not an API

M2 asked which VLM/LLM to build against; the answer was Qwen2.5-VL-7B run
locally through Ollama, for both transcript parsing and image answering --
no API key, no per-call cost, no rate limit. This only works because the
machine turned out to have an NVIDIA RTX 2000 Ada (16 GB VRAM) that hadn't
been used yet (M2's Whisper bench ran on CPU because a GPU wasn't
confirmed at the time). `ollama pull qwen2.5vl:7b` (~6 GB) and Ollama's
own GPU detection handled the rest -- `ollama ps` confirmed 100% GPU
utilization during the real run below.

## `src/ayn_vqa/ollama_client.py`

One class, one method (`chat`), shared by both `stages/parse.py` and
`stages/select_vlm.py` -- there is exactly one place that knows Ollama's
`/api/chat` request shape (model, messages, optional base64 image,
optional JSON-schema `format`). Verified against a live Ollama instance
(the already-installed `llama3.1:8b`) before writing any stage code
against it, the same "confirm the real contract before coding to it"
discipline used for the Fanar client in M2.

## `src/ayn_vqa/stages/parse.py`

`OllamaTranscriptParser` turns raw ASR text into `{question, option_0,
option_1, option_2}` using Ollama's **JSON-schema-constrained decoding** --
the model is not asked nicely to return JSON, it structurally cannot
return anything else. M2's bench found the phrase "الخيارات هي" ("the
options are") in ~80% of sampled transcripts; the prompt mentions it as a
hint, not a rule the code depends on, since the other ~20% still need to
parse correctly.

## `src/ayn_vqa/stages/select_vlm.py`

`OllamaJointMCQSelector` shows the model the image and all three options
**together** in one prompt (not three independent yes/no calls) -- the
project's plan doc argues independent per-option scoring throws away the
contrastive signal MCQ distractors depend on ("more plausible than the
other two" isn't answerable one option at a time). The predicted index is
schema-constrained to the literal enum `{0, 1, 2}`.

This directly neutralizes a specific, named weakness in the official
Qwen2.5-Omni baseline: its answer parser is `re.search(r"[012]", ...)`
with a **silent fallback to 0** on anything that doesn't match --
including a perfectly correct answer phrased in Arabic digits or words.
The project's plan doc suspected a nontrivial share of the baseline's
0.398 MSA score was parse failure, not reasoning failure. Constrained
decoding doesn't parse around that bug better; it removes the
free-text-output precondition the bug needs to exist at all.

## `src/ayn_vqa/error_analysis.py`

A pure function over already-cached stage output (never a fresh model
call): joins transcript + parse + prediction + gold label into one table,
plus per-country and per-category accuracy breakdowns. This is what turns
"the model got 41/50" into "here are the exact 9 items it missed and
which country/category they're in" -- the habit of *reading failures by
hand* the project's plan doc calls out as the actual skill this milestone
teaches, not just a number to report.

## `src/ayn_vqa/run_pipeline.py`

The `aynvqa-run` cascade: ASR -> parse -> select -> submit -> score ->
error analysis, each stage cached under `artifacts/<split>/<stage>/`
(M2's `artifacts.py`). Concretely, this pipeline's first real run cost
**zero new Whisper calls** for its 50-item sample, because M2's ASR bench
had already cached `whisper-medium` transcripts for that exact
seed+sample -- the caching design paying for itself one milestone later,
not hypothetically.

Two fallback paths, both logged and counted (never silent):
- A record whose transcript failed or was empty skips straight to a
  `fallback: parse failed (...)` prediction (index 0) -- never sent to the
  VLM with an empty question.
- A record whose parse failed, or whose VLM call itself errored, gets the
  same fallback-index-0 treatment, distinguishable in the logs and in
  `prediction.raw` (`"fallback: ..."` vs. `"error: ..."`) from a real
  model answer.

**Submission is manual, by design.** `run_pipeline` produces
`prediction.csv` and stops there. Actually uploading it to Codabench --
using one of a limited number of daily submission attempts, under your
account -- is not something this code does automatically; that decision
and action are yours.

## Real result: MSA `dev`, full 500 items

```
accuracy           0.7160  <-- RANKING
balanced_accuracy  0.7141
macro_f1           0.7143
```

**358/500 correct.** Vs. the official Qwen2.5-Omni baseline's published
**0.398** on MSA and **0.664** on the English track (same images, same
labels, from the same 3B end-to-end model) -- this cascade beats the MSA
baseline by **+32 points** and the *English* baseline by **+5 points**,
on the harder (MSA) audio track. Format check passed; 500/500 ASR
succeeded, 500/500 parsed, 500/500 predicted (0 fallbacks, 0 VLM-call
errors) -- every stage ran clean end to end at full scale, not just on a
sample. Runtime: ~2h4m end to end (450 new Whisper-medium transcriptions
on CPU dominate that; the 1,000 Ollama calls on GPU are the fast part).

A seeded 50-item sample run first (kept, not overwritten, in
`experiments.md`) had measured 82% -- inside the ~71-93% 95% confidence
interval a sample that size implies, but on the optimistic side of the
full number. Worth internalizing for every later milestone: a 50-item
bench is the right tool for *fast, directional* comparisons (which
Whisper size, which prompt), and the wrong tool for *the* number you'd
report -- that's what the full-split run is for, exactly as it played out
here.

Per-country accuracy ranges from Palestine (58.6%, n=29) to Sudan (93.1%,
n=29) -- a real spread across the uniformly-sized 18 countries, not
noise from an unbalanced sample size (every country has 29-30 dev
items). Per-category, the weakest is "Objects, Materials & Clothing"
(58.7%) and "Food & Cooking" (59.4%) -- both categories built on
fine-grained visual discrimination between similar-looking items
(clothing styles, dish types) rather than scene-level recognition or
landmark identification, exactly the failure mode the project's plan doc
predicted generic VLMs would have on this benchmark's culturally-specific
distinctions. "History, Geography & National Identity" (83.6%) and
"Vehicles & Transportation" (82.9%) are the strongest -- categories where
a distinctive, nameable object (a landmark, a vehicle model) is more
often the deciding cue than a subtle cultural detail.

**Clean error attribution, the habit this milestone is meant to build:**
across all 500 items, `asr_error` and `parse_error` are both **zero** --
every one of the 142 wrong predictions is a genuine VLM visual-reasoning
miss, not an upstream cascade failure. That's a real, checked fact (not
an assumption): the error table joins transcript, parse, and prediction
per item specifically so this question -- "which stage actually killed
each wrong item" -- has an answer instead of a guess. It means the
highest-leverage next step is improving the *visual/cultural*
discrimination itself (few-shot exemplars from `train`, an OCR channel,
permutation ensembling) rather than chasing ASR or parsing further.

## What's still true from M2 that carries forward

Task 1a MSA only. `FanarAuraASR`/`OpenAITranscribeASR` from M2 remain
ready-but-unexercised. No OCR channel yet (§9 opportunity #5 in the plan
doc) -- signage/text-in-image reading is still whatever Qwen2.5-VL picks
up on its own, not a dedicated evidence channel. No permutation
ensembling for position bias (M4). No few-shot exemplars from the 3,000
labeled `train` items (M4/M8).
