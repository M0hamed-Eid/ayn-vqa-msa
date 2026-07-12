# M4 -- Pipeline robustness: design walkthrough and full-scale evaluation

The "why", file by file, for M4, plus the full 500-item ablation study that
validates it. Companion to [`docs/M0_DATA_AUDIT.md`](M0_DATA_AUDIT.md),
[`docs/M1_BASELINES.md`](M1_BASELINES.md), [`docs/M2_ASR_BENCH.md`](M2_ASR_BENCH.md),
and [`docs/M3_PIPELINE.md`](M3_PIPELINE.md). **Task 1a, MSA track only.**

## Why this milestone exists

M3's error analysis reported `asr_error` and `parse_error` as zero across
all 142 wrong dev predictions, and concluded every wrong prediction was a
genuine VLM visual-reasoning miss. That conclusion was wrong, but not
because M3 miscounted -- `asr_error`/`parse_error` only catch *hard*
failures (an exception, an empty transcript). A pre-implementation analysis
of the 142 wrong items (hand-reading every transcript and, for the 73 that
looked textually fair, every image) found that the parser's JSON schema
forces exactly three non-empty options every time, so when a transcript
doesn't actually contain three cleanly recoverable ones, it fabricates
content instead of failing: an empty string, the literal preamble
"الخيارات هي" ("the options are") captured as if it were an option, a
duplicated option, or placeholder text. **69 of the 142 (48.6%) were this
kind of silent pipeline corruption, not visual-reasoning failures.**

## What was implemented

- **`src/ayn_vqa/option_quality.py`** -- `check_option_quality()` and
  `check_repetition_loop()`, pure functions (no Protocol -- one rule-based
  implementation, not several swappable backends) formalizing the
  heuristics hand-verified against all 142 original errors: empty/near-empty
  options, leftover preamble text, fabricated "no answer" placeholders,
  duplicate options, one option a substring of another, and ASR
  repetition-loop hallucinations (e.g. a single letter repeated 30+ times).
- **`error_analysis.py`** -- `option_quality_ok`/`option_quality_reasons`
  columns on every row, plus `summarize_option_quality()`. Every future
  pipeline run now reports the pipeline-vs-genuine split automatically.
- **`run_pipeline.py`** -- a 3-step repair escalation ladder, gated by
  `repair_enabled` (`Settings.repair_enabled`, default `True`;
  `--no-repair-enabled`/`AYNVQA_REPAIR_ENABLED=false` for ablation runs):
  1. Re-parse the *same* transcript with a stricter prompt (cheap, no new
     ASR call).
  2. If still degenerate, re-transcribe with a stronger ASR backend
     (Whisper large-v3) and re-parse that.
  3. If still degenerate, give up -- the item proceeds to selection with
     whatever the original parse produced, exactly as before repair existed.

  Both steps reuse the existing `_run_asr_stage`/`_run_parse_stage` caching
  under distinct config keys, so a second run with the same config replays
  free. Repair-on and repair-off runs use different `selector_config_key`s
  (`...__repair` suffix) so an ablation comparison can never silently read
  the other condition's cached predictions.
- **`stages/parse.py`** -- a stricter repair-retry prompt (`_STRICT_PROMPT`),
  selected via `OllamaTranscriptParser.parse(..., strict=True)`. **No schema
  change** -- still exactly `{question, option_0, option_1, option_2}`, per
  design decision to keep the validation layer independent of any future
  confidence-estimation change.
- **`stages/asr.py`** -- `_register_windows_cuda_dll_dirs()`. Getting
  Whisper large-v3 onto the machine's GPU wasn't just a config flag:
  `faster-whisper`/`ctranslate2` need `cublas64_12.dll`/cuDNN at runtime,
  neither was installed, and unlike PyTorch, `ctranslate2` doesn't
  auto-register pip-installed CUDA wheels on Windows. Fixed by adding
  `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` as an optional `gpu` dependency
  group (not core -- would break installs on non-NVIDIA machines) plus an
  `os.add_dll_directory()` call, gated to `device="cuda"` and Windows only.
- 106 tests (up from 82), ruff clean, mypy strict clean.

## Benchmark methodology

Two full runs over `dev`/`msa`, all 500 records, identical settings except
`repair_enabled`, same machine, same caches wherever valid:

- `aynvqa-run --split dev --language msa --no-repair-enabled` -- the
  baseline. Whisper-medium (CPU), `ollama-parse`, `ollama-joint-mcq`: all
  three stages fully cached from M3's original run, so this reproduces
  M3's exact 358/500 result.
- `aynvqa-run --split dev --language msa --repair-enabled` -- same ASR/parse/
  select stages, repair ladder on, Whisper large-v3 (GPU, float16) as the
  step-2 backend.

A separate general ASR-only bench (`aynvqa-transcribe --whisper-model-size
large-v3`, same seeded 50-item sample M2 used for medium-vs-small) checks
transcription characteristics independent of repair.

**A methodology caveat worth stating plainly:** because repair changes the
select-stage cache key for *every* item (not just the ~25% flagged ones),
both runs re-called the VLM for all 500 records. Checking the ~412 items
whose parsed options were byte-identical in both runs found the VLM's own
prediction differs on 3 of them (0.73% resampling noise, even at
`temperature=0.0` -- plausibly GPU floating-point non-associativity) and
flips correctness on 2 (one favorable, one unfavorable -- net zero on the
headline number here, but a real source of run-to-run jitter to keep in
mind for future ablations).

## Headline results

| | repair off (baseline) | repair on |
|---|---:|---:|
| Accuracy | 0.7160 (358/500) | **0.7380 (369/500)** |
| Balanced accuracy | 0.7141 | 0.7370 |
| Macro F1 | 0.7143 | 0.7369 |
| Wrong predictions | 142 | 131 |
| Pipeline artifact (code-reported, `option_quality_ok`) | 71 | **27** |
| Genuine visual-reasoning miss + ambiguous ground truth | 71 | 104 |
| Runtime (this machine, this ablation) | ~4s (full cache reuse) | ~56m |

**+11 net correct (+2.2 accuracy points), and pipeline-artifact-flagged
wrong predictions dropped 61% (71 -> 27)**, from a prompt-only repair layer
with no schema change and no new evidence channel.

Digging into the net +11: **24 items flipped wrong -> correct, 13 flipped
correct -> wrong.** The 13 aren't a straightforward regression --
inspecting them individually found: 2 are the VLM resampling noise
described above; several are cases where the model had been "correct" on a
degenerate option set only by default (e.g. picking the one non-empty
option among three, or process-of-elimination against blank slots) and,
once given genuinely reconstructed options, made a real but wrong
visual-reasoning call; and at least one is a **new failure mode**: the
repair-retry re-split a transcript in a *different order* than the
original parse, which can misalign the fixed integer gold label against
the now-differently-ordered option text (concretely: a car-brand question
where "Rolls-Royce" was gold-index 0 before repair and correct-index text
after repair moved to index 1). This is a real, previously unconsidered
risk of the escalation ladder, not a data artifact -- see Limitations below.

Reconciling against the original hand-built 142-item taxonomy (id-for-id
identical to the repair-off wrong set, confirming full reuse was valid) to
split "genuine visual miss" from "ambiguous/debatable ground truth" -- a
distinction `option_quality_ok` can't make, since it only checks structural
form:

| | repair off (hand-verified) | repair on (partially hand-verified*) |
|---|---:|---:|
| Pipeline artifact | 69 (49 auto-agrees + 20 manual-only) | ~38 |
| Genuine visual-reasoning miss | 32 | ~35 confirmed + most of 34 unverified |
| Ambiguous / debatable ground truth | 19 | ~24 confirmed |
| (checker/manual disagreement) | 22 auto-flags manual calls fine | -- |

*The repair-off column is a full manual re-verification (same one used in
the original error-taxonomy analysis). The repair-on column carries every
manual tag forward for the 97 wrong items whose parsed options are
byte-identical between conditions (unaffected by repair, tag still valid),
and hand-reviewed the 21 carryover items whose options *did* change plus a
sample of the 13 newly-wrong items (see below) -- but 34 wrong items
(mostly newly-wrong or restructured-and-still-wrong) were not individually
re-viewed against their images, only checked against `option_quality_ok`.
Treat the repair-on genuine/ambiguous split as a well-informed estimate,
not a full re-audit; `option_quality_ok`'s own 27/131 pipeline-artifact
count is exact and reproducible by anyone re-running the pipeline.

## Repair mechanics: how much each step contributed

```
500 records
  -> 125 (25.0%) parsed options flagged degenerate
       -> 57  (45.6%) resolved by step 1 (stricter reparse, same transcript, no new ASR call)
       -> 68  (54.4%) still degenerate -> escalated to Whisper large-v3
            -> 31 (45.6% of 68) resolved by step 2 (large-v3 transcript + stricter reparse)
            -> 37 (54.4% of 68) still degenerate after both steps -- proceeds unrepaired
```

**Runtime.** The measured 56-minute repair-on run is dominated by
re-running the VLM select call for all 500 items (a consequence of the
cache-isolation design that makes on/off ablations safe to compare, not of
repair itself). Repair's own marginal cost -- the extra parse/ASR calls
that only the 125 flagged items pay for -- is 125 step-1 reparse calls +
68 large-v3 transcriptions (1.3s mean, GPU) + 68 step-2 reparse calls,
roughly **16 minutes** on top of a normal full run. In other words: in
ordinary use (repair always on, no side-by-side comparison needed),
expect ~500 unchanged select calls plus ~16 minutes of extra work for the
25% of items that need it -- not 56 minutes.

**General ASR comparison** (medium vs. large-v3, same seeded 50-item
sample M2 used):

| backend | n_ok | mean chars | mean latency |
|---|---:|---:|---:|
| whisper-medium (CPU, int8) | 50/50 | 114 | 7.92s |
| whisper-large-v3 (GPU, float16) | 50/50 | 115 | **1.32s** |

Once the CUDA DLL issue was fixed, large-v3 is ~6x *faster* than medium on
CPU, not slower, with near-identical transcript length. A nice side
benefit, not what was being measured.

## Every case that needed Whisper-large: why, and what actually fixed it

All 68 escalation cases were read by hand (medium transcript vs. large-v3
transcript vs. the resulting options, before and after). The 31 that
resolved:

| Mechanism | Count | % |
|---|---:|---:|
| Better transcription (real word/content-level fix) | 12 | 39% |
| Large-v3's more consistent comma punctuation helped the parser split | 11 | 35% |
| Neither -- looks like arbitrary LLM retry variance | 7 | 23% |
| Content plausible-looking but not clearly grounded in either transcript (hallucinated) | 4 (subset of above) | 13% |

Representative examples:

- **Better transcription, unambiguous:** *"ماهي العلمة التجارية الظاهرة
  علي محور الدراجة؟"* (bike brand question) -- medium transcribed
  `كم باني ولا؟ يعني ألدز الدراجة شي مانو`, entirely unintelligible.
  Large-v3 transcribed the same audio as `كامبانيولا عينولدز الدراجة
  شيمانو` -- three real bicycle-component brands (Campagnolo, Reynolds,
  Shimano) cleanly recoverable, and the VLM then answered correctly.
  Also recovered: a genuine ASR repetition-loop (one item's medium
  transcript was 32 repeated instances of a single letter; large-v3
  produced two clean GPU model numbers instead) and several cases where
  medium duplicated or dropped a word that large-v3 transcribed once,
  correctly.
- **Punctuation, not content, was the fix:** e.g. a ferry-company
  question -- medium's transcript ran the three company names together
  with no punctuation (`الجزائرية للنقل البحري كورسيكا العبارات شركة
  البحر المتوسط للشحن البحري`); large-v3 produced the *same three names*
  but comma-separated (`...البحري، كورسيكا العبارات، شركة...`), and the
  strict-reparse prompt used the commas to split correctly where it
  couldn't before. This pattern repeated across ~1/3 of resolutions --
  large-v3 isn't hearing more, it's punctuating more consistently, and the
  parser is comma-sensitive.
- **Hallucination that happens to pass the structural checker:** a
  suitcase-materials question where large-v3's transcript only supports
  two real materials (fiberglass, metals) but the reparsed options include
  a third, "iron" (`الحديد`), never mentioned in either transcript.
  Structurally this is `is_degenerate=False` -- three distinct non-empty
  strings -- so it counts as "resolved," even though the third option was
  invented. This happened in at least 4/31 resolutions and is a real limit
  of a purely structural quality check (see Limitations).
- **One clear regression:** a Rio-landmark question where large-v3
  mis-transcribed the city name itself (`الريو` -> `الرياض`, Rio ->
  Riyadh) and produced worse, less-grounded options than medium had. Bigger
  isn't always better per-item, even if it wins in aggregate.

## The 37 unresolved cases: root cause

Every unresolved case (flagged degenerate, still degenerate after both
repair steps) was read by hand and assigned a primary root cause:

| Root cause | Count | % | Fixable by |
|---|---:|---:|---|
| Dataset issue -- the item genuinely has only two real spoken options (or needs OCR, e.g. an aircraft tail number never spoken aloud) | 16 | 43% | Nothing pipeline-side; needs a data-level flag or acceptance as a ceiling |
| ASR limitation -- neither model produces clean content (foreign brand names, heavily degraded audio, one outright transcription regression) | 11 | 30% | A better ASR model (diminishing returns beyond large-v3) or accept as noise floor |
| Parser prompt limitation -- the content *was* recoverable (in one case, large-v3 even produced clear comma delimiters) but the strict-reparse prompt still failed to segment it, or left the boilerplate preamble attached as a prefix instead of a standalone option | 9 | 24% | **Sharper repair-retry prompt -- cheap, targeted, already diagnosed** |
| Checker false positive -- the parsed options are arguably already fine (e.g. "suede leather" vs. "leather" trips the substring-overlap check but may be two legitimately different options) | 1 | 3% | Loosen the `option_quality` heuristic |

The largest bucket (43%) is a hard ceiling -- no ASR or parsing
improvement recovers an option that was never spoken. The second-largest
(30%) is close to a ceiling too, since large-v3 is already a strong open
ASR model. **The parser-prompt bucket (24%) is the one with real, cheap,
already-diagnosed headroom left**, and it's corroborated by the resolved
cases too: 23% of *successful* resolutions look like arbitrary retry
variance rather than a clear fix, and at least 13% show the parser
inventing content when it should have said "not found" -- both symptoms
of the same underlying issue, that the strict-retry prompt doesn't
consistently use the cues it's given (punctuation, "don't invent"
instructions) or fail honestly when it should.

## Lessons learned / limitations

1. **Structural validity is not semantic correctness.** `option_quality_ok`
   catches empty/duplicate/boilerplate options reliably, but a parser that
   confidently invents a plausible-sounding third option passes the same
   check. At least 4-6 of the 31 escalation resolutions, and several of
   the 21 restructured-but-still-wrong carryover items, show this pattern.
2. **The automated checker and careful manual review disagree ~30% of the
   time** (42/142 on the original error set: 22 items the checker missed
   that manual reading caught, 20 the checker flagged that manual reading
   judged as a fair test). It's a good fast proxy, not a ground truth --
   worth remembering before trusting `option_quality_ok` counts alone for
   any high-stakes claim.
3. **Repair-retry can reorder options relative to the original spoken
   sequence**, which risks misaligning the fixed integer gold label
   against the regenerated option text. Not designed for or caught by any
   existing test; found by hand-reviewing the 13 newly-wrong-after-repair
   items. Worth a follow-up: either detect reordering (e.g. fuzzy-match
   new options back to old ones by position) or accept it as a known,
   small source of noise.
4. **VLM selection has a small but real non-zero resampling noise rate**
   (~0.7% of predictions differ on byte-identical prompts, even at
   `temperature=0.0`). Small enough to not change this run's conclusions,
   large enough to keep in mind for any future ablation with a smaller
   expected effect size.
5. **The stricter reparse prompt is inconsistent, not just imperfect** --
   on rare occasions it produces distinctly *worse*, seemingly unrelated
   content on a retry of a transcript that had already parsed acceptably
   the first time (one clothing-item question went from three correct,
   lettered options to three unrelated words on retry). This suggests the
   checker's false-positive rate matters more than its size alone implies:
   a needless repair attempt isn't free, it's a small risk of regression.

## Recommended next step

**A second, evidence-driven pass on the repair-retry prompt, before
starting few-shot or OCR.** Concretely, informed directly by the 37
unresolved and 31 resolved cases read for this evaluation:

- Explicitly instruct the parser to split on comma delimiters when present
  (missed at least once here despite the commas being right there).
- Explicitly instruct it to never retain the introductory phrase even as a
  *prefix* within an option (several "resolved" options still start with
  "الخيارات هي عبارة عن...").
- Explicitly instruct it to prefer splitting a single compound clause
  joined by "و" (and) into separate options when the clause plausibly
  describes more than one distinct thing (the single largest recurring
  pattern in the unresolved set).
- Explicitly instruct it not to invent a plausible-sounding option when
  fewer than three are genuinely present, and say so instead -- directly
  targets the hallucination pattern found in both the resolved and
  unresolved sets.

**Why this before few-shot or OCR, despite few-shot's larger addressable
pool:** few-shot targets genuine visual-reasoning misses, now the majority
of remaining errors (roughly 35-70 of the 131, depending on how the
unverified carryover items eventually resolve) -- a real, larger
opportunity in absolute terms. But it's also a materially bigger
investment: exemplar retrieval design, prompt engineering, and careful
anti-regression evaluation, all with less predictable effect size than a
prompt tweak targeting failure patterns already read by hand with specific
failing examples in front of us. OCR is smaller still at full scale -- the
original 142-item analysis found only 5 confirmed OCR-relevant errors
(3.5%), and this evaluation's unresolved-case reading found the same
pattern again (an aircraft registration number literally never spoken in
the audio) without surfacing a materially larger pool. Given the prompt
refinement above costs a few hours (no new infrastructure, same test
pattern already established) against an estimated **+0.8 to +1.6
additional accuracy points** (roughly half the unresolved parser-limitation
cases, plus a reduction in the resolved-but-hallucinated share) -- it's the
better next hour of work before committing to either bigger milestone.

No code has been changed for this recommendation -- awaiting review.
