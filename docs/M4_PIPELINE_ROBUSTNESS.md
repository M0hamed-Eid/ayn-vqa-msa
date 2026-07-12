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

## Addendum: parser prompt v2

The recommendation above was approved, with one addition: pilot the new
prompt on the 37 unresolved cases specifically (cheap, fast) before
committing to a full 500-item re-run, and pay particular attention to
option ordering given Limitation 3 above.

### Design

v2 keeps the schema and the `parse(..., strict=True)` mechanism exactly as
they are -- only `_STRICT_PROMPT`'s text changed (see
`src/ayn_vqa/stages/parse.py`). Five explicit rules replace v1's shorter,
looser instruction: strip the "options are" preamble even when it appears
as a *prefix* attached to real option text (not just as a standalone
option); treat commas as strong option boundaries; split compound,
delimiter-free clauses into their most plausible three-way grouping without
breaking a single clause's internal "و" apart from itself; preserve the
transcript's original spoken order; and -- the one deliberate reversal from
v1 -- never invent content, using the most plausible leftover fragment
instead of a placeholder or an unsupported claim about the image.

### 37-case pilot: methodology and results

Built an isolated evaluation harness (scratchpad-only, no production files
touched) that called Ollama directly with the v2 prompt against each of
the 37 unresolved cases' large-v3 transcript -- the same input v1's
strict-reparse had already failed on, for a clean apples-to-apples read --
plus, as a bonus check, each case's original medium transcript. Scored
every result with the existing `check_option_quality()` (unchanged, reused
read-only) and a purpose-built order-preservation check (fuzzy-matches
each option's approximate position back into the transcript and verifies
positions are non-decreasing). Every one of the 37 outputs was then read
by hand against its transcript, not just scored automatically -- consistent
with Limitation 2 above.

**Result: 5/37 (13.5%) structurally resolved** on the large-v3 transcript
-- well below what an eyeballed read of the 37 transcripts suggested going
in (the empirical run corrected that prior). Breaking down what worked and
what didn't:

- **Comma-boundary recovery: confirmed working.** The one pilot case with
  real commas (`67635542e05efee4`, named in the "Recommended next step"
  above) split cleanly into 3, order preserved.
- **Boilerplate-prefix stripping: confirmed working.** A case whose
  previous output still carried "الخيارات هي" as a prefix on option_0 no
  longer does.
- **Fabrication: dropped sharply.** 1/37 residual fabricated placeholder,
  versus a recurring pattern under v1 (hallucinated "iron," a "no colorful
  flags in image" negation, "third option unavailable" placeholders).
- **Order preservation: no confirmed real violations.** 28/30 checkable
  cases came out monotonic. The 2 flagged violations turned out, on
  inspection, to be artifacts of the position-checker's own substring
  matching (matching an earlier occurrence of a word that also recurs
  later in the transcript) rather than genuine reordering by the model.
  Given this was the specific risk flagged for extra attention, that's a
  meaningful result on its own, independent of the low resolution rate.
- **Compound/no-delimiter splitting: partial.** 31/32 still-degenerate
  cases fail for the same reason (`empty_option`), but the *shape* of the
  failure changed: the model now typically finds a correct or reasonable
  **2-way** split and stops, rather than v1's typical single blob with two
  blank slots. Real, visible progress that still trips the same structural
  check.

Dug into *why* rather than patching around it: **0 of the 32 still-
degenerate cases have any unused transcript text left over** after their
last populated option (checked programmatically). The model isn't
ignoring available text -- for most of these, the transcript genuinely
only contains two cleanly recoverable segments (e.g. `426f876e14e6b1bb`:
"2080 5780"; `d85b420ae6bc205e`: "123 سويا", the aircraft-registration/OCR
case). A smaller minority (roughly 5-6 of the 32) do have a genuine
three-way split achievable by sub-dividing one of the two groups the model
already found, which it didn't attempt.

One further, unplanned finding: the bonus medium-transcript run resolved a
**different, non-overlapping** set of 4 cases from the large-v3 run's 5 --
evidence of real sensitivity in how this 7B parser model responds to small
phrasing differences, not a clean "better transcript always parses better"
relationship.

### Decision: freeze v2, measure end-to-end impact

A further prompt iteration was considered and explicitly rejected: the
0-unused-leftover-text finding shows the dominant remaining failure isn't
a wording gap in the instructions, it's that a fixed three-non-empty-option
schema cannot represent a transcript that honestly only contains two
recoverable options without either fabricating (v1's failure mode) or
leaving a slot blank (v2's) -- more prompt engineering was judged unlikely
to move that number further. v2 was frozen as the production repair-retry
prompt (replacing v1 in place, same file, same mechanism) on the strength
of its confirmed wins -- comma-splitting, prefix-stripping, fabrication
reduction, and clean order preservation -- to be validated by a full
500-item repair-enabled run under the same methodology as the original M4
ablation.

### Known limitation, deferred: the fixed three-option schema

Recorded here as future architectural work, **not scheduled or
implemented**: M4's original design deliberately kept the parser's output
schema fixed at exactly three non-empty options, to keep the validation
layer independent of any schema change (see the M4 design decisions at the
top of this document). The v2 pilot is the first concrete evidence that
this has a real cost -- an unknown but nonzero share of transcripts (most
plainly, the dataset/OCR-needed cases already identified) genuinely
support only two recoverable options, and no prompt wording can make a
schema honestly represent a count it doesn't allow. Possible future
directions, none evaluated: an explicit "fewer than three recovered"
signal in the parse output (a real schema change, with all the downstream
consequences that implies for the selector stage and for comparability
with every prior milestone's numbers); or a confidence/coverage field
alongside the three slots. Deliberately left as a future option, not a
next step -- changing the parser's output contract now would affect the
rest of the pipeline and complicate comparison with M0-M4's results, which
the schema-freeze decision in M4 was specifically meant to avoid.

### Full-scale v2 comparison

Same methodology as the original M4 ablation: full 500-item dev/msa,
repair enabled, v2 as the only change from the frozen M4 baseline. The
`off` leg was not re-run -- repair-disabled runs never call the
repair-retry prompt, so v1 and v2 are identical for that condition and the
original M4 numbers stand. The repair-stage and select-stage caches
*are* keyed by config name rather than prompt content, so the three
v1-prompt-dependent cache files were archived (not deleted --
`artifacts/_v1_prompt_archive/`) before this run to force a genuine
recompute rather than silently replaying v1's cached predictions; output
went to a separate `reports/m4_parser_v2/` directory so the original M4
ablation CSVs remain untouched.

| | off | v1 (frozen M4) | v2 |
|---|---:|---:|---:|
| Accuracy | 0.7160 (358/500) | 0.7380 (369/500) | **0.7480 (374/500)** |
| Wrong | 142 | 131 | 126 |
| Flagged degenerate (any correctness) | 125 | 37 | 50 |
| Repair: resolved by reparse / by ASR escalation / still degenerate | -- | 57 / 31 / 37 | 33 / 42 / **50** |
| Runtime | ~4s (cache reuse) | ~56 min | ~63 min |

**v2 beats v1 by +5 correct (+1.0 accuracy point) -- but resolves *fewer*
of the originally-125-flagged items structurally (75 vs 88), leaving more
(50 vs 37) still flagged degenerate.** This is the fabrication/honesty
trade-off from the pilot showing up at full scale, confirmed by the
runtime breakdown: v2 resolves fewer cases at the cheap reparse-only step
(33 vs v1's 57 -- consistent with the pilot's finding that v2 declines to
force a third option into existence) but *more* once escalated to a
better transcript (42 vs v1's 31). v1's higher "resolved" count included
real fabrication (documented in the base M4 report); v2's honesty about
not finding a third option, it turns out, costs less accuracy than v1's
confident invention did.

**Regression check** (the user's explicit adoption criterion): comparing
v1's and v2's predictions record-by-record, 19 flipped wrong-to-correct
and 14 flipped correct-to-wrong (net +5, matching the accuracy delta).
Every one of the 14 was read by hand against its transcript and options,
not just counted:

| Category | Count |
|---|---:|
| Pure VLM resampling noise (options byte-identical between v1 and v2, prediction differs anyway) | 2 |
| Selector sensitivity to a cosmetic rewording (v2's text is a lateral change or arguably cleaner, selector flipped anyway) | 3 |
| **Genuine v2 parse-quality regression** (v1's retry had already produced an adequate split; v2's retry made it worse) | **6** |
| v1's "correct" answer came from a lucky guess against a broken/degenerate options set; v2 gave the selector an honest (if imperfect) parse and it made a real, wrong visual-reasoning call | 2 |
| Novel failure mode not seen in the 37-case pilot: v2 absorbed part of the *question* text into an option slot | 1 |

Two things worth being direct about:

1. **The 6-case "genuine regression" bucket is not a new problem v2
   introduced.** Every one of them is the same failure already named as
   Limitation 5 in the base M4 report -- the strict-reparse prompt is
   inconsistent, and occasionally makes an already-acceptable parse worse
   on retry (one case produced entirely unrelated meta-commentary text; in
   general v2 was not designed to fix retry *reliability*, only specific
   structural patterns, and it didn't). v1 carried the same risk; this run
   is the first time it's been quantified rather than just documented from
   a handful of examples.
2. **Taken literally, "no regressions" is not the outcome.** 14 records
   got worse. The net is positive (19 > 14, and only 2 of the 14 are pure
   noise, so this isn't just noise cancelling out), and the fixed cases
   are mechanistically the ones v2 was built for -- both the comma-split
   case (`67635542e05efee4`) and the compound-clause case
   (`477582a210bc5ada`) named throughout the pilot as target examples
   flipped wrong-to-correct here in the real pipeline, not just the
   isolated harness. But "measurable improvement, no regressions" as
   literally stated was not fully met, and that distinction is left for
   review rather than resolved unilaterally here.

### Decision: adopt v2, freeze M4

v2 is adopted as the production repair-retry prompt. The improvement over
v1 is real (+1.0 accuracy point, net of noise) and not free, but the
regression mode is a pre-existing, already-documented risk rather than one
v2 introduced, and the mechanism behind the gains is exactly what v2 was
built for, now confirmed at full scale rather than just on the 37-case
pilot.

**Known limitation, deferred: question text leaking into an option.**
Recorded as future work, not fixed. One full-scale regression
(`6173f511282d1039`) shows the strict-reparse prompt, on retry, absorbing
part of the *question* itself into an option slot (producing an option
that reads as a restatement of the question rather than an answer to it) --
a failure mode not seen anywhere in the 37-case pilot and not previously
documented for v1 either. `option_quality.check_option_quality()` has no
check for this (it only inspects the options for internal degeneracy, not
their relationship to the question text), so it passed structurally.
Occurred once in 500 records; too rare from a single occurrence to design
a targeted fix around with any confidence, but worth watching for if it
recurs. A question-text-overlap check (e.g. flag an option that shares an
unusually long substring with the question) would be the natural next
structural check to add if it does.

**M4 (pipeline robustness) is frozen as of this run.** Production
defaults: `repair_enabled=True`, repair-retry prompt v2, Whisper large-v3
as the repair-escalation ASR backend. Frozen baseline for all future
milestones: **74.80% (374/500), dev/msa**, from
`reports/m4_parser_v2/error_analysis_whisper-medium__ollama-parse__ollama-joint-mcq__repair_dev_msa.csv`.
Every subsequent milestone's ablations compare against this number, the
same way M4 compared against M3's 71.60%.
