# M5 -- Genuine visual-reasoning errors: CoT and few-shot retrieval

Companion to [`docs/M4_PIPELINE_ROBUSTNESS.md`](M4_PIPELINE_ROBUSTNESS.md).
**Frozen M4 baseline: 74.80% (374/500), dev/msa** -- repair enabled,
parser prompt v2 -- is the comparison point for everything below.

## Scope: what's actually addressable

The frozen baseline's 126 wrong predictions were reconciled against the
M4 taxonomy (carrying forward every tag whose underlying options didn't
change, hand-categorizing the rest):

| Bucket | Count | % |
|---|---:|---:|
| Pipeline artifact | 55 | 43.7% |
| **Genuine visual-reasoning miss** | **43** | **34.1%** |
| Ambiguous / debatable ground truth | 22 | 17.5% |
| Unreviewed carryover | 4 | 3.2% |
| Pure VLM resampling noise | 2 | 1.6% |

M5 targets the 43-item genuine-visual-miss bucket. Ceiling: even fully
solving it is +43/500 = +8.6 points; realistic techniques recover a
fraction of that.

## Design

Two independent techniques, evaluated separately before any combination
(per explicit direction): a chain-of-thought (CoT) select-prompt variant,
and few-shot exemplar retrieval from `train`. Both were prototyped as a
43-item pilot on the genuine-visual-miss set before any full-scale run --
same discipline as M4's parser-v2 iteration.

**CoT** (`stages/select_vlm.py`, `use_cot` flag): the JSON schema gains
`visible_details` and `reasoning` fields *before* `answer_index`, so
constrained decoding generates a description and reasoning chain before
committing to an index, rather than jumping straight to one. No retrieval,
no new dependency -- purely a prompt/schema change, isolated by giving the
selector a distinct `.name` (`ollama-joint-mcq-cot`) so its cache can never
collide with the zero-shot selector's.

**Few-shot** (`stages/retrieve.py`, new): retrieves k=2 exemplars per
query by VLM-predicted category/subcategory (a 9-category, 31-subcategory
closed taxonomy enumerated from all 3000 `train_msa.jsonl` records), not
image embeddings -- this project has never needed a torch/CLIP dependency,
and predicted category is the only signal legitimately available at
inference time (`devtest`/`test` don't carry ground-truth
category/subcategory, only `train`/`dev` do). Retrieved train records are
hydrated into full exemplars (question/options text + ground-truth answer)
by running them through the *same, unmodified* `_run_asr_stage`/
`_run_parse_stage` functions M4's repair ladder already uses, pointed at
`train`'s own cache namespace. Exemplars are shown as a multi-turn
conversation (`OllamaClient.chat_messages`, new) -- each exemplar a
user/assistant turn pair, ending with the real query. Required raising
Ollama's context window: its own 4096-token default 400s on more than one
image, confirmed by direct testing; `fewshot_num_ctx` defaults to 16384.

Both are off by default (`cot_enabled`, `fewshot_enabled` in
`config.py`), same reasoning as every prior unvalidated M4 flag: nothing
changes for existing callers until a flag is explicitly set.

## CoT: results

**43-item pilot: 13/43 (30.2%) recovered**, zero new infrastructure. Hand
read against images: ~10/13 genuinely well-grounded (texture reasoning
for fabric ID, wheel-count reasoning for vehicle type, one correct
historical-knowledge recall the model explicitly flagged as not
image-derived), ~3/13 weaker elimination-style reasoning that still
landed correct.

**Full 500-item run: 75.00% (375/500) -- net +1 correct.** Far below the
pilot's implied ceiling. 63 total flips (32 fixed, 31 regressed) -- almost
double the parser v1-to-v2 change's 33, because CoT changes *every*
select call's behavior, not a targeted subset.

All 31 regressions read against their reasoning traces (not just
counted), plus a sample of the fixes as a cross-check. Two concrete,
recurring, previously-undocumented patterns:

1. **Reasoning-to-index inconsistency**, at least 7 clear cases: the
   `reasoning` field explicitly names the correct option, or explicitly
   argues against the option ultimately chosen, and `answer_index` doesn't
   match anyway (e.g. reasoning concludes "aligns with Taraweeh prayers"
   -- the correct option -- output index names a different one). The fixed
   sample shows the identical instability landing correct by chance (one
   trace concludes "none of the options can be identified" and still
   outputs the right index; another references "option 3" when only
   0-2 exist). This reads as a token-generation reliability gap between
   the reasoning fields and the final integer, not inconsistent visual
   judgment.
2. **Fabricated-placeholder seduction**, at least 4 cases: a degenerate
   option (literally "the option is incorrect", "the third option", or a
   restated question -- all known `option_quality` pipeline artifacts)
   gets carefully reasoned about and selected, where a quicker zero-shot
   pass apparently skips past it. Deliberation backfiring on already-broken
   input.

Remainder: genuinely hard, defensible misses (fine texture discrimination,
Sudan-vs-Egypt pyramid knowledge, an unrecognized culturally-specific
instrument) and a few already-known degenerate-parse items that aren't a
fair test of reasoning quality either way.

**Not adopted as-is.** The mechanism isn't broken -- roughly half the
churn is genuine reasoning gain -- but a specific, likely-fixable
reasoning/index binding gap cancels most of it out at full scale. A cheap
v2 (tighten the reasoning-to-index binding; explicitly instruct the model
to discount fabricated-placeholder-looking options) is a plausible,
low-cost follow-up, not attempted here.

## Few-shot: results

**43-item pilot: 6/43 (14.0%) recovered.** Every query retrieved its full
k=2 exemplars (no retrieval failures). Weaker than CoT's pilot signal, so
before spending a full-scale run on it, checked *why*.

Cross-referenced every query's retrieved exemplars against their actual
`train` category (both exemplars always shared one category -- the
subcategory-then-category fallback is working as designed) and against
what the query was actually asking about. Roughly half the 43 got a clean,
correct category match; the rest range from questionable to clearly
wrong -- "type of car shown" retrieved *Culture, Arts & Entertainment*
exemplars instead of *Vehicles & Transportation*; "Sufi dervish ritual
clothing" retrieved *Modern Culture & Trends* instead of
*Religion & Spirituality*; "type of transport shown" retrieved
*Geography, Buildings & Landmarks*. 5 of the 6 successful flips came from
a clean category match; only 1 came from a questionable one.

**Diagnosis: the bottleneck looks like classification precision, not the
few-shot mechanism itself.** A single 9-way category call from one image
is a coarse signal, and roughly a fifth of the pilot's misclassifications
are the kind of error a sharper classification prompt, a two-stage
verify-then-retrieve step, or (at real infrastructure cost) image-embedding
retrieval would directly address.

**Full 500-item run: 74.80% (374/500) -- net zero change from the frozen
baseline.** Requested after review, to get a complete comparison rather
than stop at the pilot. 36 total flips: exactly 18 fixed, 18 regressed --
an even split, not a near-miss.

Recomputed retrieval (deterministic per `record_id`+seed, so reproducible
after the fact) for all 36 and cross-checked exemplar categories against
each question, the same method used for the pilot. This time the picture
is less clean than the pilot suggested: both the fixed and regressed sets
contain a mix of clean category matches and clear misses (a wind-turbine
question retrieved *Objects, Materials & Clothing* exemplars; several
historical/cultural questions retrieved well-matched History or Culture
exemplars) -- but *outcome doesn't cleanly track match quality* at this
larger, more representative scale the way it appeared to on 43 items.
Worth being direct about: the pilot's "good retrieval correlates with
success" read doesn't fully hold up under a larger sample. One item
(`9a9d0d68b0ea372f`, the Bahrain Tree of Life question) regressed under
*both* CoT and few-shot independently, suggesting it's a genuinely hard
item rather than a technique-specific failure.

**Net assessment: at k=2 with category-based retrieval, few-shot shows no
reliable effect in either direction at full scale** -- a starker, cleaner
null result than CoT's own diagnosed-but-real +1.

## Lessons learned

1. **A pilot on the hard subset doesn't reliably predict the full-scale
   net effect** when the technique changes behavior on *every* call, not
   a targeted subset. M4's repair ladder only touched flagged items, so
   its pilots generalized cleanly; CoT and few-shot both touch every
   select call, and CoT's 30%-pilot-to-+1-net gap is the clearest evidence
   yet that this matters. Worth designing future pilots to sample some
   already-correct items too, not just the target failure bucket, if this
   pattern shows up again.
2. **Reading reasoning traces, not just counting flips, found a bug
   automated scoring couldn't** -- the reasoning-to-index mismatch. A
   pure accuracy number would have shown "CoT: +1, marginal" and stopped
   there; reading *why* found a specific, fixable mechanism.
3. **Retrieval-quality diagnosis from a small pilot doesn't automatically
   generalize either** -- the same caution as lesson 1, one level down.
   The 43-item pilot's "good category match correlates with success" read
   was a reasonable hypothesis from the data available at the time, and
   partially explained the pilot's weaker-than-CoT signal, but the full
   500-item regression set didn't reproduce as clean a correlation. Small
   samples can suggest a mechanism convincingly without it being the whole
   story -- worth remembering before treating any pilot-scale diagnosis as
   settled.

## Final comparison and M5 summary

| | Accuracy | vs. frozen baseline | Flips (fixed / regressed) |
|---|---:|---:|---:|
| **Frozen M4 baseline** (no CoT, no few-shot) | 74.80% (374/500) | -- | -- |
| **CoT** | 75.00% (375/500) | +1 net | 32 / 31 |
| **Few-shot** (k=2, category retrieval) | 74.80% (374/500) | +0 net | 18 / 18 |

Neither technique is adopted as a production default. Both flags
(`cot_enabled`, `fewshot_enabled`) stay off, matching every other
unvalidated M4/M5 experiment in this codebase.

**CoT -- strengths:** zero new infrastructure, zero new dependencies;
recovers real, well-grounded cases (confirmed by reading, not just
counting); the mechanism itself works when it works. **Limitations:** a
diagnosed reasoning-to-answer-index reliability gap (~7 of 31 regressions,
plus the same instability visible in the fixed sample) and a fabricated-
placeholder-option vulnerability (~4 of 31) roughly cancel out the real
gains at full scale; substantially slower and more token-hungry per call
than zero-shot. **Recommendation:** don't adopt as-is. A cheap v2
(explicit reasoning-must-match-index instruction, explicit instruction to
discount placeholder-looking options) is a plausible fast follow, deferred
as future work, not implemented now.

**Few-shot -- strengths:** clean, reusable, Protocol-based infrastructure
(`retrieve.py`, multi-turn `OllamaClient.chat_messages`, exemplar-aware
selector) that composes with CoT and required zero new ML dependencies;
retrieval mechanics work reliably (100% of pilot and regression-set
queries got their full exemplar count, deterministic and reproducible).
**Limitations:** net zero effect at full scale with the current k=2,
category-only retrieval; category classification has a real, if smaller-
than-initially-estimated, error rate; meaningfully higher latency and
compute cost than zero-shot or CoT (classification + exemplar ASR/parse
hydration + slower multi-image select calls) for no measured accuracy
gain in its current form. **Recommendation:** don't adopt as-is. Sharper
classification, more exemplars, or image-embedding retrieval are all
plausible directions, deferred as future work, not implemented now.

**Why defer both rather than iterate now:** each already went through one
full pilot-then-full-scale cycle with a concrete root-cause diagnosis
recorded above; further iteration is a new design-and-validate cycle in
its own right, not a quick fix. Freezing here keeps M5's evidence clean
and gives a real baseline (both techniques' exact failure modes, on the
record) for whoever picks this back up.

## M5 status: frozen

No change to production defaults. The frozen M4 baseline (74.80%,
repair enabled, parser prompt v2, no CoT, no few-shot) remains the
reference point for any future milestone. `cot_enabled` and
`fewshot_enabled` exist as fully-implemented, fully-tested, off-by-default
flags for whoever revisits this -- CoT v2 and few-shot classifier
improvements are explicitly future work, not scheduled.
