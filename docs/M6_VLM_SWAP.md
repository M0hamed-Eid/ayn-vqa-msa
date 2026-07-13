# M6 -- VLM swap: qwen2.5vl:7b -> qwen3-vl:8b

Companion to [`docs/M5_FEWSHOT_RETRIEVAL.md`](M5_FEWSHOT_RETRIEVAL.md).
M5 concluded prompt-level techniques (CoT, few-shot retrieval) had hit
diminishing returns against the frozen M4 baseline -- 74.80% (374/500),
dev/msa. M6 tests a different lever: the underlying vision-language model
itself, pipeline otherwise unchanged (repair enabled, parser prompt v2).

## Model selection

Requested candidates: InternVL, a newer Qwen-VL, or another competitive
open-source model. Checked directly against Ollama's actual library
before committing to anything:

- **InternVL is not in Ollama's official catalog** -- only unofficial
  community GGUF conversions (e.g. third-party `blaifa/InternVL3`
  uploads), needing manual `Modelfile` setup with no first-party quality
  assurance and unverified compatibility with the JSON-schema-constrained
  decoding this entire pipeline depends on for every prediction.
- **Qwen3-VL is a first-party official Ollama model** -- the direct
  successor generation to the current qwen2.5vl, satisfying the "newer
  Qwen-VL" option with materially lower risk. 8B variant, 6.14GB
  download, comparable footprint to the current 5.97GB qwen2.5vl:7b.

Given InternVL's availability gap, proceeded with **qwen3-vl:8b only**
(confirmed by direction, not assumed). Verified structured-output
compatibility with a 5-item smoke test through the actual production
`OllamaJointMCQSelector` before committing to a full run -- clean
`answer_index` extraction on all 5, no schema violations. Also noted:
qwen3-vl reports a `thinking` capability (vs. qwen2.5vl's plain
`vision, completion`) and per-call latency was markedly higher and more
variable (7.8s-71.7s in the smoke sample, vs. qwen2.5vl's typical
~4-5s) -- flagged before running, not discovered after.

**Hardware note, unrelated to the model choice but surfaced during this
work:** `nvidia-smi` reports this machine's GPU as an RTX 2000 Ada
Generation (16GB), not the RTX 4080 referenced when the M4 repair-ASR
backend was configured. Same VRAM capacity either way, so it didn't
change any capacity planning here, but it's a lower-throughput
workstation card than a 4080 -- worth knowing for future latency
expectations.

**A real bug fixed before running anything:** `OllamaJointMCQSelector`'s
cache-key `.name` only varied with the CoT flag, never with `model`. A
bare `--ollama-select-model` swap would have silently collided with the
default model's existing select-stage cache and replayed its qwen2.5vl
predictions without ever calling qwen3-vl -- format-checking and scoring
fine, showing "no change," while never actually invoking the new model.
Fixed to include the model name in `.name` whenever it differs from the
default (mirroring how `asr_config_key` already varies with Whisper's
model size); covered by two new tests.

## Methodology

Same M4 pipeline, same 500 dev/msa records, repair enabled, parser prompt
v2 unchanged -- only `--ollama-select-model qwen3-vl:8b`. No CoT, no
few-shot (isolating the VLM-swap variable alone, same discipline as
every prior M4/M5 ablation). Runtime: ~2h14m (22:24-00:38), the longest
full run so far, consistent with the higher observed per-call latency.

## Results

**79.00% (395/500) -- +4.2 points, +21 correct net over the frozen
baseline.** By a wide margin the largest single-change improvement of
M4-M6: parser v2 was +2.2 points, CoT +0.2, few-shot +0.0.

| | Accuracy | vs. frozen baseline |
|---|---:|---:|
| Frozen M4 baseline (qwen2.5vl:7b select) | 74.80% (374/500) | -- |
| CoT (M5) | 75.00% (375/500) | +0.2 |
| Few-shot, k=2 (M5) | 74.80% (374/500) | +0.0 |
| **qwen3-vl:8b select (M6)** | **79.00% (395/500)** | **+4.2** |

48 flipped wrong-to-correct, 27 flipped correct-to-wrong (net +21).
Pipeline-artifact-flagged wrong predictions among the remaining errors
dropped slightly too (36 -> 33), though that's a secondary effect of
which specific items ended up wrong, not a parsing change -- parsing was
identical in both runs.

**All 27 regressions read against their content** (no reasoning trace
available here -- this run used the plain, non-CoT schema, same as the
frozen baseline). The most useful finding: **at least 10 of the 27 are
items that also regressed under CoT and/or few-shot in M5**, several
landing on the exact same wrong answer index across techniques:

- A "RACING" logo team-vs-brand distinction, a weather question with a
  self-contradictory garbled option, a courtyard tile-pattern question
  with heavily garbled options, and the Sudan-vs-Egypt pyramid question
  all independently regressed under CoT *and* qwen3-vl, several to the
  identical wrong index.
- The red-statue "abstract art" interpretation question regressed under
  **all three** of CoT, few-shot, and qwen3-vl -- to the same wrong
  answer every time.
- Two items (`3d9b67cd6c71e037`, `b45082c9bb88c6fe`) are already-known
  pipeline-artifact cases from the original M4 taxonomy -- one has no
  real options in the transcript at all, the other's gold label points at
  a blank slot in a genuinely broken parse. Not a fair test of any
  select-stage change.

Read plainly: a substantial share of qwen3-vl's regressions are on items
already established as hard, ambiguous, or parser-broken regardless of
which technique or model answers them -- not new weaknesses this swap
introduced. The remaining ~15-17 regressions look like ordinary
fine-grained visual/knowledge misses (material identification, staff-role
distinction, specific organizational branding) -- the normal cost of any
model swap, not a systematic problem.

## Decision: adopt qwen3-vl:8b as the new default

Per the stated criterion -- a single model giving a clear improvement
gets adopted -- **`ollama_select_model` in `config.py` now defaults to
`qwen3-vl:8b`**. `ollama_parse_model` is unchanged (`qwen2.5vl:7b`) --
only the select stage was evaluated and changed; swapping the parser
model is a separate, unevaluated question. The ensemble step in the
original plan is not needed: the criterion for it ("no single model is
consistently better") wasn't met -- qwen3-vl won clearly and by a wide
margin.

**New frozen baseline for all future milestones: 79.00% (395/500),
dev/msa** -- repair enabled, parser prompt v2, select stage on
qwen3-vl:8b, no CoT, no few-shot. `cot_enabled`/`fewshot_enabled` remain
off by default; whether they're worth re-evaluating against this new,
stronger baseline (rather than the old qwen2.5vl one they were tuned
against) is an open question for whoever picks this back up, not
answered here.

## Limitations

1. **Latency.** ~2h14m for 500 items, meaningfully slower than the ~1h
   frozen baseline (which reuses cache almost entirely) or a fresh
   qwen2.5vl run. Real cost for any future full-scale ablation against
   this new baseline.
2. **Single-run result.** Like every other ablation in this project, this
   is one run at temperature 0 -- the ~0.7% VLM resampling noise rate
   documented in M4 applies here too. A +21 net swing is far too large to
   be noise-explained, but exact per-item reproducibility wasn't
   re-verified for this specific comparison.
3. **Regressions weren't all explained.** ~15-17 of the 27 are ordinary,
   plausible visual/knowledge misses without a reasoning trace to confirm
   *why* -- unlike the CoT analysis in M5, this run's schema doesn't
   surface the model's reasoning, so root-causing individual regressions
   further wasn't possible without re-running with a CoT-style schema
   (not done here).
