# M2 -- ASR bench: design walkthrough

The "why", file by file, for M2: pick a transcription stack with data, not
vibes, before building anything downstream that depends on it. Companion to
[`docs/M0_DATA_AUDIT.md`](M0_DATA_AUDIT.md), [`docs/M1_BASELINES.md`](M1_BASELINES.md),
and [`../../AynVQA-ArabicNLP26/project_analysis_and_plan.md`](../../AynVQA-ArabicNLP26/project_analysis_and_plan.md)
(§11, M2). **Task 1a only** -- nothing here touches Task 1b/1c
(hallucination detection); `backbone.py`'s `evaluate_tf` parser, vendored
alongside the scorer in M1, stays unused.

## Scope decision: which ASR backends actually run

The roadmap's original M2 sketch names three systems (Whisper large-v3,
Fanar Aura-STT, a frontier multimodal API). Two of those need a paid API
key not configured in this environment, and calling them costs money per
request -- not something to spend without asking first. Given the choice,
**only local Whisper (via `faster-whisper`) is actually exercised here**;
`FanarAuraASR` and `OpenAITranscribeASR` are complete, correct client code
(see below) that stay unrun until `AYNVQA_FANAR_API_KEY` /
`AYNVQA_OPENAI_API_KEY` are set in `.env`. That still delivers real,
data-driven ASR selection -- just narrowed to "which Whisper size" instead
of "which vendor" for now.

## `src/ayn_vqa/stages/asr.py`

`ASRBackend` is a `Protocol`, matching `stages/select.py`'s pattern from
M1: one method, `transcribe(record_id, audio_path) -> Transcript`. Three
implementations:

- **`WhisperLocalASR`** -- the one backend actually run. Uses
  `faster-whisper` (a CTranslate2 reimplementation of Whisper), not plain
  `openai-whisper`, because CTranslate2's int8 quantization runs
  meaningfully faster on CPU with no confirmed GPU in this environment.
  Model weights download from Hugging Face on first use per size and are
  cached afterward. Testable without downloading real weights: the
  constructor accepts an injectable `model` matching a minimal
  `_WhisperModelLike` protocol (just the one method we call), so
  `tests/test_asr.py` verifies segment-joining, timing, and error handling
  against a fake model instead of a real one.
- **`FanarAuraASR`** -- request/response shape copied *faithfully* from
  the organizers' own baseline notebook
  (`ImageEval2026-tasks/task1/baselines/baseline_task1a_fanar_cascade_colab.ipynb`,
  fetched directly from their GitHub repo to get this right rather than
  guessing): `POST {base_url}/audio/transcriptions`, `Authorization:
  Bearer <key>`, multipart `file` + `model`/`format` form fields. A
  transcript from this backend, once run, is directly comparable to their
  published cascade baseline. Tested via `httpx.MockTransport`, which
  intercepts the request at the real `httpx` transport layer -- this
  verifies the actual request our code sends (method, URL, headers,
  multipart body), not just our own assumptions about `httpx`'s API.
- **`OpenAITranscribeASR`** -- `gpt-4o-transcribe` by default (current
  model name confirmed via OpenAI's docs, not assumed from training data,
  since API model names change). Same `httpx.MockTransport` testing
  approach.

Both API-backed classes raise a clear `RuntimeError` at *construction*
time if their key is missing ("set AYNVQA_FANAR_API_KEY...") rather than
failing confusingly on the first `.transcribe()` call.

## `src/ayn_vqa/artifacts.py`

Generic JSONL cache: `artifacts/<split>/<stage>/<config_key>.jsonl`. Every
stage from here on (ASR now; parsing/evidence/selection later) is a
function over records that costs real time or money and is cheap to
replay from disk -- this is the one place that knows how to read/write
that cache. `append_jsonl` writes one row at a time, flushed immediately,
so a long run interrupted partway through keeps everything already
computed. `read_jsonl_cache` defaults to keying rows by `"record_id"`
(matching the convention every stage dataclass in this codebase uses --
`AudioStat`, `ImageStat`, `MediaCheck`, `Transcript`) but accepts an
`id_field` override for the one exception, the raw dataset's own
`Task1aRecord.id`.

**A real bug this caught during testing:** the first version of
`read_jsonl_cache` hardcoded the key `"id"`, matching `Task1aRecord`'s
field name -- but `Transcript` (like every dataclass this project builds
itself) uses `record_id`. The integration test
(`test_run_asr_bench_reuses_cache_on_second_run`) failed with a `KeyError`
reading its own cache back on a second run, which is exactly the kind of
thing an integration test across the real read/write round-trip catches
that isolated unit tests of either half wouldn't.

## `src/ayn_vqa/run_asr_bench.py`

The `aynvqa-transcribe` CLI: sample a split with M0's `sample_records`
(seeded, never a slice -- see `docs/M0_DATA_AUDIT.md` for why), run one
backend over it, cache every result, print success/failure counts,
latency and transcript-length stats, and a handful of full transcripts.
Deliberately prints the transcripts themselves rather than just
summarizing them: latency and character count are things this process can
measure, but only a native Arabic speaker can judge whether a transcript
is actually *right* -- that judgment is left to the console output, not
automated away with a metric that doesn't exist yet (no hand-transcribed
references -- see "What's still missing" below).

**A real bug this run caught:** on Windows, Rich's console can fall back
to a legacy renderer that writes through the system's ANSI codepage
(`cp1252` here) instead of UTF-8. `cp1252` cannot encode Arabic, so every
sample-transcript log line crashed with `UnicodeEncodeError` -- silently
swallowed by `logging`'s own error handler, so the run still completed
and the cache was still written correctly (that path uses explicit
`encoding="utf-8"` file I/O, untouched by the console bug), but the
console output was unreadable stack traces instead of transcripts. Fixed
in `logging_utils.py` by forcing Rich's `legacy_windows=False` and
reconfiguring `stdout`/`stderr` to UTF-8 -- worth knowing if you add any
other Arabic-printing code path later.

## `src/ayn_vqa/asr_compare.py`

Deliberately separate from `run_asr_bench.py`: that module executes a
backend (costs time/money); this one only ever reads already-cached JSONL
and never calls a model or API, so comparing runs -- e.g.
`whisper-small` vs. `whisper-medium` -- costs nothing and can be re-run
freely as more backends/sizes get benched.

## Real findings (MSA track, `dev`, seeded 50-item sample, seed=42)

```
            n_ok  mean_chars  median_chars  mean_latency_sec
whisper-medium  50      114.4         109.5              7.92
whisper-small   50      114.2         108.0              4.23
```

- **Both sizes transcribed all 50/50 files with no failures.** Transcript
  length is essentially identical between sizes (medium isn't just
  "saying more"); latency is not -- medium is ~1.9x slower per file on
  this CPU. Extrapolated to the full MSA `dev`+`devtest` (1,000 files):
  small ≈ 70 min, medium ≈ 130 min. Both are feasible; the choice comes
  down to accuracy per the spot-check below, not runtime.
- **Medium adds punctuation small mostly doesn't:** question marks
  (`؟`) appear in 14/50 (28%) of medium's transcripts vs. 11/50 (22%) of
  small's. Neither is close to consistent -- most questions end up as one
  run-on clause with no punctuation at all, on both sizes.
- **The "options are" marker (`الخيارات هي`) appears in 41/50 (82%) of
  medium's transcripts vs. 39/50 (78%) of small's.** This is the single
  most useful number in this bench for what comes next: a future
  `TranscriptParser` that splits on this literal phrase would correctly
  segment ~4 in 5 items on either Whisper size, and needs a fallback (or a
  better model) for the rest -- not a guess, a measured number to design
  against.
- **Content-word errors are real and visible even without formal WER,**
  confirming the roadmap's #1 predicted failure mode. Same audio, two
  sizes, same segment: small transcribed *"الشرف"* ("honor" -- wrong word
  entirely) where medium correctly produced *"الشرفة"* ("the balcony" --
  matches the visual content). Medium is not uniformly better, though: on
  a different item small correctly produced *"الغرض"* ("purpose") where
  medium substituted the nonsense *"الغرد"*. Neither size is free of
  named-entity/content-word errors -- exactly why this needs a native
  speaker's judgment, not just these two aggregate numbers, before
  committing to one size for M3.
- **Recommendation:** default to `whisper-medium` for the M3 cascade
  (better punctuation, better marker consistency, and the one qualitative
  regression spotted is balanced by a qualitative win elsewhere) with
  `whisper-small` kept as the fast option for quick iteration during
  development. Revisit once either a hand-transcribed reference set
  enables real WER, or `large-v3`/Fanar/OpenAI get benched against it.

## What's still missing (carried forward, not a defect in M2)

- **No WER scoring yet.** The roadmap's M2 wants ASR quality scored
  against hand-transcribed references; none exist yet. That's a listening
  task only a native MSA speaker can really do well (see the project
  plan's own framing: "your ears are the best analysis tool on this
  team"). Whisper's raw transcripts are cached and ready the moment
  reference transcripts for even 10-20 items exist.
- **`TranscriptParser` (question/option-0/1/2 segmentation) isn't built
  yet.** It needs an LLM to do reliably (the organizers' own Fanar
  baseline doesn't bother -- it hands the *raw* unsegmented transcript
  straight to the VLM with an in-prompt instruction that options come in
  order). One genuinely useful thing this bench already surfaced for that
  future work: every sampled transcript contains the literal phrase
  **"الخيارات هي"** ("the options are") right before the three options --
  a real, consistent lexical anchor, not a guess.
