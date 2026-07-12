"""Environment-driven settings.

Every path, seed, and threshold that later stages need lives here instead of
being hardcoded at the call site. That's what lets the same code run on a
teammate's machine (different `DATA_ROOT`), in CI (a tiny fixture dataset),
or years from now (M8's GPU box) without touching a single `.py` file --
only `.env`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/ayn_vqa/config.py -> parents[2] is the project root (ayn-vqa-msa/).
# Resolving paths against this, rather than `Path.cwd()`, means `uv run
# aynvqa-audit` behaves the same whether you invoke it from the project
# root, a notebook, or a test runner started from somewhere else.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All fields are overridable via `AYNVQA_<FIELD_NAME>` env vars or `.env`
    (see `.env.example`). Defaults assume the dataset clone is a sibling
    directory named `AynVQA-ArabicNLP26`, which is how this project was set
    up -- override `AYNVQA_DATA_ROOT` if yours lives elsewhere.
    """

    model_config = SettingsConfigDict(
        env_prefix="AYNVQA_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = Path("../AynVQA-ArabicNLP26")
    reports_dir: Path = Path("reports")
    artifacts_dir: Path = Path("artifacts")
    random_seed: int = 42
    sample_grid_n: int = 24
    near_dup_max_distance: int = 4
    log_level: str = "INFO"

    # ASR (M2). `whisper_*` runs locally, no key needed. `fanar_*`/`openai_*`
    # are read here but only used if you set the matching API key -- see
    # docs/M2_ASR_BENCH.md for which backends are actually exercised so far.
    asr_sample_n: int = 50
    # "medium" per M2's own bench: more consistent punctuation and a higher
    # "options are" marker-match rate than "small", at ~1.9x the latency --
    # see docs/M2_ASR_BENCH.md for the numbers behind this default.
    whisper_model_size: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    fanar_api_key: str | None = None
    fanar_base_url: str = "https://api.fanar.qa/v1"
    fanar_stt_model: str = "Fanar-Aura-STT-1"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_transcribe_model: str = "gpt-4o-transcribe"

    # M3: local VLM (Qwen2.5-VL via Ollama) for both transcript parsing and
    # joint-MCQ image answering. No API key -- Ollama serves it locally.
    pipeline_sample_n: int | None = None  # None = the whole split
    ollama_base_url: str = "http://localhost:11434"
    ollama_parse_model: str = "qwen2.5vl:7b"
    ollama_select_model: str = "qwen2.5vl:7b"

    # M4: pipeline-robustness repair escalation. On by default, but a
    # single flag away from an ablation run (`AYNVQA_REPAIR_ENABLED=false`
    # or `--no-repair-enabled`) comparing with/without. `repair_whisper_*`
    # is the stronger ASR model step 2 of the escalation ladder falls back
    # to; float16 (not medium's int8) because this runs on GPU, where it's
    # both faster and more accurate than a CPU-oriented quantization.
    repair_enabled: bool = True
    repair_whisper_model_size: str = "large-v3"
    repair_whisper_device: str = "cuda"
    repair_whisper_compute_type: str = "float16"

    # M5: chain-of-thought select prompt. Off by default -- a 43-item pilot
    # looked promising but this hasn't been validated at full scale yet
    # (see docs/M5_FEWSHOT_RETRIEVAL.md); flip with `--cot-enabled` for an
    # ablation run once it has been.
    cot_enabled: bool = False

    # M5: few-shot exemplar retrieval from `train`. Off by default, same
    # reason as `cot_enabled` -- unvalidated at full scale. `fewshot_k`
    # exemplars are retrieved per query; `fewshot_num_ctx` raises Ollama's
    # context window for those calls only (its own 4096 default 400s on
    # more than one image -- see docs/M5_FEWSHOT_RETRIEVAL.md).
    fewshot_enabled: bool = False
    fewshot_k: int = 2
    fewshot_num_ctx: int = 16384

    def resolved_data_root(self) -> Path:
        return self._resolve(self.data_root)

    def resolved_reports_dir(self) -> Path:
        return self._resolve(self.reports_dir)

    def resolved_artifacts_dir(self) -> Path:
        return self._resolve(self.artifacts_dir)

    @staticmethod
    def _resolve(path: Path) -> Path:
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()


def get_settings() -> Settings:
    """Factory rather than a module-level singleton: tests construct their
    own `Settings(data_root=tmp_path, ...)` without fighting import-time
    caching or environment leakage between test cases.
    """
    return Settings()
