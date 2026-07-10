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
    random_seed: int = 42
    sample_grid_n: int = 24
    near_dup_max_distance: int = 4
    log_level: str = "INFO"

    def resolved_data_root(self) -> Path:
        return self._resolve(self.data_root)

    def resolved_reports_dir(self) -> Path:
        return self._resolve(self.reports_dir)

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
