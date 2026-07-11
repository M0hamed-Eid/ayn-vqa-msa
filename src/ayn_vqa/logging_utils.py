"""One place that configures `logging` for the whole package.

Modules never call `logging.basicConfig` themselves and never `print()` for
anything other than final human-facing summaries -- that's what makes it
possible to silence the audit down to warnings-only, or redirect it to a
file, from a single call site instead of hunting through every module.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger with a Rich console handler and, optionally,
    a plain-text file handler for a permanent per-run record.

    `force=True` lets this be called more than once per process (e.g. a
    notebook cell re-run) without stacking duplicate handlers that would
    otherwise print every message twice.

    This project's data is Arabic. On Windows, Rich's console can fall
    back to a legacy renderer that writes through the system's ANSI
    codepage (`cp1252` on a US/UK machine) rather than UTF-8 -- which
    cannot encode Arabic and crashes the log call outright (the exception
    is swallowed by `logging`'s own error handler, so it doesn't stop the
    run, but every Arabic log line becomes a stack trace instead of text).
    `legacy_windows=False` forces Rich's normal ANSI/UTF-8 path instead;
    reconfiguring `stdout`/`stderr` to UTF-8 covers the same failure mode
    for any plain (non-Rich) output.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    console = Console(legacy_windows=False)
    handlers: list[logging.Handler] = [
        RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=False)
    ]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
