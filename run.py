#!/usr/bin/env python3
"""Script entry point that executes the AlienBench benchmark.

This is a thin wrapper around the module CLI in ``alienbench/__main__.py``.
It exists so the benchmark can be executed as a plain script, from any
working directory, without requiring the caller to know that ``alienbench``
must be importable as a package:

    python3 alienbench/run.py                        # full pipeline
    python3 run.py                                   # same, from inside alienbench/
    ./run.py --config test_config.yaml               # smoke test

Called with no arguments, it runs the full pipeline (generate → extract →
score → analyze → latex → paper), which is the ``run`` subcommand of the module CLI.
Any arguments are forwarded verbatim to that CLI, so every subcommand and
flag documented in README.md is available here:

    python3 alienbench/run.py run --from extract --to score
    python3 alienbench/run.py --model GPTSixAstra   # one model, full judge panel
    python3 alienbench/run.py ablation judges
    python3 alienbench/run.py human sample --seed 42

Arguments that name a stage, an ablation, or a human-validation action are
passed through unchanged. A leading option (``--config``, ``--from``,
``--to``, ``--human``) with no subcommand is treated as an option to ``run``,
so ``run.py --config test_config.yaml`` behaves like
``python3 -m alienbench run --config test_config.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

# Subcommands understood by alienbench/__main__.py. Anything else in the
# first argument position means the caller passed options only, and the
# default ``run`` subcommand is inserted.
_COMMANDS = {
    "generate", "extract", "score", "analyze", "latex", "paper",
    "run", "all", "ablation", "human",
}


def _ensure_importable() -> None:
    """Put the directory containing the ``alienbench`` package on ``sys.path``.

    Executing this file as a script places ``alienbench/`` itself on
    ``sys.path``, not its parent. The package uses absolute ``alienbench.*``
    imports throughout, so without this the first import fails. The parent is
    inserted at the front so this checkout takes precedence over any
    installed copy of the same name.
    """
    root = str(_PROJECT_ROOT)
    if sys.path and sys.path[0] == root:
        return
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def _normalize(argv: list[str]) -> list[str]:
    """Insert the default ``run`` subcommand when only options were given."""
    if not argv or argv[0].startswith("-"):
        if "--help" in argv or "-h" in argv:
            return argv
        return ["run", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    _ensure_importable()
    from alienbench.__main__ import main as cli_main

    args = _normalize(list(sys.argv[1:] if argv is None else argv))
    try:
        cli_main(args)
    except KeyboardInterrupt:
        # Every stage is checkpointed, so an interrupted run resumes from
        # where it stopped when the same command is issued again.
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
