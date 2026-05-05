"""AlienBench CLI entry point.

Usage:
    python -m alienbench [--config CONFIG] {generate,extract,score,analyze,latex,run}
    python -m alienbench generate [--config CONFIG]         # equivalent

Stages can run individually or chained via ``run`` (alias: ``all``):

    python -m alienbench run                                 # full pipeline
    python -m alienbench run --from extract                  # resume from stage 2
    python -m alienbench run --to score                      # stop after stage 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

STAGES = ["generate", "extract", "score", "analyze", "latex"]

# Model key used for pre-collected human responses (Prolific study).
# Generation records must already exist on disk; see README.md § Human baseline.
_HUMAN_MODEL_KEY = "human/prolific-baseline"
_HUMAN_STAGES = {"extract", "score"}
_STAGE_DESCRIPTIONS = {
    "generate": "Stage 1: Generate creature descriptions",
    "extract":  "Stage 2: Extract Ward features via LLM judge",
    "score":    "Stage 3: Score Ward departures",
    "analyze":  "Stage 4: Statistical analysis and figures",
    "latex":    "Stage 5: Generate LaTeX tables for results/",
}


def _stage_runner(stage: str):
    """Lazy-import the ``run`` function for a stage."""
    if stage == "generate":
        from alienbench.generate import run
    elif stage == "extract":
        from alienbench.extract import run
    elif stage == "score":
        from alienbench.score import run
    elif stage == "analyze":
        from alienbench.analyze import run
    elif stage == "latex":
        from alienbench.latex_tables import run
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return run


def _run_stage(
    stage: str,
    config_path: str,
    human_models: list[str] | None = None,
) -> None:
    from alienbench.config import load_config
    from alienbench.provenance import write_run_manifest

    runner = _stage_runner(stage)
    if human_models and stage in _HUMAN_STAGES:
        runner(config_path, human_models=human_models)
    else:
        runner(config_path)

    # Record provenance for stages that produce results in results_dir
    if stage in ("analyze", "latex"):
        cfg = load_config(config_path)
        path = write_run_manifest(cfg, Path(cfg.results_dir), stage=stage)
        logger.info("Updated run manifest: %s", path)


_DEFAULT_CONFIG = str(Path(__file__).parent / "config.yaml")


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help=f"Path to config YAML file (default: {_DEFAULT_CONFIG})",
    )


def _add_human_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--human",
        action="store_true",
        default=False,
        help=(
            f"Process pre-collected human responses ({_HUMAN_MODEL_KEY!r}) "
            "through the extract and score stages in addition to the models "
            "listed in config.yaml. Human generation records must already "
            "exist on disk at "
            "data/generations/human__prolific-baseline/baseline/responses.jsonl. "
            "The generate stage is unaffected."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alienbench",
        description=(
            "AlienBench: benchmark for structural biological novelty in LLM "
            "generations, based on Ward's Alien Planet structured-imagination task."
        ),
    )
    # Allow --config on the top-level parser for backwards compatibility
    _add_config_arg(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for stage in STAGES:
        sub = subparsers.add_parser(stage, help=_STAGE_DESCRIPTIONS[stage])
        _add_config_arg(sub)
        if stage in _HUMAN_STAGES:
            _add_human_arg(sub)

    run_sub = subparsers.add_parser(
        "run",
        aliases=["all"],
        help="Run the full pipeline (generate → extract → score → analyze → latex).",
    )
    _add_config_arg(run_sub)
    _add_human_arg(run_sub)
    run_sub.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGES,
        default=STAGES[0],
        help="Stage to start from (inclusive; default: generate).",
    )
    run_sub.add_argument(
        "--to",
        dest="to_stage",
        choices=STAGES,
        default=STAGES[-1],
        help="Stage to stop at (inclusive; default: latex).",
    )

    abl_sub = subparsers.add_parser(
        "ablation",
        help="Run an ablation study (see paper sec:ablations).",
    )
    _add_config_arg(abl_sub)
    abl_sub.add_argument(
        "name",
        choices=["prompt", "dimensions", "judges"],
        help="Which ablation to run.",
    )

    human_sub = subparsers.add_parser(
        "human",
        help="Human validation harness: sample generations, ingest CSV, analyse.",
    )
    _add_config_arg(human_sub)
    human_action = human_sub.add_subparsers(dest="human_action", required=True)

    h_sample = human_action.add_parser("sample", help="Draw a stratified sample and emit a CSV template.")
    _add_config_arg(h_sample)
    h_sample.add_argument("--samples-per-stratum", type=int, default=5)
    h_sample.add_argument("--seed", type=int, default=42)
    h_sample.add_argument("--out", default=None, help="Output CSV path (default: results/human_sample.csv).")

    h_ingest = human_action.add_parser("ingest", help="Validate a filled CSV and store as a human judge.")
    _add_config_arg(h_ingest)
    h_ingest.add_argument("--annotator", required=True, help="Annotator identifier (letters, digits, underscores).")
    h_ingest.add_argument("csv_path", help="Path to the filled CSV.")

    h_analyze = human_action.add_parser("analyze", help="Compute human-judge Krippendorff α.")
    _add_config_arg(h_analyze)
    h_analyze.add_argument(
        "--annotator",
        action="append",
        default=None,
        help="Restrict to a specific annotator id; may be repeated. Default: auto-discover.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    # Subparser --config wins if set, else fall back to top-level default
    config_path = getattr(args, "config", None) or _DEFAULT_CONFIG

    if args.command == "ablation":
        if args.name == "prompt":
            from alienbench.ablation_prompt import run as abl_run
        elif args.name == "dimensions":
            from alienbench.ablation_dimensions import run as abl_run
        elif args.name == "judges":
            from alienbench.ablation_judges import run as abl_run
        else:
            parser.error(f"Unknown ablation: {args.name!r}")
        abl_run(config_path)
        return

    if args.command == "human":
        from alienbench import human

        if args.human_action == "sample":
            human.sample(
                config_path,
                samples_per_stratum=args.samples_per_stratum,
                seed=args.seed,
                out_path=args.out,
            )
        elif args.human_action == "ingest":
            human.ingest(config_path, args.annotator, args.csv_path)
        elif args.human_action == "analyze":
            aliases = (
                [f"{human.HUMAN_PREFIX}{a.strip().lower()}" for a in args.annotator]
                if args.annotator else None
            )
            human.analyze(config_path, human_aliases=aliases)
        else:
            parser.error(f"Unknown human action: {args.human_action!r}")
        return

    human_models = [_HUMAN_MODEL_KEY] if getattr(args, "human", False) else None

    if args.command in ("run", "all"):
        start = STAGES.index(args.from_stage)
        stop = STAGES.index(args.to_stage)
        if start > stop:
            parser.error(
                f"--from {args.from_stage!r} comes after --to {args.to_stage!r} in the pipeline."
            )
        to_run = STAGES[start:stop + 1]
        logger.info("Running stages: %s", " → ".join(to_run))
        for stage in to_run:
            logger.info("=" * 60)
            logger.info("Stage: %s", stage)
            logger.info("=" * 60)
            _run_stage(stage, config_path, human_models=human_models)
        return

    _run_stage(args.command, config_path, human_models=human_models)


if __name__ == "__main__":
    main()
