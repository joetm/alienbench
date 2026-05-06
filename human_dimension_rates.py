"""Print per-dimension departure rates for the human baseline.

Usage (from the alienbench/ directory):
    python human_dimension_rates.py [--config config.yaml] [--emit-tex]

Replicates the averaging logic in latex_tables._make_ward_dimensions_table:
  1. Average per-dimension scores across judges per generation.
  2. Average across generations.

With ``--emit-tex``, also writes ``results/tab_human_dimensions.tex`` for
direct \\input in Section 4.2 of the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HUMAN_MODEL = "human/prolific-baseline"

DIMENSIONS = [
    ("symmetry",       "Body Symmetry"),
    ("sensory_organs", "Sensory Organs"),
    ("locomotion",     "Locomotion"),
    ("body_plan",      "Body Plan"),
    ("skin_covering",  "Skin / Body Covering"),
    ("reproduction",   "Reproduction"),
    ("metabolism",     "Metabolism / Energy Source"),
    ("communication",  "Communication"),
    ("habitat",        "Habitat"),
    ("cognition",      "Cognitive Architecture"),
]

# Compact labels matching the radar tick labels in `radar.py` so the
# emitted LaTeX table aligns visually with Figure 1.
_RADAR_LABELS = {
    "symmetry":       "Symmetry",
    "sensory_organs": "Sensing",
    "locomotion":     "Locomotion",
    "body_plan":      "Body plan",
    "skin_covering":  "Covering",
    "reproduction":   "Reproduction",
    "metabolism":     "Metabolism",
    "communication":  "Communication",
    "habitat":        "Habitat",
    "cognition":      "Cognition",
}


def _model_dir(model_id: str) -> str:
    return model_id.replace("/", "__")


def _load_config(config_path: Path) -> dict:
    try:
        import yaml
        with config_path.open() as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    # Minimal fallback: extract data_dir and judge_models with regex-free parsing
    text = config_path.read_text()
    cfg: dict = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data_dir:"):
            cfg["data_dir"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
        if stripped.startswith("results_dir:"):
            cfg["results_dir"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
        if stripped.startswith("judge_models:"):
            cfg["judge_models"] = []
    # collect judge_models list items
    in_judges = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("judge_models:"):
            in_judges = True
            continue
        if in_judges:
            if stripped.startswith("- "):
                cfg.setdefault("judge_models", []).append(
                    stripped[2:].split("#")[0].strip()
                )
            elif stripped and not stripped.startswith("#"):
                in_judges = False
    return cfg


def _iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def _emit_human_tex(dim_rates: dict[str, float], n_gen: int, results_dir: Path) -> Path:
    """Write a compact one-row LaTeX table of human-baseline departure rates.

    Output path: ``<results_dir>/tab_human_dimensions.tex``. Dimension labels
    match the axes of ``fig:ward_radar`` so the table reads as a numeric
    annotation of the radar's missing baseline shape.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    headers = " & ".join(_RADAR_LABELS[d] for d, _ in DIMENSIONS)

    def _cell(dim_id: str) -> str:
        v = dim_rates.get(dim_id, float("nan"))
        return "---" if np.isnan(v) else f"{v * 100:.1f}\\%"

    cells = " & ".join(_cell(d) for d, _ in DIMENSIONS)

    n_cols = len(DIMENSIONS)
    col_spec = "l" + "c" * n_cols
    lines = [
        "\\begin{table}[htb]",
        "  \\centering",
        "  \\caption{Human baseline Ward feature departure rate (\\%) per dimension, "
        f"computed from the $N{{=}}{n_gen}$ Prolific generations after averaging across "
        "judges per generation and then across generations. Dimension labels match the "
        "axes of \\autoref{fig:ward_radar}.}",
        "  \\small",
        "  \\label{tab:human_dimensions}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    & {headers} \\\\",
        "    \\midrule",
        f"    Human & {cells} \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "  }",
        "\\end{table}",
        "",
    ]
    out_path = results_dir / "tab_human_dimensions.tex"
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--emit-tex",
        action="store_true",
        help="Also write results/tab_human_dimensions.tex for paper Section 4.2.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    cfg = _load_config(config_path)
    data_dir = Path(cfg.get("data_dir", "data"))
    judge_models: list[str] = cfg.get("judge_models", [])

    if not judge_models:
        sys.exit("No judge_models found in config.")

    dim_ids = [d[0] for d in DIMENSIONS]

    # Collect per-generation, per-judge dimension scores.
    # records[generation_id][judge] = {dim_id: 0|1, ...}
    records: dict[str, dict[str, dict[str, float]]] = {}

    human_dir = _model_dir(HUMAN_MODEL)
    found_any = False

    for judge in judge_models:
        judge_dir = _model_dir(judge)
        # Human scores live under data/scores/<judge>/<human_model>/baseline/ward_scores.jsonl
        scores_path = data_dir / "scores" / judge_dir / human_dir / "baseline" / "ward_scores.jsonl"
        if not scores_path.exists():
            print(f"  [warn] not found: {scores_path}", file=sys.stderr)
            continue
        found_any = True
        for rec in _iter_jsonl(scores_path):
            gen_id = rec["generation_id"]
            per_dim = rec.get("per_dimension", {})
            records.setdefault(gen_id, {})[judge] = {
                d: float(per_dim.get(d, 0)) for d in dim_ids
            }

    if not found_any:
        sys.exit(
            f"No ward score files found for {HUMAN_MODEL}. "
            "Run 'python -m alienbench extract --human' and "
            "'python -m alienbench score --human' first."
        )

    n_gen = len(records)
    if n_gen == 0:
        sys.exit("No generation records loaded.")

    # Step 1: average across judges per generation
    gen_means: dict[str, dict[str, float]] = {}
    for gen_id, judge_scores in records.items():
        gen_means[gen_id] = {}
        for dim in dim_ids:
            vals = [judge_scores[j][dim] for j in judge_scores if dim in judge_scores[j]]
            gen_means[gen_id][dim] = float(np.mean(vals)) if vals else float("nan")

    # Step 2: average across generations
    dim_rates: dict[str, float] = {}
    for dim in dim_ids:
        vals = [gen_means[g][dim] for g in gen_means if not np.isnan(gen_means[g][dim])]
        dim_rates[dim] = float(np.mean(vals)) if vals else float("nan")

    # Output
    n_judges_found = len({j for g in records.values() for j in g})
    print(f"Human baseline: {HUMAN_MODEL}")
    print(f"Generations: {n_gen}  |  Judges: {n_judges_found} of {len(judge_models)}")
    print()
    print(f"{'Dimension':<30}  {'Rate':>6}")
    print("-" * 40)
    for dim_id, label in DIMENSIONS:
        v = dim_rates.get(dim_id, float("nan"))
        rate_str = f"{v * 100:5.1f}%" if not np.isnan(v) else "  n/a"
        print(f"{label:<30}  {rate_str:>6}")
    print("-" * 40)
    overall = np.nanmean(list(dim_rates.values()))
    print(f"{'Mean Ward score (0–10)':<30}  {overall * 10:5.2f}")

    if args.emit_tex:
        results_dir = Path(cfg.get("results_dir", "results"))
        out_path = _emit_human_tex(dim_rates, n_gen, results_dir)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
