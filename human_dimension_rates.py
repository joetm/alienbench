"""Print per-dimension departure rates for the human baseline.

Usage (from the repository root):
    python -m alienbench.human_dimension_rates [--config alienbench/config.yaml] [--emit-tex]

Replicates the averaging logic in latex_tables._make_ward_dimensions_table:
  1. Average per-dimension scores across judges per generation.
  2. Average across generations.

With ``--emit-tex``, also writes ``results/tab_human_dimensions.tex``, a
one-row ``tabular`` (the float and caption are authored in the paper).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from alienbench.config import load_config
from alienbench.dimensions import PAPER_TEXT, WARD_DIMENSIONS
from alienbench.latex_tables import _human_dimension_means, load_human_scores


def _emit_human_tex(dim_rates: dict[str, float], results_dir: Path, label: str) -> Path:
    """Write a one-row ``tabular`` of human-baseline departure rates."""
    results_dir.mkdir(parents=True, exist_ok=True)
    headers = " & ".join(PAPER_TEXT[d["id"]]["short"] for d in WARD_DIMENSIONS)
    cells = " & ".join(
        "--" if np.isnan(dim_rates[f"dim_{d['id']}"]) else f"{dim_rates[f'dim_{d['id']}'] * 100:.1f}\\%"
        for d in WARD_DIMENSIONS
    )
    lines = [
        f"\\begin{{tabular}}{{l{'c' * len(WARD_DIMENSIONS)}}}",
        "  \\toprule",
        f"  & {headers} \\\\",
        "  \\midrule",
        f"  {label} & {cells} \\\\",
        "  \\bottomrule",
        "\\end{tabular}",
        "",
    ]
    out_path = results_dir / "tab_human_dimensions.tex"
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="alienbench/config.yaml")
    parser.add_argument("--emit-tex", action="store_true",
                        help="Also write results/tab_human_dimensions.tex.")
    args = parser.parse_args()

    if not Path(args.config).exists():
        sys.exit(f"Config not found: {args.config}")
    cfg = load_config(args.config)

    human_df = load_human_scores(cfg)
    if human_df.empty:
        sys.exit(
            f"No ward score files found for {cfg.human_baseline.id}. "
            "Run 'python -m alienbench extract --human' and "
            "'python -m alienbench score --human' first."
        )
    dim_rates = _human_dimension_means(human_df)
    n_gen = human_df["generation_id"].nunique()
    n_judges = human_df["judge_model"].nunique()

    print(f"Human baseline: {cfg.human_baseline.id}")
    print(f"Generations: {n_gen}  |  Judges: {n_judges} of {len(cfg.judge_models)}\n")
    print(f"{'Dimension':<30}  {'Rate':>6}")
    print("-" * 40)
    for d in WARD_DIMENSIONS:
        v = dim_rates[f"dim_{d['id']}"]
        print(f"{d['label']:<30}  {v * 100:5.1f}%")
    print("-" * 40)
    print(f"{'Mean Ward score (0-10)':<30}  {np.nansum(list(dim_rates.values())):5.2f}")

    if args.emit_tex:
        out_path = _emit_human_tex(dim_rates, Path(cfg.results_dir), cfg.human_baseline.name)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
