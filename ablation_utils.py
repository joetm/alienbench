"""Shared helpers for ablation modules.

Centralises the judges-then-generations-then-models aggregation convention
and the rank-equality / Spearman predicates used by more than one ablation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats


RANK_ATOL = 1e-9


def ranks_equal(a: pd.Series, b: pd.Series) -> bool:
    """True iff the two per-model score vectors induce the same ranking.

    Uses average ranks, so ties map to ties. Returns False if fewer than two
    models are shared between ``a`` and ``b``.
    """
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return False
    ra = stats.rankdata(a.loc[common].values, method="average")
    rb = stats.rankdata(b.loc[common].values, method="average")
    return bool(np.allclose(ra, rb, atol=RANK_ATOL))


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman ρ between ``a`` and ``b`` on their shared index.

    Returns ``NaN`` when fewer than three models overlap or when either
    vector has zero variance on the shared index.
    """
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return float("nan")
    if a.loc[common].nunique() < 2 or b.loc[common].nunique() < 2:
        return float("nan")
    return float(stats.spearmanr(a.loc[common].values, b.loc[common].values).statistic)


def n_rank_changes(a: pd.Series, b: pd.Series) -> int:
    """Count models whose rank differs between ``a`` and ``b``.

    Returns -1 when fewer than two models overlap.
    """
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return -1
    ra = stats.rankdata(a.loc[common].values, method="average")
    rb = stats.rankdata(b.loc[common].values, method="average")
    return int(np.sum(ra != rb))
