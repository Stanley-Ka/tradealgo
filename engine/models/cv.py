from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    name: str


def rolling_year_splits(dates: pd.Series, min_train_years: int = 3) -> List[Fold]:
    """Create one validation fold per calendar year after an initial training burn-in.

    For year Y, train uses all data with year < Y, and validation uses dates within year Y.
    Requires at least `min_train_years` before the first fold is emitted.
    """
    s = pd.to_datetime(dates).copy()
    years = s.dt.year
    unique_years = sorted(years.unique())
    folds: List[Fold] = []
    for i, y in enumerate(unique_years):
        if i < min_train_years:
            continue
        train_mask = years < y
        val_mask = years == y
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        folds.append(
            Fold(
                train_idx=np.where(train_mask.values)[0],
                val_idx=np.where(val_mask.values)[0],
                name=f"Y{y}",
            )
        )
    return folds


def time_kfold_purged(
    dates: pd.Series,
    n_splits: int = 5,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> List[Fold]:
    """Purged time-based K-Fold splitting.

    - Splits the sorted index into contiguous folds by date.
    - For each fold k, validation is fold k. Training is all other folds
      with a purge window around the validation block removed and an
      embargo after validation to reduce leakage.
    - purge_days removes samples within +/- purge_days of val boundaries.
    - embargo_days removes samples in the embargo period immediately after the val end.
    """
    s = pd.to_datetime(dates).copy()
    order = np.argsort(s.values)
    s_sorted = s.iloc[order].reset_index(drop=True)
    n = len(s_sorted)
    if n_splits < 2 or n_splits > n:
        raise ValueError("n_splits must be between 2 and number of samples")

    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    boundaries = np.cumsum(fold_sizes)
    starts = np.concatenate(([0], boundaries[:-1]))
    ends = boundaries  # exclusive

    folds: List[Fold] = []
    for k, (a, b) in enumerate(zip(starts, ends)):
        val_idx_sorted = np.arange(a, b)
        val_start_date = s_sorted.iloc[a]
        val_end_date = s_sorted.iloc[b - 1]

        # Train = all except val block
        tr_mask_sorted = np.ones(n, dtype=bool)
        tr_mask_sorted[a:b] = False

        # Purge window around validation boundaries
        if purge_days > 0:
            left_cut = val_start_date - pd.Timedelta(days=purge_days)
            right_cut = val_end_date + pd.Timedelta(days=purge_days)
            tr_mask_sorted &= ~((s_sorted >= left_cut) & (s_sorted <= right_cut)).values

        # Embargo after validation
        if embargo_days > 0:
            embargo_end = val_end_date + pd.Timedelta(days=embargo_days)
            tr_mask_sorted &= ~(
                (s_sorted > val_end_date) & (s_sorted <= embargo_end)
            ).values

        tr_idx_sorted = np.where(tr_mask_sorted)[0]

        # Map back to original indices
        tr_idx = order[tr_idx_sorted]
        va_idx = order[val_idx_sorted]
        folds.append(Fold(train_idx=tr_idx, val_idx=va_idx, name=f"K{k+1}"))
    return folds
