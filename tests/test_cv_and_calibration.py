from __future__ import annotations

import numpy as np
import pandas as pd

from engine.models.cv import rolling_year_splits
from engine.models.calibration import fit_platt, fit_isotonic


def test_rolling_year_splits_monotone():
    dates = pd.date_range("2015-01-01", "2023-12-31", freq="B")
    # Create 9 years roughly; expect folds after 3-year burn-in
    folds = rolling_year_splits(pd.Series(dates), min_train_years=3)
    assert len(folds) >= 4
    for f in folds:
        # train dates must be strictly before val dates
        assert f.train_idx.min() < f.val_idx.min()
        assert set(f.train_idx).isdisjoint(set(f.val_idx))


def test_calibration_monotonic_probabilities():
    # Linearly spaced scores with increasing labels probability
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 500)
    p = (x + 1) / 2
    y = rng.binomial(1, p)
    pl = fit_platt(x, y)
    iso = fit_isotonic(x, y)
    xp = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    pp = pl.apply(xp)
    ip = iso.apply(xp)
    assert (np.diff(pp) >= -1e-6).all()
    assert (np.diff(ip) >= -1e-6).all()
    assert (pp >= 0).all() and (pp <= 1).all()
    assert (ip >= 0).all() and (ip <= 1).all()
