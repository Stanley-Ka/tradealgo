from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features.baseline import compute_baseline_features
from engine.features.specialists import compute_specialist_scores


def _toy_symbol(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = np.cumsum(rng.normal(0, 1, size=n)) + 100
    high = price + rng.random(n)
    low = price - rng.random(n)
    open_ = price + rng.normal(0, 0.2, size=n)
    close = price + rng.normal(0, 0.2, size=n)
    vol = rng.integers(1e5, 5e5, size=n)
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST",
            "adj_open": open_,
            "adj_high": high,
            "adj_low": low,
            "adj_close": close,
            "adj_volume": vol,
        }
    )
    return df


def test_baseline_features_and_specialists_shapes():
    df = _toy_symbol()
    base = compute_baseline_features(df)
    # Required baseline columns exist
    for col in [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "mom_sma_5_20",
        "mom_20d",
        "price_z_20",
        "meanrev_20",
        "vol_z_20",
        "atr_pct_14",
    ]:
        assert col in base.columns
    # Specialists produce expected columns and ranges
    specs = compute_specialist_scores(base)
    for sc in ["spec_pattern", "spec_technical", "spec_sequence", "spec_nlp"]:
        assert sc in specs.columns
        s = specs[sc].astype(float)
        assert len(s) == len(base)
        assert np.isfinite(s.fillna(0).values).all()
        assert (s.fillna(0.0) <= 1.0).all()
        assert (s.fillna(0.0) >= -1.0).all()
