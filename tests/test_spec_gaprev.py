from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features.spec_gaprev import compute_spec_gap_meanrev


def _make_df() -> pd.DataFrame:
    # Build a tiny panel with controlled gaps and ATR%
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    # Base close around 100, deterministic
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    # Open prices create gaps at t=1..4 relative to prev close
    open_ = pd.Series([99.5, 98.0, 101.0, 97.0, 103.0])
    high = pd.Series([101.0, 101.0, 102.0, 100.0, 104.0])
    low = pd.Series([98.0, 97.5, 99.0, 96.0, 101.0])
    vol = pd.Series([1e5, 1e5, 1e5, 1e5, 1e5], dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * len(dates),
            "adj_open": open_.values,
            "adj_high": high.values,
            "adj_low": low.values,
            "adj_close": close.values,
            "adj_volume": vol.values,
            # Set a modest ATR% to make normalized gap informative
            "atr_pct_14": [0.02, 0.02, 0.02, 0.02, 0.02],
        }
    )
    return df


def test_gaprev_sign_and_bounds():
    df = _make_df()
    sc = compute_spec_gap_meanrev(df)
    # Bounds
    assert np.isfinite(sc.fillna(0)).all()
    assert (sc.fillna(0.0) <= 1.0).all()
    assert (sc.fillna(0.0) >= -1.0).all()
    # Index 1: open=98.0 vs prev_close=100 (down gap) -> positive mean reversion bias
    assert sc.iloc[1] > 0.0
    # Larger down gap at index 3 (open=97.0) should yield stronger positive than index 1
    assert sc.iloc[3] > sc.iloc[1]
    # Index 2: open=101.0 (up gap) -> negative mean reversion bias
    assert sc.iloc[2] < 0.0
