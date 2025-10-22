from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features.intraday_baseline import compute_intraday_baseline
from engine.features.specialists import compute_specialist_scores


def _make_intraday_bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01 09:30", periods=n, freq="min")
    price = 100 + np.cumsum(rng.normal(0, 0.1, size=n))
    high = price + rng.random(n) * 0.2
    low = price - rng.random(n) * 0.2
    open_ = price + rng.normal(0, 0.05, size=n)
    close = price + rng.normal(0, 0.05, size=n)
    vol = rng.integers(1e3, 5e3, size=n)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def test_intraday_specialists_run_and_ranges():
    bars = _make_intraday_bars(240)
    base = compute_intraday_baseline(bars, symbol="TEST")
    specs = compute_specialist_scores(base)
    # Ensure expected specialists exist and are in [-1,1]
    got = [c for c in specs.columns if c.startswith("spec_")]
    assert got, "no specialists computed"
    for sc in got:
        s = specs[sc].astype(float)
        assert np.isfinite(s.fillna(0)).all()
        assert (s.fillna(0.0) <= 1.0).all()
        assert (s.fillna(0.0) >= -1.0).all()
