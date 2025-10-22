from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features.spec_news import compute_spec_news


def test_windowed_news_decay_single_day():
    # Target day
    tdate = pd.Timestamp("2024-01-10")
    df = pd.DataFrame(
        {
            "date": [tdate] * 2,
            "symbol": ["AAA", "BBB"],
            "adj_open": [100.0, 100.0],
            "adj_high": [101.0, 101.0],
            "adj_low": [99.0, 99.0],
            "adj_close": [100.0, 100.0],
            "adj_volume": [100000.0, 120000.0],
        }
    )
    # Sentiment history: AAA has +1 yesterday and -1 three days ago -> should be positive with decay
    # BBB has only older negative -> near zero negative
    sent = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-09"),
                pd.Timestamp("2024-01-07"),
                pd.Timestamp("2024-01-07"),
            ],
            "symbol": ["AAA", "AAA", "BBB"],
            "sentiment": [1.0, -1.0, -0.8],
        }
    )
    s = compute_spec_news(
        df, sent, params={"window_days": 5, "half_life_days": 1.0, "agg": "mean"}
    )
    assert len(s) == 2
    # AAA: should be > 0 (recent +1 dominates older -1 with decay)
    assert s.iloc[0] > 0.0
    # BBB: only older negative -> negative but bounded
    assert s.iloc[1] < 0.0
    assert np.isfinite(s.fillna(0)).all()
    assert (s <= 1.0).all() and (s >= -1.0).all()
