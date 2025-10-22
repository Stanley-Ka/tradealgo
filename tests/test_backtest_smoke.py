from __future__ import annotations

import pandas as pd
import numpy as np

from engine.backtest.simple_daily import daily_backtest


def test_daily_backtest_smoke(tmp_path):
    # Build small synthetic features + predictions parquet files
    dates = pd.date_range("2021-01-01", periods=40, freq="B")
    symbols = ["AAA", "BBB", "CCC"]
    rows_f = []
    rows_p = []
    rng = np.random.default_rng(0)
    for d in dates:
        for s in symbols:
            fret = rng.normal(0, 0.01)
            rows_f.append({"date": d, "symbol": s, "fret_1d": fret})
            prob = 0.5 + 0.2 * (symbols.index(s) - 1) / 2.0  # BBB lowest, AAA highest
            rows_p.append(
                {"date": d, "symbol": s, "meta_prob": float(np.clip(prob, 0, 1))}
            )
    fdf = pd.DataFrame(rows_f)
    pdf = pd.DataFrame(rows_p)
    fpath = tmp_path / "f.parquet"
    ppath = tmp_path / "p.parquet"
    fdf.to_parquet(fpath, index=False)
    pdf.to_parquet(ppath, index=False)

    res = daily_backtest(
        str(fpath), str(ppath), prob_col="meta_prob", top_k=2, cost_bps=5.0
    )
    assert not res.empty
    for col in ["date", "gross_ret", "turnover", "cost", "net_ret", "equity"]:
        assert col in res.columns
