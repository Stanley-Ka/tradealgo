from __future__ import annotations

import os
import pickle
import pandas as pd
import numpy as np

from engine.tools.entry_loop import main as entry_main


class _DummyModel:
    def predict_proba(self, X):  # noqa: N802
        import numpy as _np

        n = X.shape[0]
        p1 = _np.full((n, 1), 0.9, dtype=float)
        p0 = 1.0 - p1
        return _np.concatenate([p0, p1], axis=1)


def _make_features(tmp_path: str, syms=("AAA", "BBB")) -> str:
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    rows = []
    rng = np.random.default_rng(0)
    for s in syms:
        price = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
        high = price + rng.random(len(dates))
        low = price - rng.random(len(dates))
        open_ = price + rng.normal(0, 0.2, size=len(dates))
        close = price + rng.normal(0, 0.2, size=len(dates))
        vol = rng.integers(1e5, 5e5, size=len(dates))
        for d, o, h, l, c, v in zip(dates, open_, high, low, close, vol):
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "adj_open": o,
                    "adj_high": h,
                    "adj_low": l,
                    "adj_close": c,
                    "adj_volume": float(v),
                    "atr_pct_14": 0.02,
                }
            )
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_path, "features.parquet")
    df.to_parquet(path, index=False)
    return path


def _make_universe(tmp_path: str, syms=("AAA", "BBB")) -> str:
    p = os.path.join(tmp_path, "uni.txt")
    with open(p, "w", encoding="utf-8") as f:
        for s in syms:
            f.write(f"{s}\n")
    return p


def _make_model(tmp_path: str, feats: list[str] | None = None) -> str:
    p = os.path.join(tmp_path, "model.pkl")
    payload = {"model": _DummyModel(), "features": feats or []}
    with open(p, "wb") as f:
        pickle.dump(payload, f)
    return p


def test_entry_loop_writes_positions(tmp_path):
    feat = _make_features(str(tmp_path))
    uni = _make_universe(str(tmp_path))
    mdl = _make_model(str(tmp_path))
    pos_csv = os.path.join(str(tmp_path), "positions.csv")
    log_csv = os.path.join(str(tmp_path), "entry_log.csv")
    # Run with threshold so _DummyModel passes
    args = [
        "--features",
        feat,
        "--model-pkl",
        mdl,
        "--universe-file",
        uni,
        "--top-k",
        "1",
        "--entry-threshold",
        "0.6",
        "--positions-csv",
        pos_csv,
        "--decision-log-csv",
        log_csv,
    ]
    entry_main(args)
    assert os.path.exists(pos_csv)
    dfp = pd.read_csv(pos_csv)
    assert len(dfp) == 1
    assert set(["symbol", "entry_date", "entry_price", "shares"]).issubset(dfp.columns)
