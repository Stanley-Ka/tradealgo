from __future__ import annotations

import os
import pickle
import pandas as pd

from engine.tools.position_monitor import main as monitor_main


class _LowProbModel:
    def predict_proba(self, X):  # noqa: N802
        import numpy as _np

        n = X.shape[0]
        p1 = _np.full((n, 1), 0.4, dtype=float)
        p0 = 1.0 - p1
        return _np.concatenate([p0, p1], axis=1)


def _make_features(tmp_path: str, sym="AAA") -> str:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    rows = []
    for d in dates:
        rows.append(
            {
                "date": d,
                "symbol": sym,
                "adj_open": 100.0,
                "adj_high": 101.0,
                "adj_low": 99.0,
                "adj_close": 100.0,
                "adj_volume": 100000.0,
                "atr_pct_14": 0.02,
            }
        )
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_path, "features.parquet")
    df.to_parquet(path, index=False)
    return path


def _make_model(tmp_path: str) -> str:
    p = os.path.join(tmp_path, "model.pkl")
    payload = {"model": _LowProbModel(), "features": []}
    with open(p, "wb") as f:
        pickle.dump(payload, f)
    return p


def test_monitor_prob_hysteresis(tmp_path):
    feat = _make_features(str(tmp_path))
    mdl = _make_model(str(tmp_path))
    pos_csv = os.path.join(str(tmp_path), "positions.csv")
    exits_csv = os.path.join(str(tmp_path), "exits.csv")
    state_csv = os.path.join(str(tmp_path), "mon_state.csv")
    # Seed one open position
    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "entry_date": ["2024-01-01"],
            "entry_price": [100.0],
            "shares": [10],
        }
    ).to_csv(pos_csv, index=False)
    # First run: should increment counter, no exit
    monitor_main(
        [
            "--features",
            feat,
            "--positions-csv",
            pos_csv,
            "--exits-csv",
            exits_csv,
            "--model-pkl",
            mdl,
            "--prob-exit-thresh",
            "0.45",
            "--prob-exit-consecutive",
            "2",
            "--monitor-state-csv",
            state_csv,
        ]
    )
    assert os.path.exists(state_csv)
    # Positions should remain
    assert os.path.exists(pos_csv)
    assert not os.path.exists(exits_csv)
    # Second run: counter reaches 2 -> exit
    monitor_main(
        [
            "--features",
            feat,
            "--positions-csv",
            pos_csv,
            "--exits-csv",
            exits_csv,
            "--model-pkl",
            mdl,
            "--prob-exit-thresh",
            "0.45",
            "--prob-exit-consecutive",
            "2",
            "--monitor-state-csv",
            state_csv,
        ]
    )
    # Positions removed and exit logged
    assert os.path.exists(exits_csv)
    df_ex = pd.read_csv(exits_csv)
    assert (df_ex["symbol"] == "AAA").any()
