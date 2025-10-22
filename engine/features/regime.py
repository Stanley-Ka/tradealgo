from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily regime features from cross-sectional data.

    Returns a DataFrame with unique 'date' and columns:
      - regime_vol: cross-sectional ATR% (14) z-score per day
      - regime_risk: breadth (share of positive 20D momentum) mapped to [-1,1]
    """
    req = {"date", "symbol"}
    if not req.issubset(df.columns):
        raise ValueError(
            "compute_regime_features_daily requires date and symbol columns"
        )
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()

    # Volatility regime proxy: cross-sectional average of ATR% per day, z-scored over time
    if "atr_pct_14" not in tmp.columns:
        # no proxy; create zeros
        daily_vol = tmp.groupby("date").size().reset_index(name="n")
        daily_vol["regime_vol_raw"] = 0.0
    else:
        cs_vol = tmp.groupby(["date"])["atr_pct_14"].mean().rename("regime_vol_raw")
        daily_vol = cs_vol.reset_index()

    def _z_over_time(x: pd.Series) -> pd.Series:
        mu = x.rolling(252, min_periods=20).mean()
        sd = x.rolling(252, min_periods=20).std(ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (x - mu) / sd
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

    daily_vol["regime_vol"] = (
        _z_over_time(daily_vol["regime_vol_raw"]).clip(-2, 2) / 2.0
    )

    # Risk appetite proxy: breadth of positive momentum (mom_sma_5_20>0) per day scaled to [-1,1]
    if "mom_sma_5_20" in tmp.columns:
        br = tmp.assign(pos=(tmp["mom_sma_5_20"] > 0).astype(int))
        daily = br.groupby("date")["pos"].mean().rename("breadth")
        risk = (daily * 2.0 - 1.0).clip(-1.0, 1.0).rename("regime_risk")
    else:
        # fallback: use 20D return sign breadth
        if "ret_20d" in tmp.columns:
            br = tmp.assign(pos=(tmp["ret_20d"] > 0).astype(int))
            daily = br.groupby("date")["pos"].mean().rename("breadth")
            risk = (daily * 2.0 - 1.0).clip(-1.0, 1.0).rename("regime_risk")
        else:
            # build a zero series aligned to available dates
            idx = pd.Index(daily_vol["date"].unique(), name="date")
            risk = pd.Series(0.0, index=idx, dtype=float).rename("regime_risk")

    if isinstance(risk, pd.Series):
        risk.index.name = "date"
        risk = risk.reset_index()
    else:
        # ensure we have the expected columns when risk already a DataFrame
        if "date" not in risk.columns:
            risk = risk.rename(columns={risk.columns[0]: "date"})

    out = daily_vol[["date", "regime_vol"]].merge(risk, on="date", how="left")
    return out
