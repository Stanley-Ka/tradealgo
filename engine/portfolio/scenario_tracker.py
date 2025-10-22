from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd


SCENARIO_FEATURE_COLUMNS: Tuple[str, ...] = (
    "mom_sma_5_20",
    "ret_5d",
    "ret_20d",
    "price_z_20",
    "atr_pct_14",
    "vol_z_20",
    "meanrev_20",
    "regime_vol",
    "regime_risk",
    "spec_breakout",
)


@dataclass
class TradeSnapshot:
    """State stored for an active trade so we can evaluate it on close."""

    symbol: str
    trade_id: int
    scenario: str
    open_date: pd.Timestamp
    probabilities: Dict[str, float] = field(default_factory=dict)
    cum_return: float = 1.0


class ScenarioClassifier:
    """Lightweight rule-based mapping from features to a market scenario label."""

    def __init__(
        self,
        momentum_threshold: float = 0.0125,
        breakout_z: float = 1.5,
        breakout_score: float = 0.45,
        low_vol: float = 0.018,
    ) -> None:
        self.momentum_threshold = float(momentum_threshold)
        self.breakout_z = float(breakout_z)
        self.breakout_score = float(breakout_score)
        self.low_vol = float(low_vol)

    def classify(self, row: pd.Series) -> str:
        """Return a scenario label given a feature row (gracefully handles NaNs)."""

        if row is None or row.empty:
            return "unknown"

        def _safe_get(name: str, default: float = 0.0) -> float:
            val = row.get(name, default)
            if pd.isna(val):
                return float(default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        mom = _safe_get("mom_sma_5_20")
        ret5 = _safe_get("ret_5d")
        ret20 = _safe_get("ret_20d")
        price_z = _safe_get("price_z_20")
        vol = abs(_safe_get("vol_z_20"))
        atr = _safe_get("atr_pct_14", default=np.nan)
        regime_risk = _safe_get("regime_risk")
        breakout_raw = _safe_get("spec_breakout")
        breakout_score = _safe_get("spec_breakout_prob", default=np.nan)
        breakout_score = (
            breakout_score if not np.isnan(breakout_score) else breakout_raw
        )

        # Breakout style behaviour: strong extension relative to recent range.
        if (price_z >= self.breakout_z) or (breakout_score >= self.breakout_score):
            return "breakout"
        # Persistent strength: positive momentum and breadth backdrop.
        if mom >= self.momentum_threshold and ret20 >= 0:
            if regime_risk >= 0.25:
                return "uptrend"
            return "uptrend_pullback" if ret5 < 0 else "uptrend"
        # Persistent weakness.
        if mom <= -self.momentum_threshold and ret20 <= 0:
            if regime_risk <= -0.25:
                return "downtrend"
            return "downtrend_bounce" if ret5 > 0 else "downtrend"
        # Low-momentum, low-volatility cluster.
        if abs(mom) < self.momentum_threshold and (
            vol < 0.4 or (not np.isnan(atr) and atr <= self.low_vol)
        ):
            return "consolidation"
        # Mean-reversion style washout: deeply oversold but momentum improving.
        meanrev = _safe_get("meanrev_20")
        if meanrev > 0.02 and ret5 > 0:
            return "reversal_setup"
        return "range"


class ScenarioPerformanceTracker:
    """Tracks scenarios and per-specialist win rates across trades."""

    def __init__(
        self, classifier: ScenarioClassifier, win_threshold: float = 0.0
    ) -> None:
        self.classifier = classifier
        self.win_threshold = float(win_threshold)
        self._open: Dict[Tuple[str, int], TradeSnapshot] = {}
        self._performance: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.completed_trades: list[dict] = []

    @property
    def performance(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        return self._performance

    @staticmethod
    def _extract_probabilities(row: pd.Series) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if row is None or row.empty:
            return out
        # First collect explicit probability columns.
        for col, val in row.items():
            if not isinstance(col, str) or not col.startswith("spec_"):
                continue
            if col.endswith("_prob"):
                base = col[:-5]
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if np.isnan(fval):
                    continue
                out[base] = fval
        # Fall back to raw specialist scores if probability not available.
        for col, val in row.items():
            if not isinstance(col, str) or not col.startswith("spec_"):
                continue
            if col.endswith("_prob"):
                continue
            base = col
            if base in out:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if np.isnan(fval):
                continue
            out[base] = 0.5 * (fval + 1.0)
        return out

    def _get_row(self, features_by_symbol: pd.DataFrame, symbol: str) -> pd.Series:
        if symbol not in features_by_symbol.index:
            return pd.Series(dtype=float)
        row = features_by_symbol.loc[symbol]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _ensure_performance_entry(
        self, scenario: str, specialist: str
    ) -> Dict[str, int]:
        scen_map = self._performance.setdefault(scenario, {})
        return scen_map.setdefault(specialist, {"wins": 0, "trades": 0})

    def _close_trade(self, key: Tuple[str, int], as_of: pd.Timestamp) -> dict:
        snapshot = self._open.pop(key, None)
        if snapshot is None:
            return {
                "return": np.nan,
                "win_rates": {},
                "win": False,
                "scenario": "unknown",
            }
        total_return = snapshot.cum_return - 1.0
        win = total_return > self.win_threshold
        win_rates: Dict[str, float] = {}
        for spec_name, _ in snapshot.probabilities.items():
            perf = self._ensure_performance_entry(snapshot.scenario, spec_name)
            perf["trades"] += 1
            if win:
                perf["wins"] += 1
            if perf["trades"] > 0:
                win_rates[spec_name] = perf["wins"] / perf["trades"]
        self.completed_trades.append(
            {
                "symbol": snapshot.symbol,
                "trade_id": snapshot.trade_id,
                "scenario": snapshot.scenario,
                "open_date": snapshot.open_date.normalize(),
                "close_date": as_of.normalize(),
                "return": total_return,
                "win": win,
                **{f"prob_{k}": v for k, v in snapshot.probabilities.items()},
            }
        )
        return {
            "return": total_return,
            "win_rates": win_rates,
            "win": win,
            "scenario": snapshot.scenario,
        }

    def process_day(
        self,
        decision_date: pd.Timestamp,
        log_rows: pd.DataFrame,
        feature_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        if log_rows.empty:
            return log_rows
        # Ensure deterministic symbol lookup
        feature_rows = feature_rows.drop_duplicates(subset="symbol", keep="first")
        features_by_symbol = (
            feature_rows.set_index("symbol")
            if "symbol" in feature_rows.columns
            else pd.DataFrame()
        )

        out = log_rows.copy()
        if "trade_scenario" not in out.columns:
            out["trade_scenario"] = ""
        if "trade_entry_date" not in out.columns:
            out["trade_entry_date"] = pd.NaT
        if "trade_cum_return" not in out.columns:
            out["trade_cum_return"] = np.nan

        for idx, row in out.iterrows():
            trade_id = int(row.get("trade_id", 0) or 0)
            if trade_id <= 0:
                continue
            symbol = str(row.get("symbol"))
            key = (symbol, trade_id)
            feature_row = self._get_row(features_by_symbol, symbol)
            snapshot = self._open.get(key)
            if snapshot is None and row.get("action") == "open":
                scenario = self.classifier.classify(feature_row)
                probs = self._extract_probabilities(feature_row)
                snapshot = TradeSnapshot(
                    symbol=symbol,
                    trade_id=trade_id,
                    scenario=scenario,
                    open_date=decision_date,
                    probabilities=probs,
                    cum_return=1.0,
                )
                self._open[key] = snapshot
            elif snapshot is None:
                scenario = self.classifier.classify(feature_row)
                probs = self._extract_probabilities(feature_row)
                snapshot = TradeSnapshot(
                    symbol=symbol,
                    trade_id=trade_id,
                    scenario=scenario,
                    open_date=decision_date,
                    probabilities=probs,
                    cum_return=1.0,
                )
                self._open[key] = snapshot
            if snapshot is None:
                continue
            out.at[idx, "trade_scenario"] = snapshot.scenario
            out.at[idx, "trade_entry_date"] = snapshot.open_date
            fret = row.get("fret_1d_next")
            if pd.notna(fret):
                try:
                    snapshot.cum_return *= 1.0 + float(fret)
                except (TypeError, ValueError):
                    pass
            out.at[idx, "trade_cum_return"] = snapshot.cum_return - 1.0
            if row.get("action") == "close":
                close_info = self._close_trade(key, decision_date)
                out.at[idx, "trade_cum_return"] = close_info["return"]
                for spec_name, win_rate in close_info["win_rates"].items():
                    col = f"win_rate_{spec_name}"
                    if col not in out.columns:
                        out[col] = np.nan
                    out.at[idx, col] = win_rate
        return out

    def completed_trades_frame(self) -> pd.DataFrame:
        if not self.completed_trades:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "trade_id",
                    "scenario",
                    "open_date",
                    "close_date",
                    "return",
                    "win",
                ]
            )
        return pd.DataFrame(self.completed_trades)


__all__ = [
    "ScenarioClassifier",
    "ScenarioPerformanceTracker",
    "SCENARIO_FEATURE_COLUMNS",
]
