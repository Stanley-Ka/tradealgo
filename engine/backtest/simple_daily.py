"""Simple daily backtester for probability-based signals.

Assumptions:
- Use next-day simple return `fret_1d` from the features dataset.
- Select top-K names by probability each day, equal-weighted.
- Apply costs via turnover (bps per 1.0 of weight change).

Example:
  python -m engine.backtest.simple_daily \
    --features data/datasets/features_daily_1D.parquet \
    --pred data/datasets/meta_predictions.parquet --prob-col meta_prob \
    --top-k 20 --cost-bps 5
"""

from __future__ import annotations

import argparse
import numpy as np
import os
import pandas as pd
from typing import List, Optional

from ..data.store import storage_root
from ..portfolio.scenario_tracker import (
    SCENARIO_FEATURE_COLUMNS,
    ScenarioClassifier,
    ScenarioPerformanceTracker,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simple daily top-K backtest using probabilities"
    )
    p.add_argument(
        "--config", type=str, default="", help="YAML preset/config with default paths"
    )
    p.add_argument(
        "--features",
        required=False,
        default="",
        help="Features parquet with fret_1d and date/symbol",
    )
    p.add_argument(
        "--pred",
        required=True,
        help="Predictions parquet with date/symbol and prob column",
    )
    p.add_argument(
        "--prob-col",
        type=str,
        default="meta_prob",
        help="Probability column name in pred file",
    )
    p.add_argument(
        "--top-k", type=int, default=20, help="Number of names to hold each day"
    )
    p.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="Cost per unit turnover (basis points)",
    )
    # Optional cost model with spread proxy
    p.add_argument(
        "--cost-model",
        choices=["flat", "spread"],
        default="flat",
        help="Cost model: flat bps or add spread proxy per trade",
    )
    p.add_argument(
        "--spread-k",
        type=float,
        default=1e8,
        help="Spread proxy scale: spread_bps = min(cap, max(min, k/ADV_usd))",
    )
    p.add_argument(
        "--spread-cap-bps", type=float, default=25.0, help="Cap for spread cost in bps"
    )
    p.add_argument(
        "--spread-min-bps", type=float, default=2.0, help="Floor for spread cost in bps"
    )
    p.add_argument(
        "--out", type=str, default="", help="Output parquet for daily results"
    )
    p.add_argument(
        "--rebalance",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Rebalance frequency",
    )
    p.add_argument(
        "--rebal-weekday",
        choices=["MON", "TUE", "WED", "THU", "FRI"],
        default="MON",
        help="Weekly rebal weekday",
    )
    p.add_argument(
        "--turnover-cap",
        type=float,
        default=None,
        help="Cap on daily turnover (e.g., 0.5 for 50%)",
    )
    p.add_argument(
        "--report-csv",
        type=str,
        default="",
        help="Optional CSV export of daily results",
    )
    p.add_argument(
        "--report-html",
        type=str,
        default="",
        help="Optional HTML report (summary + last 20 days)",
    )
    # Optional extras for report
    p.add_argument(
        "--sector-map-csv",
        type=str,
        default="",
        help="CSV with columns symbol,sector for exposure table",
    )
    # Risk sizing options
    p.add_argument(
        "--sector-cap",
        type=float,
        default=None,
        help="Max weight per sector (e.g., 0.30)",
    )
    p.add_argument(
        "--vol-target",
        type=float,
        default=None,
        help="Target daily portfolio volatility (approx; uses 20D vol per name)",
    )
    p.add_argument(
        "--vol-lookback",
        type=int,
        default=20,
        help="Lookback days for per-name vol estimate",
    )
    p.add_argument(
        "--kelly-cap",
        type=float,
        default=None,
        help="Cap for Kelly-style fraction scaling based on meta_prob (0..1)",
    )
    p.add_argument(
        "--mlflow", action="store_true", help="Log run to MLflow if available"
    )
    p.add_argument(
        "--mlflow-experiment",
        type=str,
        default="research-backtest",
        help="MLflow experiment name",
    )
    # Decision logging (per-day symbol diagnostics)
    p.add_argument(
        "--decision-log-csv",
        type=str,
        default="",
        help="Append per-day symbol diagnostics (prob, weights, realized)",
    )
    p.add_argument(
        "--log-all-candidates",
        action="store_true",
        help="Log all candidates each day (not just holdings)",
    )
    # Optional meta-level calibrator to transform prob column before ranking
    p.add_argument(
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Optional meta calibrator (Platt/isotonic) to apply to prob column",
    )
    # Optional specialist probability logging
    p.add_argument(
        "--log-specialist-probs",
        action="store_true",
        help="Recompute and attach per-specialist probabilities for the day to the log",
    )
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Per-specialist calibrators (from run_cv) for specialist prob logging",
    )
    p.add_argument(
        "--trade-learning-csv",
        type=str,
        default="",
        help="Optional CSV with per-trade scenario summary for downstream learning",
    )
    # Report extras
    p.add_argument(
        "--account-equity",
        type=float,
        default=100000.0,
        help="Account equity for computing shares/notional in report holdings table",
    )
    return p.parse_args(argv)


def _is_rebal_day(
    date: pd.Timestamp, mode: str, weekday: str, prev_date: Optional[pd.Timestamp]
) -> bool:
    if mode == "daily":
        return True
    wd_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
    if mode == "weekly":
        return date.weekday() == wd_map[weekday]
    # monthly: rebalance on the first trading day of each new month
    if prev_date is None:
        return True
    return date.month != prev_date.month or date.year != prev_date.year


def daily_backtest(
    features_path: str,
    pred_path: str,
    prob_col: str,
    top_k: int,
    cost_bps: float,
    cost_model: str = "flat",
    spread_k: float = 1e8,
    spread_cap_bps: float = 25.0,
    spread_min_bps: float = 2.0,
    rebalance: str = "daily",
    rebal_weekday: str = "MON",
    turnover_cap: Optional[float] = None,
    decision_log_csv: Optional[str] = None,
    log_all_candidates: bool = False,
    sector_cap: Optional[float] = None,
    vol_target: Optional[float] = None,
    vol_lookback: int = 20,
    sector_map_csv: Optional[str] = None,
    kelly_cap: Optional[float] = None,
) -> pd.DataFrame:
    # Load only the columns we need to reduce memory
    spec_cols = [
        "spec_pattern",
        "spec_technical",
        "spec_sequence",
        "spec_breakout",
        "spec_flow",
        "spec_adx",
        "spec_stoch_rsi",
        "spec_willr",
        "spec_cci",
        "spec_lstm",
        "spec_gaprev",
        "spec_nlp",
        "spec_nlp_earnings",
        "spec_sector",
    ]
    extra_cols = [
        "mom_sma_5_20",
        "ret_5d",
        "ret_20d",
        "price_z_20",
        "atr_pct_14",
        "vol_z_20",
        "meanrev_20",
        "regime_vol",
        "regime_risk",
    ]
    feat_cols = ["date", "symbol", "fret_1d", "adv20", *extra_cols, *spec_cols]
    # Remove duplicates while preserving order
    feat_cols = list(dict.fromkeys(feat_cols))
    try:
        f = pd.read_parquet(features_path, columns=feat_cols)
    except Exception:
        f = pd.read_parquet(features_path)
        missing_cols = [c for c in feat_cols if c not in f.columns]
        for col in missing_cols:
            f[col] = np.nan
        f = f[[c for c in feat_cols if c in f.columns]]
    try:
        p = pd.read_parquet(pred_path, columns=["date", "symbol", prob_col])
    except Exception:
        p = pd.read_parquet(pred_path)[["date", "symbol", prob_col]]
    f["date"] = pd.to_datetime(f["date"])
    p["date"] = pd.to_datetime(p["date"])
    # Symbols as categorical to shrink memory during merges/groupbys
    f["symbol"] = f["symbol"].astype("category")
    p["symbol"] = p["symbol"].astype("category")
    # Optional: apply meta-level calibrator to prob column

    if "meta_calibrator_pkl" in locals():
        pass
    dfp = p.copy()
    # This function will be called only by main with explicit args; we leave this transformation to main, but keep pipeline here
    df = dfp.merge(f[["date", "symbol", "fret_1d"]], on=["date", "symbol"], how="left")
    df = df.dropna(subset=[prob_col, "fret_1d"]).copy()

    # Sort by date and probability per day
    df = df.sort_values(["date", prob_col], ascending=[True, False]).reset_index(
        drop=True
    )

    results = []
    prev_weights = pd.Series(dtype=float)
    cur_weights = pd.Series(dtype=float)
    prev_date: Optional[pd.Timestamp] = None
    trade_seq: dict[str, int] = {}
    scenario_classifier = ScenarioClassifier()
    scenario_tracker = ScenarioPerformanceTracker(scenario_classifier)
    scenario_cols_all = ["symbol"]
    scenario_cols_all.extend([c for c in SCENARIO_FEATURE_COLUMNS if c in f.columns])
    scenario_cols_all.extend(
        [c for c in f.columns if isinstance(c, str) and c.startswith("spec_")]
    )
    # De-duplicate while preserving order
    seen_cols = set()
    scenario_cols_all = [
        c for c in scenario_cols_all if not (c in seen_cols or seen_cols.add(c))
    ]

    def _append_decision_log(
        date_ts: pd.Timestamp,
        grp_unique_df: pd.DataFrame,
        cur_w: pd.Series,
        prev_w: pd.Series,
        rebalanced: bool,
    ) -> None:
        if not decision_log_csv:
            return
        import os as _os

        # base set: either all candidates or current holdings
        if log_all_candidates:
            base = grp_unique_df[["symbol", prob_col]].drop_duplicates("symbol").copy()
        else:
            if len(cur_w) == 0:
                return
            base = pd.DataFrame({"symbol": cur_w.index}).merge(
                grp_unique_df[["symbol", prob_col]].drop_duplicates("symbol"),
                on="symbol",
                how="left",
            )
        # Attach ADV/ATR if present in features for the same date
        cols_extra = []
        if "adv20" in f.columns:
            cols_extra.append("adv20")
        if "atr_pct_14" in f.columns:
            cols_extra.append("atr_pct_14")
        if cols_extra:
            day_ex = f.loc[
                f["date"] == date_ts, ["symbol"] + cols_extra
            ].drop_duplicates("symbol")
            base = base.merge(day_ex, on="symbol", how="left")
        # Previous/target weights aligned to base
        w_prev = prev_w.reindex(base["symbol"]).fillna(0.0)
        w_tgt = cur_w.reindex(base["symbol"]).fillna(0.0)
        dw = (w_tgt - w_prev).abs()
        # Realized next-day return map
        r_map = (
            grp_unique_df.set_index("symbol")["fret_1d"]
            .reindex(base["symbol"])
            .fillna(0.0)
        )
        gross_contrib = w_tgt * r_map
        cost_contrib = (cost_bps / 1e4) * dw
        net_contrib = gross_contrib - cost_contrib
        # Compute trade lifecycle metadata on rebalance days
        act_map: dict[str, str] = {}
        tid_map: dict[str, int] = {}
        if rebalanced:
            for sym in base["symbol"].astype(str).tolist():
                w_old = float(w_prev.get(sym, 0.0))
                w_new = float(w_tgt.get(sym, 0.0))
                if w_old <= 0 and w_new > 0:
                    trade_seq[sym] = int(trade_seq.get(sym, 0)) + 1
                    act_map[sym] = "open"
                    tid_map[sym] = trade_seq[sym]
                elif w_old > 0 and w_new <= 0:
                    act_map[sym] = "close"
                    tid_map[sym] = int(trade_seq.get(sym, 0))
                elif w_new > w_old:
                    act_map[sym] = "increase"
                    tid_map[sym] = int(trade_seq.get(sym, 0))
                elif w_new < w_old:
                    act_map[sym] = "decrease"
                    tid_map[sym] = int(trade_seq.get(sym, 0))
                else:
                    act_map[sym] = "hold"
                    tid_map[sym] = int(trade_seq.get(sym, 0))
        out = base.copy()
        out.insert(0, "date_decision", pd.Timestamp(date_ts).date())
        out.insert(1, "selected", out["symbol"].isin(cur_w.index))
        out["w_prev"], out["w_tgt"], out["dw"] = w_prev.values, w_tgt.values, dw.values
        out["fret_1d_next"] = r_map.values
        out["gross_contrib"], out["cost_contrib"], out["net_contrib"] = (
            gross_contrib.values,
            cost_contrib.values,
            net_contrib.values,
        )
        # Standard gating columns placeholder
        for nm in ("liq_ok", "atr_ok", "earn_ok"):
            if nm not in out.columns:
                out[nm] = np.nan
        out["_risk_ok"] = True
        # Trade lifecycle
        out["trade_id"] = out["symbol"].map(tid_map).fillna(0).astype(int)
        out["action"] = out["symbol"].map(act_map).fillna("")
        # Optional: Attach per-specialist probabilities for the date
        if hasattr(parse_args, "__call__"):  # silence static analyzers
            pass
        try:
            from ..features.specialists import compute_specialist_scores as _compute_specs  # type: ignore
            from ..models.calib_utils import (
                apply_calibrator as _apply_cal,
                naive_prob_map as _naive,
            )
        except Exception:
            _compute_specs = None  # type: ignore
        # Disable specialist prob logging inside backtest decision log to avoid scope issues and keep runtime light.
        # Advanced users can compute specialist probabilities offline if needed.
        if False:
            try:
                day_full = f[
                    (f["date"] == date_ts) & (f["symbol"].isin(out["symbol"]))
                ].copy()
                det = _compute_specs(day_full, news_sentiment=None, params=None)
                # Build calibrators if provided
                cals = {}
                if getattr(args, "calibrators_pkl", ""):
                    import pickle as _pkl

                    try:
                        with open(args.calibrators_pkl, "rb") as _f:
                            payload = _pkl.load(_f)
                        cals = payload.get("models", {})
                    except Exception:
                        cals = {}
                for sc in [
                    c
                    for c in det.columns
                    if c.startswith("spec_") and not c.endswith("_prob")
                ]:
                    rawv = det[sc].astype(float).values
                    if sc in cals:
                        pv = _apply_cal(cals[sc], rawv)
                    else:
                        pv = _naive(rawv)
                    det[f"{sc}_prob"] = pv
                keep = ["symbol"] + [c for c in det.columns if c.endswith("_prob")]
                out = out.merge(
                    det[keep].drop_duplicates("symbol"), on="symbol", how="left"
                )
            except Exception as e:
                print(f"[bt] warning: failed to attach specialist probs: {e}")
        try:
            feat_cols_today = [c for c in scenario_cols_all if c in f.columns]
            feat_slice = f.loc[f["date"] == date_ts, feat_cols_today].copy()
            if "symbol" in feat_slice.columns:
                feat_slice["symbol"] = feat_slice["symbol"].astype(str)
            out["symbol"] = out["symbol"].astype(str)
            out = scenario_tracker.process_day(date_ts, out, feat_slice)
        except Exception as e:
            print(f"[bt] warning: failed to enrich scenario log: {e}")
        try:
            _os.makedirs(_os.path.dirname(decision_log_csv), exist_ok=True)
            if _os.path.exists(decision_log_csv):
                out.to_csv(decision_log_csv, mode="a", header=False, index=False)
            else:
                out.to_csv(decision_log_csv, index=False)
            print(f"[bt] appended decision log -> {decision_log_csv} rows={len(out)}")
        except Exception as e:
            print(f"[bt] decision log failed: {e}")

    # Precompute simple per-name vol estimate (rolling std of fret_1d)
    try:
        f_full = pd.read_parquet(features_path, columns=["date", "symbol", "fret_1d"])  # type: ignore
    except Exception:
        f_full = pd.read_parquet(features_path)[["date", "symbol", "fret_1d"]]
    f_full["date"] = pd.to_datetime(f_full["date"]).dt.normalize()
    f_full.sort_values(["symbol", "date"], inplace=True)
    lb = int(vol_lookback or 20)
    sigmas = (
        f_full.groupby("symbol")["fret_1d"]
        .rolling(lb, min_periods=max(5, lb // 2))
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    f_full = f_full.assign(_sigma=sigmas.fillna(sigmas.median(skipna=True)))
    sigma_map_by_date = {
        d: g.set_index("symbol")["_sigma"] for d, g in f_full.groupby("date")
    }

    # Load sector map if provided
    sector_map = None
    if hasattr(parse_args, "__call__"):
        pass
    try:
        if sector_map_csv and os.path.exists(sector_map_csv):
            sm = pd.read_csv(args.sector_map_csv)
            sm = sm[["symbol", "sector"]]
            sm["symbol"] = sm["symbol"].astype(str).str.upper()
            sector_map = sm.set_index("symbol")["sector"].to_dict()
    except Exception:
        sector_map = None

    def _project_sector_cap(w: pd.Series, cap: float) -> pd.Series:
        if sector_map is None:
            return w
        s = w.copy()
        # iteratively clip sectors above cap and renormalize remaining
        for _ in range(5):
            dfw = s.reset_index()
            dfw.columns = ["symbol", "w"]
            dfw["sector"] = dfw["symbol"].map(sector_map).fillna("UNKNOWN")
            sec_sum = dfw.groupby("sector")["w"].sum()
            over = sec_sum[sec_sum > cap]
            if over.empty:
                break
            for sec, tot in over.items():
                mask = dfw["sector"] == sec
                if tot > 0:
                    dfw.loc[mask, "w"] *= cap / float(tot)
            # renormalize to sum=1
            s = dfw.set_index("symbol")["w"]
            s = s / max(1e-12, float(s.sum()))
        return s

    for date, grp in df.groupby("date", sort=True):
        ts = pd.Timestamp(date)
        # Ensure unique symbols within the date group (keep highest probability row)
        grp_sorted = grp.sort_values(prob_col, ascending=False)
        grp_unique = grp_sorted.drop_duplicates(subset="symbol", keep="first")
        rebal_today = _is_rebal_day(ts, rebalance, rebal_weekday, prev_date)
        if rebal_today:
            picks = grp_unique.head(top_k).copy()
            if len(picks) == 0:
                prev_date = ts
                continue
            base_syms = picks["symbol"].values
            # Start with equal weights
            w = pd.Series(1.0 / len(picks), index=base_syms)
            # Kelly-style scaling based on probability edge
            try:
                if kelly_cap is not None:
                    prob = (
                        picks.set_index("symbol")[prob_col]
                        .reindex(w.index)
                        .astype(float)
                        .fillna(0.5)
                    )
                    edge = (2.0 * prob - 1.0).clip(lower=0.0)
                    k = edge
                    if float(kelly_cap) > 0:
                        k = k.clip(0.0, float(kelly_cap))
                    if k.sum() > 0:
                        w = (k / k.sum()).reindex(w.index).fillna(0.0)
            except Exception:
                pass
            # Volatility targeting via risk-parity tilt
            try:
                if vol_target is not None or True:
                    sig = sigma_map_by_date.get(
                        pd.to_datetime(date).normalize(), pd.Series(0.02, index=w.index)
                    )
                    sig = sig.reindex(w.index).fillna(
                        sig.median() if hasattr(sig, "median") else 0.02
                    )
                    inv = (1.0 / sig.replace(0.0, np.nan)).fillna(0.0)
                    if inv.sum() > 0:
                        w = (inv / inv.sum()).reindex(w.index).fillna(0.0)
            except Exception:
                pass
            # Sector cap projection if requested
            try:
                if sector_cap is not None:
                    cap = float(sector_cap)
                    if cap > 0:
                        w = _project_sector_cap(w, cap)
            except Exception:
                pass
            # Compute turnover cost on rebal days only
            # Compute turnover on current pick set (sells not fully captured in this simplified implementation)
            aligned_prev = prev_weights.reindex(w.index).fillna(0.0)
            aligned_curr = w
            delta = aligned_curr - aligned_prev
            turnover = float(delta.abs().sum())
            if turnover_cap is not None and turnover > turnover_cap:
                # Scale changes to respect the cap: w_new = prev + alpha * delta
                alpha = max(0.0, min(1.0, turnover_cap / max(1e-12, turnover)))
                aligned_curr = aligned_prev + alpha * delta
                # Renormalize to sum to 1 if needed
                s = float(aligned_curr.sum())
                if s > 0:
                    aligned_curr = aligned_curr / s
                turnover = float((aligned_curr - aligned_prev).abs().sum())
                w = aligned_curr
            prev_weights = w.copy()
            cur_weights = w
        else:
            # No rebalance: keep weights, zero turnover
            turnover = 0.0
        # Daily portfolio return always computed from current weights
        # Use returns from all symbols we currently hold; if missing, treat as 0
        # Map returns for current day by symbol (unique index)
        fret_map = grp_unique.set_index("symbol")["fret_1d"]
        aligned_w = cur_weights.reindex(fret_map.index).fillna(0.0)
        port_ret = float((aligned_w * fret_map).sum())
        # Base turnover cost
        cost = (cost_bps / 1e4) * turnover
        # Optional spread proxy on rebalance days using ADV-based heuristic
        if rebal_today and str(cost_model).lower() == "spread" and turnover > 0:
            try:
                day_ex = (
                    f.loc[f["date"] == ts, ["symbol", "adv20"]]
                    .drop_duplicates("symbol")
                    .set_index("symbol")
                )
                dw = (
                    (aligned_curr - aligned_prev)
                    .abs()
                    .reindex(aligned_curr.index)
                    .fillna(0.0)
                )
                adv = day_ex.reindex(dw.index)["adv20"].astype(float).fillna(np.nan)
                # Heuristic: spread_bps = clip(k / ADV_usd, min, cap)
                with np.errstate(divide="ignore", invalid="ignore"):
                    spread_bps = (float(spread_k) / adv).replace(
                        [np.inf, -np.inf], np.nan
                    )
                spread_bps = spread_bps.fillna(float(spread_cap_bps)).clip(
                    lower=float(spread_min_bps), upper=float(spread_cap_bps)
                )
                spread_cost = float(((dw * (spread_bps / 1e4)).fillna(0.0)).sum())
                cost += spread_cost
            except Exception:
                pass
        net_ret = port_ret - cost
        results.append(
            {
                "date": date,
                "gross_ret": port_ret,
                "turnover": turnover,
                "cost": cost,
                "net_ret": net_ret,
                "names": int((cur_weights > 0).sum()),
            }
        )
        # Decision log for this day
        # prev_ref is previous weights aligned on the set used for turnover calc.
        prev_ref = prev_weights if rebal_today else cur_weights
        _append_decision_log(ts, grp_unique, cur_weights, prev_ref, rebal_today)
        prev_date = ts

    if not results:
        # Return an empty result with expected columns to avoid errors downstream
        res = pd.DataFrame(
            columns=[
                "date",
                "gross_ret",
                "turnover",
                "cost",
                "net_ret",
                "names",
                "equity",
            ]
        )
        daily_backtest.last_scenario_tracker = scenario_tracker
        return res
    res = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    res["equity"] = (1.0 + res["net_ret"]).cumprod()
    daily_backtest.last_scenario_tracker = scenario_tracker
    return res


def summary_stats(res: pd.DataFrame) -> dict:
    if res.empty:
        return {}
    daily = res["net_ret"].values
    ann = 252
    cagr = res["equity"].iloc[-1] ** (ann / max(1, len(res))) - 1.0
    vol = np.std(daily) * np.sqrt(ann)
    sharpe = (np.mean(daily) * ann) / vol if vol > 0 else float("nan")
    mdd = float((res["equity"].cummax() - res["equity"]).max()) / float(
        res["equity"].cummax().max()
    )
    turn = res["turnover"].mean()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd, "AvgTurnover": turn}


def _build_bar_svg(
    items: List[tuple[str, float]],
    width: int = 800,
    height: int = 240,
    margin: int = 40,
    title: str = "",
) -> str:
    if not items:
        return "<svg width='800' height='240'></svg>"
    labels = [a for a, _ in items]
    vals = [float(b) for _, b in items]
    vmax = max(1e-12, max(vals))
    bw = (width - 2 * margin) / max(1, len(vals))
    rects = []
    for i, v in enumerate(vals):
        x = margin + i * bw
        h = (height - 2 * margin) * (v / vmax)
        y = height - margin - h
        rects.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw*0.8:.1f}' height='{h:.1f}' fill='#2ca02c' />"
        )
        rects.append(
            f"<text x='{x + bw*0.4:.1f}' y='{height - margin/3:.1f}' font-size='10' text-anchor='middle'>{labels[i]}</text>"
        )
    title_el = (
        f"<text x='{width/2:.1f}' y='{margin/2:.1f}' text-anchor='middle' font-size='14'>{title}</text>"
        if title
        else ""
    )
    svg = f"""
<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>
  <rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' stroke='#ddd'/>
  {title_el}
  {''.join(rects)}
</svg>
""".strip()
    return svg


def write_reports(
    res: pd.DataFrame,
    stats: dict,
    csv_path: Optional[str],
    html_path: Optional[str],
    features_path: Optional[str] = None,
    pred_path: Optional[str] = None,
    prob_col: str = "meta_prob",
    top_k: int = 20,
    account_equity: float = 100000.0,
    decision_log_csv: Optional[str] = None,
    cost_bps: float = 5.0,
    sector_map_csv: Optional[str] = None,
) -> None:
    if csv_path:
        try:
            res.to_csv(csv_path, index=False)
            print(f"[bt] CSV report -> {csv_path}")
        except Exception as e:
            print(f"[bt] CSV export failed: {e}")
    if html_path:
        try:
            tail = res.tail(20).copy()
            # Build a simple SVG equity curve
            svg = _build_equity_svg(res[["date", "equity"]])
            # Optional current holdings snapshot (top-K on last date)
            holdings_html = ""
            bars_svg = ""
            if features_path and pred_path and os.path.exists(pred_path):
                try:
                    p = pd.read_parquet(pred_path, columns=["date", "symbol", prob_col])
                except Exception:
                    p = pd.read_parquet(pred_path)[["date", "symbol", prob_col]]
                p["date"] = pd.to_datetime(p["date"]).dt.normalize()
                if not p.empty:
                    last_date = p["date"].max()
                    last = p[p["date"] == last_date].copy()
                    last = (
                        last.sort_values(prob_col, ascending=False)
                        .drop_duplicates("symbol")
                        .head(int(top_k))
                    )
                    # Attach price for that date
                    try:
                        f = pd.read_parquet(features_path, columns=["date", "symbol", "adj_close"])  # type: ignore
                    except Exception:
                        f = pd.read_parquet(features_path)[
                            ["date", "symbol", "adj_close"]
                        ]
                    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
                    f = f[f["date"] == last_date]
                    det = last.merge(f, on=["date", "symbol"], how="left")
                    det["weight"] = 1.0 / max(1, len(det))
                    det["price"] = det["adj_close"].astype(float)
                    det["shares"] = (
                        (account_equity * det["weight"] / det["price"])
                        .fillna(0.0)
                        .astype(float)
                        .apply(lambda x: int(max(0, int(x))))
                    )
                    det["notional"] = det["shares"] * det["price"]
                    det["status"] = "open"
                    holdings_cols = [
                        "symbol",
                        prob_col,
                        "price",
                        "weight",
                        "shares",
                        "notional",
                        "status",
                    ]
                    holdings_html = det[holdings_cols].to_html(index=False)
                    bars_svg = _build_bar_svg(
                        [
                            (str(r["symbol"]), float(r[prob_col]))
                            for _, r in det.iterrows()
                        ],
                        title="Top‑K probabilities (last date)",
                    )
            # Build recent trades table from decision log if available
            trades_html = ""
            if decision_log_csv and os.path.exists(decision_log_csv) and features_path:
                try:
                    dlog = pd.read_csv(decision_log_csv)
                    if (
                        not dlog.empty
                        and "w_prev" in dlog.columns
                        and "w_tgt" in dlog.columns
                    ):
                        dlog["date_decision"] = pd.to_datetime(
                            dlog["date_decision"]
                        ).dt.normalize()
                        eps = 1e-9
                        entries = dlog[
                            (dlog["w_prev"] <= eps) & (dlog["w_tgt"] > eps)
                        ].copy()
                        exits = dlog[
                            (dlog["w_prev"] > eps) & (dlog["w_tgt"] <= eps)
                        ].copy()
                        # Price map
                        try:
                            fprices = pd.read_parquet(features_path, columns=["date", "symbol", "adj_close"])  # type: ignore
                        except Exception:
                            fprices = pd.read_parquet(features_path)[
                                ["date", "symbol", "adj_close"]
                            ]
                        fprices["date"] = pd.to_datetime(fprices["date"]).dt.normalize()
                        fprices["symbol"] = fprices["symbol"].astype(str).str.upper()
                        rows: list[dict] = []
                        for _, er in entries.sort_values("date_decision").iterrows():
                            sym = str(er["symbol"]).upper()
                            dt0 = pd.Timestamp(er["date_decision"]).normalize()
                            w = float(er.get("w_tgt", 0.0))
                            pr0s = fprices.loc[
                                (fprices["date"] == dt0) & (fprices["symbol"] == sym),
                                "adj_close",
                            ].astype(float)
                            pr0 = (
                                float(pr0s.iloc[0]) if not pr0s.empty else float("nan")
                            )
                            ex = (
                                exits[
                                    (exits["symbol"].astype(str).str.upper() == sym)
                                    & (exits["date_decision"] > dt0)
                                ]
                                .sort_values("date_decision")
                                .head(1)
                            )
                            if not ex.empty:
                                dt1 = pd.Timestamp(
                                    ex["date_decision"].iloc[0]
                                ).normalize()
                                pr1s = fprices.loc[
                                    (fprices["date"] == dt1)
                                    & (fprices["symbol"] == sym),
                                    "adj_close",
                                ].astype(float)
                                pr1 = (
                                    float(pr1s.iloc[0])
                                    if not pr1s.empty
                                    else float("nan")
                                )
                                status = "closed"
                            else:
                                dt1 = pd.NaT
                                pr1 = float("nan")
                                status = "open"
                            shares = (
                                int((account_equity * w / max(pr0, 1e-9)))
                                if pr0 == pr0
                                else 0
                            )
                            ce0 = float(er.get("cost_contrib", 0.0)) * account_equity
                            ce1 = (
                                float(ex["cost_contrib"].iloc[0]) * account_equity
                                if (not ex.empty and "cost_contrib" in ex.columns)
                                else 0.0
                            )
                            pnl = (
                                (shares * (pr1 - pr0)) - (ce0 + ce1)
                                if status == "closed" and (pr0 == pr0) and (pr1 == pr1)
                                else ""
                            )
                            rows.append(
                                {
                                    "symbol": sym,
                                    "entry_date": dt0.date(),
                                    "entry_price": pr0,
                                    "shares": shares,
                                    "exit_date": (
                                        dt1.date()
                                        if isinstance(dt1, pd.Timestamp)
                                        else ""
                                    ),
                                    "exit_price": (pr1 if pr1 == pr1 else ""),
                                    "net_profit": pnl,
                                    "status": status,
                                }
                            )
                        if rows:
                            tdf = pd.DataFrame(rows)
                            trades_html = tdf.tail(20).to_html(index=False)
                except Exception as e:
                    print(f"[bt] trades section failed: {e}")

            html = []
            html.append(
                "<html><head><meta charset='utf-8'><title>Backtest Report</title>"
            )
            html.append(
                "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #eee;padding:6px 10px} .k{font-weight:bold}</style>"
            )
            html.append("</head><body>")
            html.append("<h2>Backtest Summary</h2>")
            html.append("<table>")
            for k in ["CAGR", "Sharpe", "MaxDD", "AvgTurnover"]:
                v = stats.get(k, float("nan"))
                html.append(f"<tr><td class='k'>{k}</td><td>{v:.6f}</td></tr>")
            html.append("</table>")
            # Monthly performance from daily net_ret
            try:
                mdf = res.copy()
                mdf["date"] = pd.to_datetime(mdf["date"])  # ensure dtype
                mdf["month"] = mdf["date"].dt.to_period("M").astype(str)
                mperf = (
                    mdf.groupby("month")["net_ret"]
                    .apply(lambda x: float(np.prod(1.0 + x.values) - 1.0))
                    .reset_index()
                )
                mperf = mperf.tail(12)
                mperf["ret_pct"] = (100.0 * mperf["net_ret"]).map(
                    lambda v: f"{v:+.2f}%"
                )
                monthly_html = mperf[["month", "ret_pct"]].to_html(index=False)
            except Exception as _e:
                monthly_html = ""

            html.append("<h3>Equity Curve</h3>")
            html.append(svg)
            if monthly_html:
                html.append("<h3>Monthly Performance (last 12)</h3>")
                html.append(monthly_html)
            if holdings_html:
                html.append("<h3>Current Holdings (Top‑K on last date)</h3>")
                html.append(
                    f"<p>Assumes equal weights and entry at last close; account_equity=${account_equity:,.0f}</p>"
                )
                if bars_svg:
                    html.append(bars_svg)
                html.append(holdings_html)
            # Sector exposure from last decision weights if sector map provided
            if (
                sector_map_csv
                and decision_log_csv
                and os.path.exists(decision_log_csv)
                and os.path.exists(sector_map_csv)
            ):
                try:
                    dlog = pd.read_csv(decision_log_csv)
                    if (
                        not dlog.empty
                        and "w_tgt" in dlog.columns
                        and "symbol" in dlog.columns
                    ):
                        dlog["date_decision"] = pd.to_datetime(
                            dlog["date_decision"]
                        ).dt.normalize()
                        last_dt = dlog["date_decision"].max()
                        w_last = dlog[
                            (dlog["date_decision"] == last_dt) & (dlog["w_tgt"] > 0)
                        ].copy()
                        w_last["symbol"] = w_last["symbol"].astype(str).str.upper()
                        smap = pd.read_csv(sector_map_csv)
                        smap["symbol"] = smap["symbol"].astype(str).str.upper()
                        smap = smap[["symbol", "sector"]]
                        w_last = w_last.merge(smap, on="symbol", how="left")
                        w_last["sector"].fillna("UNKNOWN", inplace=True)
                        exp = (
                            w_last.groupby("sector")["w_tgt"]
                            .sum()
                            .reset_index()
                            .sort_values("w_tgt", ascending=False)
                        )
                        exp["weight_pct"] = (100.0 * exp["w_tgt"]).map(
                            lambda v: f"{v:.1f}%"
                        )
                        sector_html = exp[["sector", "weight_pct"]].to_html(index=False)
                        html.append("<h3>Sector Exposure (last decision)</h3>")
                        html.append(sector_html)
                except Exception as _e:
                    pass
            if trades_html:
                html.append("<h3>Recent Trades</h3>")
                html.append(trades_html)
            html.append("<h3>Last 20 days</h3>")
            html.append(tail.to_html(index=False))
            html.append("</body></html>")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("\n".join(html))
            print(f"[bt] HTML report -> {html_path}")
        except Exception as e:
            print(f"[bt] HTML export failed: {e}")


def _build_equity_svg(
    df: pd.DataFrame, width: int = 800, height: int = 300, margin: int = 20
) -> str:
    if df.empty:
        return "<svg width='800' height='300'></svg>"
    xs = list(range(len(df)))
    ys = df["equity"].astype(float).values
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if ymax - ymin < 1e-12:
        ymax = ymin + 1e-6

    def scale_x(i: int) -> float:
        return margin + (width - 2 * margin) * (i / max(1, len(xs) - 1))

    def scale_y(y: float) -> float:
        # invert y for SVG (0 at top)
        return height - margin - (height - 2 * margin) * ((y - ymin) / (ymax - ymin))

    points = " ".join(f"{scale_x(i):.1f},{scale_y(y):.1f}" for i, y in enumerate(ys))
    svg = f"""
<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>
  <rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' stroke='#ddd'/>
  <polyline fill='none' stroke='#1f77b4' stroke-width='2' points='{points}'/>
</svg>
""".strip()
    return svg


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    # Resolve features from CLI or config
    feat_path = args.features
    if (not feat_path) and args.config:
        try:
            from ..infra.yaml_config import load_yaml_config  # type: ignore

            cfg = load_yaml_config(args.config)
            feat_path = (
                cfg.get("paths", {}).get("features", "")
                if isinstance(cfg.get("paths"), dict)
                else ""
            )
        except Exception:
            feat_path = ""
    if not feat_path:
        raise RuntimeError(
            "Provide --features or set paths.features in YAML via --config"
        )
    # If meta calibrator provided, transform the probability column in the pred parquet before merge
    pred_path = args.pred
    if args.meta_calibrator_pkl:
        try:
            import pickle
            import pandas as _pd

            pred_df = _pd.read_parquet(args.pred)
            with open(args.meta_calibrator_pkl, "rb") as fpk:
                payload = pickle.load(fpk)
            mdl = payload.get("model", payload)
            v = pred_df[args.prob_col].astype(float).values
            if hasattr(mdl, "predict_proba"):
                v = mdl.predict_proba(v.reshape(-1, 1))[:, 1]
            elif hasattr(mdl, "transform"):
                v = mdl.transform(v)
            pred_df[args.prob_col] = v
            # Save to a temp in-memory parquet-like path: since we can’t write temp files, pass df directly to function by monkey-patching? Simpler: write alongside as calibrated file
            import os as _os

            base, ext = _os.path.splitext(args.pred)
            calib_path = base + ".calibrated" + ext
            pred_df.to_parquet(calib_path, index=False)
            pred_path = calib_path
            print(f"[bt] applied meta calibrator to prob col; using {calib_path}")
        except Exception as e:
            print(f"[bt] warning: failed to apply meta calibrator: {e}")
    res = daily_backtest(
        feat_path,
        pred_path,
        args.prob_col,
        args.top_k,
        args.cost_bps,
        cost_model=str(getattr(args, "cost_model", "flat")),
        spread_k=float(getattr(args, "spread_k", 1e8)),
        spread_cap_bps=float(getattr(args, "spread_cap_bps", 25.0)),
        spread_min_bps=float(getattr(args, "spread_min_bps", 2.0)),
        rebalance=args.rebalance,
        rebal_weekday=args.rebal_weekday,
        turnover_cap=args.turnover_cap,
        decision_log_csv=(args.decision_log_csv or None),
        log_all_candidates=bool(args.log_all_candidates),
        sector_cap=(args.sector_cap if args.sector_cap is not None else None),
        vol_target=(args.vol_target if args.vol_target is not None else None),
        vol_lookback=int(args.vol_lookback),
        sector_map_csv=(args.sector_map_csv or None),
        kelly_cap=(args.kelly_cap if args.kelly_cap is not None else None),
    )
    tracker = getattr(daily_backtest, "last_scenario_tracker", None)
    if args.trade_learning_csv and tracker is not None:
        try:
            trades_df = tracker.completed_trades_frame()
            if not trades_df.empty:
                dir_name = os.path.dirname(args.trade_learning_csv) or "."
                os.makedirs(dir_name, exist_ok=True)
                trades_df.to_csv(args.trade_learning_csv, index=False)
                print(
                    f"[bt] scenario trade log -> {args.trade_learning_csv} rows={len(trades_df)}"
                )
            else:
                print("[bt] scenario trade log: no completed trades to write")
        except Exception as e:
            print(f"[bt] failed to write scenario trade log: {e}")
    stats = summary_stats(res)
    print(f"Stats: {stats}")
    out_path = args.out.strip()
    if not out_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "backtests"), exist_ok=True)
        out_path = os.path.join(root, "backtests", "daily_topk_results.parquet")
    res.to_parquet(out_path, index=False)
    print(f"[bt] daily results -> {out_path}")
    # Optional reports
    csv_path = args.report_csv.strip() or None
    html_path = args.report_html.strip() or None
    write_reports(
        res,
        stats,
        csv_path,
        html_path,
        features_path=args.features,
        pred_path=pred_path,
        prob_col=args.prob_col,
        top_k=args.top_k,
        account_equity=float(getattr(args, "account_equity", 100000.0)),
        decision_log_csv=(args.decision_log_csv or None),
        cost_bps=float(args.cost_bps),
        sector_map_csv=(args.sector_map_csv or None),
    )

    # Optional MLflow logging
    if args.mlflow:
        try:
            import mlflow  # type: ignore

            mlflow.set_experiment(args.mlflow_experiment)
            with mlflow.start_run(run_name="simple_daily"):
                mlflow.log_params(
                    {
                        "rebalance": args.rebalance,
                        "rebal_weekday": args.rebal_weekday,
                        "top_k": args.top_k,
                        "cost_bps": args.cost_bps,
                        "turnover_cap": args.turnover_cap
                        if args.turnover_cap is not None
                        else "none",
                    }
                )
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))
                # log artifacts
                mlflow.log_artifact(out_path)
                if csv_path and os.path.exists(csv_path):
                    mlflow.log_artifact(csv_path)
                if html_path and os.path.exists(html_path):
                    mlflow.log_artifact(html_path)
                print("[bt] MLflow logging complete.")
        except Exception as e:
            print(f"[bt] MLflow logging skipped/failed: {e}")


if __name__ == "__main__":
    main()
