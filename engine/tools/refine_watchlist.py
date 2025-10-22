from __future__ import annotations

"""Refine a broad watchlist into a top-N pre‑market shortlist using news sentiment.

Inputs:
- Features parquet with date/symbol and baseline OHLCV
- Broad watchlist text file (one SYMBOL per line)
- Either predictions parquet with meta_prob, or model pickle + calibrators to recompute meta_prob
- Daily news sentiment parquet (date,symbol,sentiment [-1,1])

Outputs:
- Text file with up to top‑K symbols (uppercased)
- Optional CSV with diagnostics (meta_prob, sentiment_prob, final score, specialist *_prob)
- Optional Discord message listing the final shortlist

Score:
  score = (1 - w) * meta_prob + w * sentiment_prob,
  where sentiment_prob = 0.5 + 0.5 * spec_nlp (or mapped/calibrated if available)
"""

import argparse
import os
from typing import Optional, List

import numpy as np
import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..models.calib_utils import (
    load_spec_calibrators as load_cals,
    apply_calibrator as apply_cal,
    naive_prob_map as naive_map,
)
from ..infra.yaml_config import load_yaml_config
from ..infra.reason import consensus_for_symbol
from ..infra.env import load_env_files


def _warn_nonfinite(label: str, values: pd.Series | np.ndarray) -> None:
    try:
        ser = pd.Series(values, copy=False, dtype=float)
    except Exception:
        ser = pd.to_numeric(pd.Series(values, copy=False), errors="coerce")
    arr = ser.to_numpy()
    mask = ~np.isfinite(arr)
    if mask.any():
        print(
            f"[refine] warning: {label} produced {int(mask.sum())} non-finite entries"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refine watchlist to top-K with news sentiment boost"
    )
    p.add_argument(
        "--config", type=str, default="", help="YAML with defaults (paths, specialists)"
    )
    p.add_argument("--features", required=False, default="")
    p.add_argument("--watchlist-in", required=True)
    p.add_argument(
        "--pred",
        type=str,
        default="",
        help="Optional predictions parquet with date,symbol,meta_prob",
    )
    p.add_argument(
        "--model-pkl", type=str, default="", help="Meta model pickle if --pred missing"
    )
    p.add_argument(
        "--oof", type=str, default="", help="OOF parquet to fit calibrators (optional)"
    )
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Pickle with per-specialist calibrators (optional)",
    )
    p.add_argument(
        "--news-sentiment", type=str, default="", help="Daily sentiment parquet/CSV"
    )
    p.add_argument(
        "--date",
        type=str,
        default="",
        help="Target date YYYY-MM-DD (default latest in features)",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--blend-sentiment",
        type=float,
        default=0.25,
        help="Weight of sentiment component in final score",
    )
    # Intraday mixing (optional)
    p.add_argument(
        "--intraday-features",
        type=str,
        default="",
        help="Intraday snapshot parquet to blend with daily meta",
    )
    p.add_argument(
        "--mix-intraday",
        type=float,
        default=0.0,
        help="Blend weight for intraday meta (0..1)",
    )
    p.add_argument(
        "--intraday-alpha",
        type=float,
        default=0.0,
        help="Additional alpha blend from intraday signals (0..1)",
    )
    # Sector favoring (optional)
    p.add_argument(
        "--sector-map-csv",
        type=str,
        default="",
        help="CSV with columns symbol,sector to compute sector strength",
    )
    p.add_argument(
        "--sector-boost",
        type=float,
        default=0.10,
        help="Additive boost scale based on sector strength z-score",
    )
    p.add_argument(
        "--sector-score",
        choices=["meta", "sentiment", "pre"],
        default="meta",
        help="Which metric to use for sector strength",
    )
    # Breadth indicators (sector strength from adv/dec and 52w highs ratio)
    p.add_argument(
        "--breadth-lookback-days",
        type=int,
        default=252,
        help="Lookback days for 52-week highs breadth",
    )
    p.add_argument(
        "--breadth-weight",
        type=float,
        default=0.10,
        help="Weight added to pre_score from sector breadth strength",
    )
    # Earnings blackout
    p.add_argument(
        "--earnings-file",
        type=str,
        default="",
        help="Parquet/CSV with columns date,symbol for earnings dates",
    )
    p.add_argument(
        "--earnings-blackout",
        type=int,
        default=2,
        help="Days +/- earnings date to exclude from shortlist",
    )
    p.add_argument("--sentiment-window-days", type=int, default=3)
    p.add_argument("--sentiment-half-life", type=float, default=1.5)
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--min-adv-usd", type=float, default=None)
    p.add_argument("--out", required=True, help="Output shortlist text file")
    p.add_argument("--out-csv", type=str, default="", help="Optional diagnostics CSV")
    p.add_argument(
        "--discord-webhook",
        type=str,
        default="",
        help="Optional Discord webhook to send shortlist",
    )
    # Two-tower blending (daily meta with pre-market tower) with regime-based weight
    p.add_argument(
        "--two-tower",
        action="store_true",
        help="Blend daily meta with pre-market tower using regime-weighted mix",
    )
    p.add_argument(
        "--tower-regime", choices=["regime_vol", "regime_risk"], default="regime_vol"
    )
    p.add_argument(
        "--tower-weight-low",
        type=float,
        default=0.20,
        help="Pre-tower weight when regime is low",
    )
    p.add_argument(
        "--tower-weight-high",
        type=float,
        default=0.50,
        help="Pre-tower weight when regime is high",
    )
    return p.parse_args(argv)


def _read_watchlist(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f.readlines()
            if ln.strip() and not ln.startswith("#")
        ]


def main(argv: Optional[List[str]] = None) -> None:
    load_env_files()
    args = parse_args(argv)
    cfg = load_yaml_config(args.config) if args.config else {}

    # Resolve features path: prefer CLI; else, from config paths.features
    feat_path = args.features or (
        cfg.get("paths", {}).get("features")
        if isinstance(cfg.get("paths"), dict)
        else ""
    )
    if not feat_path:
        raise RuntimeError("Provide --features or set paths.features in YAML")
    f = pd.read_parquet(feat_path)
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["symbol"] = f["symbol"].astype(str).str.upper()
    all_dates = sorted(f["date"].unique())
    if not all_dates:
        raise RuntimeError("features has no dates")
    target_date = pd.Timestamp(args.date) if args.date else all_dates[-1]
    wl_syms = set(_read_watchlist(args.watchlist_in))
    day = f[(f["date"] == target_date) & (f["symbol"].isin(wl_syms))].copy()
    if day.empty:
        # Fallback to the most recent date in features where at least one watchlist symbol exists
        cand = (
            f[f["symbol"].isin(wl_syms)]
            .groupby("date")["symbol"]
            .nunique()
            .sort_index()
        )
        cand = cand[cand > 0]
        if not cand.empty:
            fb_date = pd.Timestamp(cand.index[-1])
            if fb_date != target_date:
                print(
                    f"[refine] no overlap on {target_date.date()}; falling back to {fb_date.date()}"
                )
                target_date = fb_date
                day = f[(f["date"] == target_date) & (f["symbol"].isin(wl_syms))].copy()
        if day.empty:
            print(
                f"[refine] no overlap for watchlist on {target_date.date()}; nothing to write"
            )
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            open(args.out, "w", encoding="utf-8").close()
            return

    # Price/ADV filters (best-effort)
    prc = day.get("adj_close", day.get("close"))
    if args.min_price is not None:
        day = day[prc >= float(args.min_price)]
    if args.max_price is not None:
        day = day[prc <= float(args.max_price)]
    if args.min_adv_usd is not None and "adv20" in day.columns:
        day = day[day["adv20"] >= float(args.min_adv_usd)]
    if day.empty:
        print(f"[refine] empty after price/liquidity filters on {target_date.date()}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w", encoding="utf-8").close()
        return

    # Earnings blackout gate (optional)
    if args.earnings_file:
        try:
            earn_df = (
                pd.read_parquet(args.earnings_file)
                if args.earnings_file.lower().endswith(".parquet")
                else pd.read_csv(args.earnings_file)
            )
            earn_df["date"] = pd.to_datetime(earn_df["date"]).dt.normalize()
            earn_df["symbol"] = earn_df["symbol"].astype(str).str.upper()
            blk = earn_df.copy()
            blk["start"], blk["end"] = blk["date"] - pd.Timedelta(
                days=int(args.earnings_blackout)
            ), blk["date"] + pd.Timedelta(days=int(args.earnings_blackout))
            in_blk = (
                blk[(blk["start"] <= target_date) & (blk["end"] >= target_date)][
                    "symbol"
                ]
                .unique()
                .tolist()
            )
            if in_blk:
                before = len(day)
                day = day[~day["symbol"].isin(in_blk)]
                if day.empty:
                    print(
                        "[refine] all candidates filtered by earnings blackout; writing empty shortlist"
                    )
                    os.makedirs(os.path.dirname(args.out), exist_ok=True)
                    open(args.out, "w", encoding="utf-8").close()
                    return
                print(f"[refine] earnings blackout removed {before - len(day)} symbols")
        except Exception as e:
            print(f"[refine] earnings gate failed: {e}")

    # Load sentiment
    sent_path = args.news_sentiment or cfg.get("paths", {}).get("sentiment_out", "")
    sentiment = None
    if sent_path:
        try:
            sentiment = load_sentiment_file(sent_path)
        except Exception:
            sentiment = None

    # Compute specialist scores, including news specialist
    spec_params = cfg.get("specialists", {})
    # Ensure default news params for pre-market
    if isinstance(spec_params, dict):
        spec_params = dict(spec_params)
        spec_params.setdefault("news", {})
        if isinstance(spec_params["news"], dict):
            spec_params["news"].setdefault("enabled", True)
            spec_params["news"]["window_days"] = args.sentiment_window_days
            spec_params["news"]["half_life_days"] = args.sentiment_half_life
            spec_params["news"].setdefault("agg", "mean")
    specs = compute_specialist_scores(day, news_sentiment=sentiment, params=spec_params)

    # Build per-specialist probabilities
    calibrators = load_cals(
        calibrators_pkl=(
            args.calibrators_pkl
            or cfg.get("calibration", {}).get("calibrators_pkl", "")
        )
        or None,
        oof_path=(args.oof or cfg.get("paths", {}).get("oof", "")) or None,
        kind=cfg.get("calibration", {}).get("kind", "platt"),
    )
    prob_cols: List[str] = []
    for sc in [
        c for c in specs.columns if c.startswith("spec_") and not c.endswith("_prob")
    ]:
        raw = specs[sc].astype(float).values
        prob = (
            apply_cal(calibrators.get(sc), raw)
            if (calibrators and sc in calibrators)
            else naive_map(raw)
        )
        specs[f"{sc}_prob"] = prob
        prob_cols.append(f"{sc}_prob")

    # Compute meta_prob (predictions parquet or model)
    meta_prob = None
    if args.pred:
        try:
            p = pd.read_parquet(args.pred)
            p["date"] = pd.to_datetime(p["date"]).dt.normalize()
            pp = p[p["date"] == target_date][["symbol", "meta_prob"]]
            meta_prob = (
                specs[["symbol"]]
                .merge(pp, on="symbol", how="left")["meta_prob"]
                .astype(float)
                .values
            )
        except Exception:
            meta_prob = None
    if meta_prob is None:
        if not args.model_pkl and not cfg.get("paths", {}).get("meta_model"):
            # fallback: equal-weight across specialist probabilities
            meta_prob = specs[prob_cols].mean(axis=1).fillna(0.5).values
        else:
            import pickle

            model_path = args.model_pkl or cfg.get("paths", {}).get("meta_model")
            with open(model_path, "rb") as fpk:
                payload = pickle.load(fpk)
            clf = payload.get("model")
            feature_names = payload.get("features") or prob_cols
            # Add regime features if the model expects them
            try:
                need_reg = [
                    c
                    for c in ("regime_vol", "regime_risk")
                    if c in feature_names and c not in specs.columns
                ]
                if need_reg:
                    from ..features.regime import compute_regime_features_daily as _reg

                    reg = _reg(f)
                    reg["date"] = pd.to_datetime(reg["date"]).dt.normalize()
                    reg_day = reg[reg["date"] == target_date]
                    if not reg_day.empty:
                        for rc in need_reg:
                            specs[rc] = float(reg_day.iloc[0][rc])
            except Exception:
                # leave as missing; will be filled later
                pass
            # Add odds/logit transforms if expected
            for c in list(prob_cols):
                base = c
                if (
                    f"{base}_odds" in feature_names
                    and f"{base}_odds" not in specs.columns
                ):
                    pclip = pd.Series(specs[base]).astype(float).clip(1e-6, 1 - 1e-6)
                    specs[f"{base}_odds"] = (pclip / (1 - pclip)).astype(float)
                if (
                    f"{base}_logit" in feature_names
                    and f"{base}_logit" not in specs.columns
                ):
                    if f"{base}_odds" in specs.columns:
                        specs[f"{base}_logit"] = np.log(
                            specs[f"{base}_odds"].astype(float)
                        )
                    else:
                        pclip = (
                            pd.Series(specs[base]).astype(float).clip(1e-6, 1 - 1e-6)
                        )
                        specs[f"{base}_logit"] = np.log(
                            (pclip / (1 - pclip)).astype(float)
                        )
            # Add simple interaction terms pc__x__reg if expected
            for name in feature_names:
                if "__x__" in name and name not in specs.columns:
                    a, b = name.split("__x__", 1)
                    if a in specs.columns and b in specs.columns:
                        try:
                            specs[name] = specs[a].astype(float) * specs[b].astype(
                                float
                            )
                        except Exception:
                            pass
            for col in feature_names:
                if col not in specs.columns:
                    # Sensible defaults: prob features ~0.5, regime ~0.0, interactions ~0.0
                    if (
                        col in ("regime_vol", "regime_risk")
                        or "__x__" in col
                        or col.endswith("_odds")
                        or col.endswith("_logit")
                    ):
                        specs[col] = 0.0
                    else:
                        specs[col] = 0.5
            X = specs[feature_names].values
            meta_prob = (
                clf.predict_proba(X)[:, 1]
                if hasattr(clf, "predict_proba")
                else clf.predict(X)
            )
    specs["meta_prob"] = meta_prob

    # Optional: intraday mixing of meta probability
    mix_w = float(max(0.0, min(1.0, args.mix_intraday)))
    if args.intraday_features and mix_w > 0:
        try:
            intr = pd.read_parquet(args.intraday_features)
            if "symbol" not in intr.columns:
                raise RuntimeError("intraday features parquet must include 'symbol'")
            intr = intr.copy()
            intr["symbol"] = intr["symbol"].astype(str).str.upper()
            # If date column exists, use the latest row per symbol; else treat as latest snapshot
            if "date" in intr.columns:
                intr["date"] = pd.to_datetime(intr["date"]).dt.normalize()
                # pick latest per symbol
                intr = intr.sort_values(["symbol", "date"]).groupby("symbol").tail(1)
            # Restrict to candidates
            intr_day = intr[intr["symbol"].isin(specs["symbol"])].copy()
            if not intr_day.empty:
                # Recompute specialist probs on intraday snapshot
                specs_intra = compute_specialist_scores(
                    intr_day, news_sentiment=sentiment, params=spec_params
                )
                prob_cols_i: List[str] = []
                for sc in [
                    c
                    for c in specs_intra.columns
                    if c.startswith("spec_") and not c.endswith("_prob")
                ]:
                    rawi = specs_intra[sc].astype(float).values
                    probi = (
                        apply_cal(calibrators.get(sc), rawi)
                        if (calibrators and sc in calibrators)
                        else naive_map(rawi)
                    )
                    specs_intra[f"{sc}_prob"] = probi
                    prob_cols_i.append(f"{sc}_prob")
                # Meta on intraday
                meta_i = None
                if args.pred:
                    # No easy way to use pred for intraday; skip to model if available
                    pass
                if meta_i is None:
                    if args.model_pkl or cfg.get("paths", {}).get("meta_model"):
                        import pickle

                        model_path = args.model_pkl or cfg.get("paths", {}).get(
                            "meta_model"
                        )
                        with open(model_path, "rb") as fpk:
                            payload = pickle.load(fpk)
                        clf = payload.get("model")
                        feature_names = payload.get("features") or prob_cols_i
                        for col in feature_names:
                            if col not in specs_intra.columns:
                                specs_intra[col] = 0.5
                        X_i = specs_intra[feature_names].values
                        meta_i = (
                            clf.predict_proba(X_i)[:, 1]
                            if hasattr(clf, "predict_proba")
                            else clf.predict(X_i)
                        )
                    else:
                        meta_i = (
                            specs_intra[prob_cols_i].mean(axis=1).fillna(0.5).values
                        )
                mp_i = pd.DataFrame(
                    {"symbol": specs_intra["symbol"].values, "meta_i": meta_i}
                )
                specs = specs.merge(mp_i, on="symbol", how="left")
                specs["meta_prob"] = (1.0 - mix_w) * specs["meta_prob"].astype(
                    float
                ) + mix_w * specs["meta_i"].fillna(specs["meta_prob"]).astype(float)
                specs = specs.drop(
                    columns=[c for c in ["meta_i"] if c in specs.columns]
                )
                print(f"[refine] blended intraday meta with weight={mix_w}")
                # Optional intraday alpha boost
                ia = float(max(0.0, min(1.0, args.intraday_alpha)))
                if ia > 0:
                    try:
                        import numpy as _np

                        # Join intraday features for alpha signals
                        keep_cols = [
                            "symbol",
                            "adj_close",
                            "vol_z_20",
                            "vwap_dev_20",
                            "breakout_high_20",
                        ]
                        idf = intr_day[
                            [c for c in keep_cols if c in intr_day.columns]
                        ].copy()
                        merged = specs.merge(
                            idf, on="symbol", how="left", suffixes=("", "_i")
                        )
                        # Gap proxy: (intraday last / daily prev close) - 1
                        prevc = day.set_index("symbol")["adj_close"].to_dict()
                        merged["prev_close"] = merged["symbol"].map(prevc)
                        gap = (
                            merged.get(
                                "adj_close_i", merged.get("adj_close", _np.nan)
                            ).astype(float)
                            / merged["prev_close"].astype(float)
                        ) - 1.0
                        gap = (
                            gap.replace([_np.inf, -_np.inf], _np.nan)
                            .fillna(0.0)
                            .clip(-0.1, 0.1)
                        )

                        def _sig(x):
                            return 1.0 / (1.0 + _np.exp(-x))

                        volz = merged.get("vol_z_20", 0.0)
                        vwapd = merged.get("vwap_dev_20", 0.0)
                        brk = merged.get("breakout_high_20", 0.0)
                        # Normalize to 0..1
                        a_gap = _sig(8.0 * gap)
                        a_vol = _sig(volz.astype(float))
                        a_vwp = _sig(5.0 * vwapd.astype(float))
                        a_brk = (brk.astype(float) > 0.0).astype(float)
                        alpha_i = (
                            0.40 * a_gap + 0.25 * a_vol + 0.15 * a_vwp + 0.20 * a_brk
                        )
                        alpha_i = alpha_i.clip(0.0, 1.0)
                        specs["pre_score"] = (1.0 - ia) * specs.get(
                            "pre_score", specs["meta_prob"]
                        ).astype(float) + ia * alpha_i.values
                        print(f"[refine] applied intraday alpha (w={ia})")
                    except Exception as _e:
                        print(f"[refine] intraday alpha failed: {_e}")
        except Exception as e:
            print(f"[refine] intraday mixing failed: {e}")

    # Sentiment probability from news specialist
    if "spec_nlp_prob" in specs.columns:
        sprob = specs["spec_nlp_prob"].astype(float)
    elif "spec_nlp" in specs.columns:
        sprob = 0.5 + 0.5 * specs["spec_nlp"].astype(float)
    else:
        sprob = pd.Series(0.5, index=specs.index)
    w = float(max(0.0, min(1.0, args.blend_sentiment)))
    # Pre-market tower: start from daily meta blended with sentiment
    pre_tower = (1.0 - w) * specs["meta_prob"].astype(float) + w * sprob.astype(float)
    _warn_nonfinite("pre_tower sentiment blend", pre_tower)

    # Sector breadth strength (adv/decl and 52w highs ratio) -> boost pre_tower
    if (
        args.sector_map_csv
        and os.path.exists(args.sector_map_csv)
        and float(args.breadth_weight) != 0.0
    ):
        try:
            sm = pd.read_csv(args.sector_map_csv)
            sm["symbol"] = sm["symbol"].astype(str).str.upper()
            sm = sm[["symbol", "sector"]].dropna()
            # Load recent window of features to compute breadth on target_date
            lookback_days = int(max(5, args.breadth_lookback_days))
            start_b = target_date - pd.Timedelta(days=lookback_days + 10)
            f_b = f[(f["date"] >= start_b) & (f["date"] <= target_date)].copy()
            f_b["symbol"] = f_b["symbol"].astype(str).str.upper()
            fb = f_b.merge(sm, on="symbol", how="left")
            # Advancers/decliners ratio on target_date using ret_1d if present; fallback to meta_prob>0.5
            day_b = fb[fb["date"] == target_date].copy()
            if "ret_1d" not in day_b.columns:
                if "adj_close" in fb.columns:
                    fb = fb.sort_values(["symbol", "date"]).reset_index(drop=True)
                    fb["ret_1d"] = fb.groupby("symbol")["adj_close"].pct_change(1)
                else:
                    fb["ret_1d"] = 0.0
                day_b = fb[fb["date"] == target_date].copy()
            day_b["adv"] = (
                pd.to_numeric(day_b.get("ret_1d"), errors="coerce").fillna(0.0) > 0
            ).astype(int)
            adv_ratio = day_b.groupby("sector")["adv"].mean().rename("adv_ratio")
            # 52-week highs ratio (rolling 252-day max equals last close)
            highs_ratio = None
            if "adj_close" in fb.columns:
                g = fb.sort_values(["symbol", "date"]).groupby("symbol")
                roll_max = g["adj_close"].transform(
                    lambda s: s.rolling(252, min_periods=50).max()
                )
                fb["is_high"] = (fb["adj_close"] >= roll_max * 0.999).astype(int)
                day_h = fb[fb["date"] == target_date]
                highs_ratio = (
                    day_h.groupby("sector")["is_high"].mean().rename("highs_ratio")
                )
            # Combine breadth metrics
            br = adv_ratio.to_frame()
            if highs_ratio is not None:
                br = br.join(highs_ratio, how="outer")
            br = br.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            br["breadth"] = br.mean(axis=1)
            # Normalize breadth to [-0.5, +0.5] by z over sectors
            std = br["breadth"].std(ddof=0)
            if not np.isfinite(std) or std == 0.0:
                z = pd.Series(0.0, index=br.index)
            else:
                z = (br["breadth"] - br["breadth"].mean()) / std
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2, 2) / 4.0
            z_map = z.to_dict()
            specs = specs.merge(sm, on="symbol", how="left")
            breadth_boost = specs["sector"].map(z_map).fillna(0.0) * float(
                args.breadth_weight
            )
            # Align indices explicitly to avoid pandas alignment errors
            pre_arr = np.asarray(pre_tower, dtype=float)
            pre_tower = pd.Series(pre_arr, index=specs.index)
            pre_tower = (
                pre_tower + pd.to_numeric(breadth_boost, errors="coerce").fillna(0.0)
            ).clip(0.0, 1.0)
            _warn_nonfinite("pre_tower after breadth boost", pre_tower)
        except Exception as e:
            print(f"[refine] breadth computation failed: {e}")

    # Sector favoring: compute sector strength and boost (z within sector)
    if (
        args.sector_map_csv
        and os.path.exists(args.sector_map_csv)
        and float(args.sector_boost) != 0.0
    ):
        try:
            sm = pd.read_csv(args.sector_map_csv)
            sm["symbol"] = sm["symbol"].astype(str).str.upper()
            sm = sm[["symbol", "sector"]].dropna()
            specs = specs.merge(sm, on="symbol", how="left")
            if "sector" in specs.columns:
                base_metric = specs["meta_prob"].astype(float)
                if str(args.sector_score) == "sentiment":
                    base_metric = sprob.astype(float)
                elif str(args.sector_score) == "pre":
                    base_metric = specs["pre_score"].astype(float)
                # z-score by sector
                g = specs.groupby("sector")[base_metric.name]
                mean = g.transform("mean")
                std = g.transform("std").replace(0.0, np.nan)
                z = (base_metric - mean) / std
                z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                boost = float(args.sector_boost) * z
                pre_arr2 = np.asarray(pre_tower, dtype=float)
                pre_tower = pd.Series(pre_arr2, index=specs.index)
                pre_tower = (
                    pre_tower + pd.to_numeric(boost, errors="coerce").fillna(0.0)
                ).clip(0.0, 1.0)
                _warn_nonfinite("pre_tower after sector boost", pre_tower)
                print(
                    f"[refine] applied sector boost (mode={args.sector_score}, scale={args.sector_boost})"
                )
        except Exception as e:
            print(f"[refine] sector boost failed: {e}")

    # Two-tower blending (daily meta vs pre-market tower) using regime weight
    if args.two_tower:
        try:
            from ..features.regime import compute_regime_features_daily as _reg

            reg = _reg(f)
            reg["date"] = pd.to_datetime(reg["date"]).dt.normalize()
            reg_day = reg[reg["date"] == target_date]
            if not reg_day.empty and args.tower_regime in reg_day.columns:
                raw_val = reg_day.iloc[0][args.tower_regime]
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    val = 0.0
                if not np.isfinite(val):
                    val = 0.0  # fall back to neutral mix when regime is missing
                r01 = max(0.0, min(1.0, 0.5 * (val + 1.0)))
                w_lo = float(args.tower_weight_low)
                w_hi = float(args.tower_weight_high)
                w_mix = w_lo + (w_hi - w_lo) * r01
                if not np.isfinite(w_mix):
                    w_mix = w_lo  # default to daily weight if computation failed
                specs["final_prob"] = (1.0 - w_mix) * specs["meta_prob"].astype(
                    float
                ) + w_mix * pre_tower.astype(float)
                print(
                    f"[refine] two-tower blend: regime={args.tower_regime} value={val:.3f} w_pre={w_mix:.2f}"
                )
            else:
                specs["final_prob"] = pre_tower
        except Exception as e:
            print(f"[refine] two-tower failed: {e}")
            specs["final_prob"] = pre_tower
    else:
        specs["final_prob"] = pre_tower

    # Guard against NaN/inf from earlier adjustments (fallback to meta_prob)
    final_series = pd.Series(specs["final_prob"]).astype(float)
    fallback = pd.Series(specs["meta_prob"]).astype(float).fillna(0.5)
    mask = ~np.isfinite(final_series)
    if mask.any():
        print(
            f"[refine] warning: final_prob had {int(mask.sum())} non-finite entries; applying meta fallback"
        )
    final_series = final_series.where(~mask, fallback)
    specs["final_prob"] = final_series.clip(0.0, 1.0)

    specs["pre_score"] = specs["final_prob"].astype(float)

    # Rank and output top-K
    out = specs.sort_values("pre_score", ascending=False).head(int(args.top_k)).copy()
    syms = out["symbol"].astype(str).str.upper().drop_duplicates().tolist()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as ftxt:
        for s in syms:
            ftxt.write(s + "\n")
    print(f"[refine] wrote {len(syms)} symbols -> {args.out}")

    # Optional CSV diagnostics
    if args.out_csv:
        diag_cols = ["symbol", "meta_prob", "pre_score"] + [
            c for c in specs.columns if c.startswith("spec_") and c.endswith("_prob")
        ]
        diag = out[diag_cols].copy()
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        diag.to_csv(args.out_csv, index=False)
        print(f"[refine] wrote diagnostics -> {args.out_csv}")

    # Optional Discord summary
    if args.discord_webhook:
        try:
            import requests  # type: ignore

            lines: List[str] = []
            today = pd.Timestamp.today().date()
            lines.append(f"Pre-market shortlist {today} (top {len(syms)})")
            for _, r in out.iterrows():
                sym = str(r["symbol"]).upper()
                meta = float(r.get("meta_prob", np.nan))
                spre = float(r.get("pre_score", np.nan))
                reasons = consensus_for_symbol(specs, sym)
                lines.append(f"- {sym}: score={spre:.2f} meta={meta:.2f} — {reasons}")
            msg = "\n".join(lines)
            if len(msg) > 1800:
                msg = msg[:1790] + "…"
            requests.post(args.discord_webhook, json={"content": msg}, timeout=10)
            print("[refine] sent Discord summary")
        except Exception as e:
            print(f"[refine] discord failed: {e}")


if __name__ == "__main__":
    main()
