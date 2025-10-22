"""Train an LSTM specialist on rolling windows of daily features.

Outputs a PyTorch checkpoint usable by spec_lstm (compute_spec_lstm) via
params.model_path. The model predicts a continuous score that is squashed
to [-1,1] with tanh inside the specialist.

Targets:
- return: simple forward return. Uses fret_1d if present; otherwise computes pct_change(1) and aligns as forward.
- log_return: log forward return.
- updown: binary up/down target; BCE with logits.

Usage:
  python -m engine.models.train_lstm_spec \
    --features data/datasets/features_daily_1D.parquet \
    --feature-set ret1_vol --window 30 --hidden 32 --epochs 5 \
    --out data/models/spec_lstm.pt
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM specialist on daily features")
    p.add_argument("--features", required=True)
    p.add_argument(
        "--feature-set", choices=["ret1", "ret1_vol", "tech"], default="ret1"
    )
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=str, default="")
    p.add_argument("--end", type=str, default="")
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Use last fraction of time-sorted windows for validation",
    )
    p.add_argument(
        "--target", choices=["return", "log_return", "updown"], default="return"
    )
    p.add_argument("--scaler", choices=["none", "zscore"], default="zscore")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--patience", type=int, default=6, help="Early stopping patience in epochs"
    )
    p.add_argument(
        "--clip-norm", type=float, default=1.0, help="Gradient clipping L2 norm (0=off)"
    )
    p.add_argument("--out", type=str, default="data/models/spec_lstm.pt")
    return p.parse_args(argv)


def _safe_series(
    df: pd.DataFrame, name: str, default: Optional[pd.Series] = None
) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    if default is not None:
        return default.astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(max(1, period))
    avg_gain = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False, min_periods=6).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=13).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=5).mean()
    hist = macd - signal
    return macd, signal, hist


def _build_features(df: pd.DataFrame, feature_set: str) -> np.ndarray:
    close = df["adj_close"].astype(float)
    ret1 = close.pct_change(1).fillna(0.0)
    feats: List[pd.Series] = [ret1]
    if feature_set in ("ret1_vol", "tech"):
        vol10 = ret1.rolling(10, min_periods=5).std(ddof=0).fillna(0.0)
        feats.append(vol10)
    if feature_set == "tech":
        # price z-score 20
        mean20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std(ddof=0)
        price_z = ((close - mean20) / std20.replace(0.0, np.nan)).fillna(0.0)
        feats.append(price_z)
        # RSI14
        rsi = _compute_rsi(close, 14)
        rsi_n = (rsi - 50.0) / 50.0
        feats.append(rsi_n)
        # MACD gap normalized by price
        macd, signal, gap = _compute_macd(close)
        macd_n = (gap / close.replace(0.0, np.nan)).fillna(0.0).clip(-0.1, 0.1) / 0.1
        feats.append(macd_n)
        # BB width
        ma20 = mean20
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        bb_w = (
            ((upper - lower) / ma20.replace(0.0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        feats.append(bb_w)
        # ATR% if present
        atrp = _safe_series(df, "atr_pct_14", pd.Series(np.nan, index=df.index))
        feats.append(atrp.fillna(0.0))
    X = np.vstack([s.values for s in feats]).T
    return X.astype(np.float32)


def _build_target(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
    # y_raw aligned to same length as df; mask for valid targets
    close = df["adj_close"].astype(float)
    if target == "updown":
        if "label_up_1d" in df.columns:
            y = df["label_up_1d"].astype(int).values  # {0,1}
        else:
            r = df.get("fret_1d", close.pct_change(1)).astype(float)
            y = (r > 0).astype(int).values
        mask = np.isfinite(y)
        return y.astype(np.float32), mask
    if target == "log_return":
        lr = np.log(close.replace(0.0, np.nan)).diff().fillna(0.0)
        # forward align using provided fret_1d if present
        if "fret_1d" in df.columns:
            lr = np.log1p(df["fret_1d"].astype(float).clip(-0.2, 0.2))
        y = lr.values
    else:  # return
        if "fret_1d" in df.columns:
            y = df["fret_1d"].astype(float).values
        else:
            y = close.pct_change(1).astype(float).values
    # Clip to a reasonable band to stabilize MSE/Huber
    y = np.clip(y, -0.05, 0.05)
    mask = np.isfinite(y)
    return y.astype(np.float32), mask


def _windows_by_symbol(
    df: pd.DataFrame, feature_set: str, window: int, target: str
) -> Tuple[List[np.ndarray], List[float], List[pd.Timestamp]]:
    Xs: List[np.ndarray] = []
    ys: List[float] = []
    dates: List[pd.Timestamp] = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        X = _build_features(g, feature_set)
        y_raw, mask = _build_target(g, target)
        # windows ending at t-1 predicting y at t
        N, F = X.shape
        for t in range(window, N):
            y = y_raw[t]
            if not np.isfinite(y):
                continue
            seq = X[t - window : t, :]
            if np.any(~np.isfinite(seq)):
                continue
            Xs.append(seq)
            ys.append(float(y))
            dates.append(pd.Timestamp(g.loc[t, "date"]))
    return Xs, ys, dates


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("PyTorch is required: pip install torch") from e

    rng = np.random.default_rng(int(args.seed))
    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])  # ensure dtype
    if args.start:
        df = df[df["date"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["date"] <= pd.Timestamp(args.end)]

    Xs, ys, dates = _windows_by_symbol(
        df, args.feature_set, int(args.window), args.target
    )
    if not Xs:
        raise RuntimeError("No training windows created. Check feature_set/window.")
    X = np.stack(Xs).astype(np.float32)  # (N, T, F)
    y = np.asarray(ys, dtype=np.float32)
    dts = np.asarray(dates)

    # Time-based split: last val_frac by date
    order = np.argsort(dts)
    X = X[order]
    y = y[order]
    dts = dts[order]
    n = len(X)
    n_val = max(1, int(n * float(args.val_frac)))
    X_tr, y_tr = X[: n - n_val], y[: n - n_val]
    X_va, y_va = X[n - n_val :], y[n - n_val :]

    # Fit scaler on train features if requested
    scaler = None
    if args.scaler == "zscore":
        mu = X_tr.mean(axis=(0, 1))  # mean per feature over all time steps
        sd = X_tr.std(axis=(0, 1))
        sd[sd == 0] = 1.0
        scaler = (mu, sd)

        def _apply_scale(Z: np.ndarray) -> np.ndarray:
            return (Z - mu) / sd

        X_tr = _apply_scale(X_tr)
        X_va = _apply_scale(X_va)

    input_dim = X.shape[-1]
    model = nn.LSTM(
        input_dim,
        int(args.hidden),
        num_layers=int(args.num_layers),
        batch_first=True,
        dropout=float(args.dropout) if int(args.num_layers) > 1 else 0.0,
    )
    head = nn.Sequential(
        nn.Dropout(p=float(args.dropout)),
        nn.Linear(int(args.hidden), 1),
    )
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    if args.target == "updown":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.SmoothL1Loss()
    # Some torch versions don't support 'verbose' kwarg; keep it minimal for compatibility
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2
    )

    def _run_epoch(xn: np.ndarray, yn: np.ndarray, train: bool) -> float:
        t_x = torch.from_numpy(xn)
        t_y = torch.from_numpy(yn).view(-1, 1)
        if train:
            model.train()
        else:
            model.eval()
        total = 0.0
        bs = int(args.batch_size)
        for i in range(0, len(xn), bs):
            xb = t_x[i : i + bs]
            yb = t_y[i : i + bs]
            if train:
                opt.zero_grad()
            out, _ = model(xb)
            pred = head(out[:, -1, :])
            # For updown target, y is {0,1}
            loss = loss_fn(pred, yb)
            if train:
                loss.backward()
                if float(args.clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(head.parameters()),
                        max_norm=float(args.clip_norm),
                    )
                opt.step()
            total += float(loss.detach().cpu().item()) * len(xb)
        return total / max(1, len(xn))

    best_va = float("inf")
    bad = 0
    for ep in range(int(args.epochs)):
        tr = _run_epoch(X_tr, y_tr, True)
        va = _run_epoch(X_va, y_va, False)
        print(
            f"[lstm] epoch {ep+1}/{args.epochs} train_loss={tr:.6f} val_loss={va:.6f}"
        )
        scheduler.step(va)
        if va + 1e-6 < best_va:
            best_va = va
            bad = 0
            best_state = {
                "net": model.state_dict(),
                "head": head.state_dict(),
                "opt": opt.state_dict(),
            }
        else:
            bad += 1
            if bad >= int(args.patience):
                print("[lstm] early stopping")
                break

    # Save checkpoint compatible with spec_lstm loader
    # Use best weights if captured
    try:
        model.load_state_dict(best_state["net"])  # type: ignore
        head.load_state_dict(best_state["head"])  # type: ignore
    except Exception:
        pass

    ckpt = {
        "meta": {
            "input_dim": int(input_dim),
            "hidden": int(args.hidden),
            "num_layers": int(args.num_layers),
            "window": int(args.window),
            "feature_set": args.feature_set,
            "target": args.target,
            "scaler": {
                "kind": args.scaler,
                "mean": (scaler[0].tolist() if scaler else None),
                "std": (scaler[1].tolist() if scaler else None),
            },
        },
        "net_state": model.state_dict(),
        "head_state": head.state_dict(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[lstm] saved -> {args.out}")


if __name__ == "__main__":
    main()
