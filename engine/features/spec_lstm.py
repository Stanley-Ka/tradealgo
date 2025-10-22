from __future__ import annotations

"""Optional LSTM/GRU specialist scaffold.

This module is deliberately optional: if PyTorch is not installed, functions
will return zeros or raise an informative error. The intent is to provide a
drop-in score generator consistent with other specialists.
"""

from typing import Tuple, List

import numpy as np
import pandas as pd


def compute_spec_lstm(
    df: pd.DataFrame,
    model: object | None = None,
    window: int = 30,
    feature_set: str = "ret1",
    params: dict | None = None,
) -> pd.Series:
    """Compute an LSTM-based score in [-1,1] using a pre-trained model.

    - If `model` is None or torch is unavailable, returns zeros.
    - `feature_set` can be 'ret1' (1D returns) or 'ret1_vol' (return and rolling vol).
    - `window` is the lookback length per sample.
    """
    if isinstance(params, dict):
        window = int(params.get("window", window))
        feature_set = str(params.get("feature_set", feature_set))

    try:
        import torch  # type: ignore
    except Exception:
        # Torch not available; neutral score
        return pd.Series(0.0, index=df.index, dtype=float)

    # Allow passing a path to a saved model via params
    if model is None and isinstance(params, dict) and params.get("model_path"):
        try:
            model = load_lstm_from_checkpoint(params["model_path"])  # type: ignore
        except Exception:
            return pd.Series(0.0, index=df.index, dtype=float)
    if model is None:
        return pd.Series(0.0, index=df.index, dtype=float)

    # Build features consistent with trainer
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
        gap = macd - signal
        return macd, signal, gap

    close = df["adj_close"].astype(float)
    ret1 = close.pct_change(1).fillna(0.0)
    feats: List[pd.Series] = [ret1]
    if feature_set in ("ret1_vol", "tech"):
        vol10 = ret1.rolling(10, min_periods=5).std(ddof=0).fillna(0.0)
        feats.append(vol10)
    if feature_set == "tech":
        mean20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std(ddof=0)
        price_z = ((close - mean20) / std20.replace(0.0, np.nan)).fillna(0.0)
        feats.append(price_z)
        rsi_n = (_compute_rsi(close, 14) - 50.0) / 50.0
        feats.append(rsi_n)
        _, _, gap = _compute_macd(close)
        macd_n = (gap / close.replace(0.0, np.nan)).fillna(0.0).clip(-0.1, 0.1) / 0.1
        feats.append(macd_n)
        ma20 = mean20
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        bb_w = (
            ((upper - lower) / ma20.replace(0.0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        feats.append(bb_w)
        atrp = df.get("atr_pct_14", pd.Series(0.0, index=df.index))
        feats.append(atrp.fillna(0.0))
    X = np.vstack([s.values for s in feats]).T

    # Build rolling windows ending at each index
    N, F = X.shape
    if N < window:
        return pd.Series(0.0, index=df.index, dtype=float)
    # Apply scaler from checkpoint meta if available
    try:
        scaler = getattr(model, "_scaler_meta", None)  # injected below
    except Exception:
        scaler = None
    if scaler and isinstance(scaler, tuple) and scaler[0] is not None:
        mu, sd = scaler
        sd = np.where(np.asarray(sd) == 0.0, 1.0, sd)
        X = (X - np.asarray(mu)) / np.asarray(sd)

    xs = np.lib.stride_tricks.sliding_window_view(X, (window, F))
    xs = xs.reshape(-1, window, F)
    # Pad the first (window-1) positions with zeros to align to df index
    pad = np.zeros((window - 1, window, F), dtype=xs.dtype)
    xs_full = np.concatenate([pad, xs], axis=0)
    with torch.no_grad():
        x_t = torch.from_numpy(xs_full.astype(np.float32))
        y = model(x_t).detach().cpu().numpy().reshape(-1)
    # Map to [-1,1]
    target_kind = getattr(model, "_target", "return")
    if target_kind == "updown":
        # logits -> prob -> [-1,1]
        prob = 1.0 / (1.0 + np.exp(-y))
        score = 2.0 * prob - 1.0
    else:
        # regression: keep bounded
        score = np.tanh(y)
    return pd.Series(score, index=df.index, dtype=float).fillna(0.0)


class SimpleLSTMModel:  # pragma: no cover - scaffold only
    """Minimal LSTM wrapper with PyTorch, exposing a callable for inference.

    Usage:
        mdl = SimpleLSTMModel(input_dim=1, hidden=32)
        mdl.train_numpy(X_windows, y_targets, epochs=5)
        scores = mdl(torch.from_numpy(X_windows))
    """

    def __init__(self, input_dim: int, hidden: int = 32, num_layers: int = 1):
        try:
            import torch.nn as nn  # type: ignore
            import torch  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for SimpleLSTMModel") from e
        self.torch = torch
        self.nn = nn
        self.net = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def __call__(self, x):  # x: (N, T, F)
        h, _ = self.net(x)
        out = self.head(h[:, -1, :])
        return out

    def train_numpy(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 5, lr: float = 1e-3
    ):
        torch = self.torch
        nn = self.nn
        x_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
        opt = torch.optim.Adam(
            list(self.net.parameters()) + list(self.head.parameters()), lr=lr
        )
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(int(epochs)):
            opt.zero_grad()
            pred = self(x_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()


def load_lstm_from_checkpoint(path: str):  # pragma: no cover - lightweight loader
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("PyTorch required to load LSTM checkpoint") from e
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.get("meta", {})
    input_dim = int(meta.get("input_dim", 1))
    hidden = int(meta.get("hidden", 32))
    num_layers = int(meta.get("num_layers", 1))
    mdl = SimpleLSTMModel(input_dim=input_dim, hidden=hidden, num_layers=num_layers)
    mdl.net.load_state_dict(ckpt["net_state"])  # type: ignore
    mdl.head.load_state_dict(ckpt["head_state"])  # type: ignore
    mdl.net.eval()
    # Inject meta for scaler/target handling
    sc = meta.get("scaler", {})
    mdl._scaler_meta = (sc.get("mean"), sc.get("std"))  # type: ignore[attr-defined]
    mdl._target = meta.get("target", "return")  # type: ignore[attr-defined]
    return mdl
