Trading Engine – CPU/GPU Roadmap (7950X3D + 7900 XTX ROCm)
Global Conventions

Python ≥3.10, float32 for numeric features/labels, int8 for enums.

Columnar storage: Parquet (pyarrow), partitioned by symbol= and year=.

Per-symbol, per-chunk processing to keep hot arrays ≤ 16–64 MB (friendly to 3D V-Cache).

Avoid cross-sectional pandas ops in hot loops; do per-symbol compute, join later if needed.

Prefer polars for I/O + simple transforms; NumPy/Numba for kernels.

Parallelism: OS process pool at symbol-chunk level; NumPy threads inside workers.

GPU: PyTorch (ROCm), AMP, torch.compile, pinned-memory DataLoaders.

0) Environment

Task: Add a reproducible environment and perf-sane thread defaults.

Files

env/requirements.txt

env/mkl_threads.sh (or env/openblas_threads.sh)

requirements.txt

numpy
pandas
polars>=1.6
pyarrow
numba
joblib
ray[default]
scikit-learn
lightgbm
torch # ROCm build installed separately if needed
torchmetrics
onnx
onnxruntime
tqdm
pyyaml
matplotlib


Thread defaults (choose one)
env/openblas_threads.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1


DoD

pip install -r env/requirements.txt succeeds.

Running any script prints the current thread env vars.

1) Data Schema & Labeling

Goal: Standardize columns and make H={3,7,14} day labels.

Files

engine/data/schema.py

engine/data/labeling.py

engine/data/io.py

# engine/data/schema.py
from dataclasses import dataclass

FEATURE_DTYPES = {
    "date": "datetime64[ns]",
    "symbol": "category",
    "open": "float32", "high": "float32", "low": "float32", "close": "float32", "volume": "float32",
    # specialist scores appended as float32: "spec_<name>_score": "float32"
}

HORIZONS = (3, 7, 14)
LABEL_FLOATS = {f"label_ret_h{h}": "float32" for h in HORIZONS}
LABEL_BINS = {f"label_bin_h{h}": "int8" for h in HORIZONS}

# engine/data/labeling.py
import numpy as np
import pandas as pd
from .schema import HORIZONS

def make_horizon_labels(df: pd.DataFrame, thresholds=(0.0, 0.0, 0.0)) -> pd.DataFrame:
    """
    df must contain 'close'. Assumes market-day spacing per row (no missing business days per symbol).
    Returns df with label_ret_hX (float32) and label_bin_hX (int8).
    """
    close = df["close"].to_numpy(dtype=np.float32)
    for h, tau in zip(HORIZONS, thresholds):
        fut = np.empty_like(close); fut[:] = np.nan
        fut[:-h] = (close[h:] - close[:-h]) / close[:-h]
        df[f"label_ret_h{h}"] = fut.astype(np.float32)
        df[f"label_bin_h{h}"] = (fut > tau).astype("int8")
    return df

# engine/data/io.py
import polars as pl
from pathlib import Path

def write_parquet(df_pl: pl.DataFrame, base: str, symbol: str, year: int) -> str:
    outdir = Path(base) / f"symbol={symbol}" / f"year={year}"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "part.parquet"
    df_pl.write_parquet(path)
    return str(path)

def scan_parquet(base: str) -> pl.LazyFrame:
    return pl.scan_parquet(f"{base}/symbol=*/year=*/part.parquet")


DoD

Unit test: labels match expected forward returns on a tiny OHLCV toy set.

Parquet writes with symbol= and year= partitions.

2) Vectorized Feature Builder (Numba + Process Pool)

Goal: Per-symbol chunking; vector kernels; process pool.

Files

engine/features/vector_math.py

engine/features/build_features.py

# engine/features/vector_math.py
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    n = arr.size
    out = np.empty(n, dtype=np.float32); s = 0.0
    for i in range(n):
        s += arr[i]
        if i >= w: s -= arr[i - w]
        out[i] = s / min(i + 1, w)
    return out

@njit(cache=True, fastmath=True)
def rsi(close: np.ndarray, w: int=14) -> np.ndarray:
    n = close.size
    out = np.empty(n, dtype=np.float32)
    gain = 0.0; loss = 0.0
    for i in range(1, n):
        d = close[i] - close[i-1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        gain = (gain*(w-1) + g) / w
        loss = (loss*(w-1) + l) / w
        rs = (gain / (loss + 1e-8))
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    out[0] = 50.0
    return out

# engine/features/build_features.py
import os, math
import polars as pl
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from engine.data.io import write_parquet
from engine.data.labeling import make_horizon_labels
from .vector_math import rolling_mean, rsi

CHUNK_DAYS = 250  # tune to keep arrays ≤ ~32MB/worker
OUT_BASE = "data/features"

def _build_one(symbol: str, df_sym: pl.DataFrame) -> list[str]:
    paths = []
    df_sym = df_sym.sort("date")
    pdf = df_sym.to_pandas()
    close = pdf["close"].to_numpy(dtype=np.float32)

    # features
    f_sma20 = rolling_mean(close, 20)
    f_rsi14 = rsi(close, 14)

    # assemble
    pdf["f_sma20"] = f_sma20
    pdf["f_rsi14"] = f_rsi14
    pdf = make_horizon_labels(pdf)

    # write by year
    pl_df = pl.from_pandas(pdf)
    for yr, g in pl_df.partition_by("date", as_dict=True).items():
        year = int(pl.col("date").dt.year().alias("y").cast(pl.Int32).select().collect()[0,0])  # safe fallback
        year = int(g.select(pl.col("date").dt.year()).to_series().unique().to_list()[0])
        paths.append(write_parquet(g, OUT_BASE, symbol, year))
    return paths

def build_all(raw_parquet_base: str, max_workers: int | None = None) -> list[str]:
    lf = pl.scan_parquet(f"{raw_parquet_base}/symbol=*/year=*/part.parquet")
    symbols = lf.select(pl.col("symbol")).unique().collect()["symbol"].to_list()
    out_paths: list[str] = []
    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as ex:
        futures = []
        for sym in symbols:
            df_sym = lf.filter(pl.col("symbol")==sym).collect()
            futures.append(ex.submit(_build_one, sym, df_sym))
        for fut in futures:
            out_paths.extend(fut.result())
    return out_paths


DoD

End-to-end feature build for a 10-symbol toy universe < 60s on 7950X3D.

Peak RSS per worker ≤ ~1.5× input chunk size; no pandas groupby in hot loops.

Benchmark target

≥ 500k rows/minute feature throughput on CPU for simple features above.

3) Specialists on Zero-Copy Panels

Files

engine/features/specialists.py

# engine/features/specialists.py
import numpy as np
from joblib import Parallel, delayed

def spec_mean_reversion(close: np.ndarray, sma: np.ndarray) -> np.ndarray:
    z = (close - sma) / (np.abs(sma) + 1e-6)
    score = 50.0 - np.clip(z*100.0, -50.0, 50.0)
    return score.astype(np.float32)

def run_specialists(panel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # panel: {"close": np.ndarray, "f_sma20": np.ndarray, ...}
    close = panel["close"]; sma = panel["f_sma20"]
    funcs = [("spec_mr", spec_mean_reversion, (close, sma))]
    outs = Parallel(n_jobs=-1, prefer="threads")(
        delayed(fn)(*args) for _, fn, args in funcs
    )
    return {name: arr for (name, _, _), arr in zip(funcs, outs)}


DoD

Specialists produce spec_<name>_score arrays with no intermediate DataFrame copies.

Parallelism uses threads unless Python loops dominate (then switch to processes).

4) Time-Series CV & Calibration (fold × specialist)

Files

engine/models/run_cv.py

# engine/models/run_cv.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

def purged_folds(n, n_splits=5, purge=5):
    step = n // n_splits
    for i in range(n_splits):
        start = i*step; end = (i+1)*step if i < n_splits-1 else n
        tr_idx = np.r_[0:max(0, start-purge), min(n, end+purge):n]
        te_idx = np.r_[start:end]
        yield tr_idx, te_idx

def calibrate(scores: np.ndarray, y: np.ndarray, method="platt"):
    if method == "platt":
        lr = LogisticRegression(max_iter=200).fit(scores.reshape(-1,1), y)
        return lambda s: lr.predict_proba(s.reshape(-1,1))[:,1]
    else:
        iso = IsotonicRegression(out_of_bounds="clip").fit(scores, y)
        return lambda s: iso.predict(s)

def cv_specialist(scores: np.ndarray, y: np.ndarray):
    metrics = []
    for tr, te in purged_folds(len(scores)):
        f = calibrate(scores[tr], y[tr], method="platt")
        p = f(scores[te])
        metrics.append({
            "auc": float(roc_auc_score(y[te], p)),
            "brier": float(brier_score_loss(y[te], p)),
        })
    return metrics


DoD

For each horizon, report fold metrics JSON under reports/cv/.

Evidence of reusing shared arrays (e.g., numpy.memmap) if datasets exceed RAM.

5) LSTM Specialist on ROCm (7900 XTX)

Files

engine/models/spec_lstm.py

engine/models/train_lstm_spec.py

# engine/models/spec_lstm.py
import torch, torch.nn as nn

class LSTMSpec(nn.Module):
    def __init__(self, in_dim: int, hidden: int=128, layers: int=2, dropout: float=0.1, out_dim: int=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))
    def forward(self, x):
        y, _ = self.lstm(x)
        return self.head(y[:, -1, :]).squeeze(-1)

# engine/models/train_lstm_spec.py
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .spec_lstm import LSTMSpec

class SeqDS(Dataset):
    def __init__(self, X, y, seq_len=128):
        self.X = X; self.y = y; self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, i):
        j = i + self.seq_len
        return self.X[i:j], self.y[j]

def train_one(X, y, in_dim, epochs=10, bs=512, seq_len=128, amp=True, compile_model=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = SeqDS(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), seq_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    model = LSTMSpec(in_dim=in_dim).to(device)
    if compile_model: model = torch.compile(model, mode="max-autotune")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
    return model


DoD

Trains with AMP + torch.compile on ROCm; plateaus without NaNs.

Inference throughput ≥ 50k sequences/s (short seq_len) on 7900 XTX.

6) Meta Model (ensemble over specialists + LSTM)

Files

engine/models/train_meta.py

engine/models/infer_meta.py

# engine/models/train_meta.py
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
import joblib, json
from pathlib import Path

def train_meta(X: np.ndarray, y: np.ndarray, outdir: str, name: str):
    base = LogisticRegressionCV(max_iter=1000, cv=5, n_jobs=-1)
    clf = CalibratedClassifierCV(base, cv=5, method="isotonic").fit(X, y)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, f"{outdir}/{name}.joblib")
    with open(f"{outdir}/{name}.json", "w") as f:
        json.dump({"features": X.shape[1]}, f)
    return clf

# engine/models/infer_meta.py
import numpy as np, joblib

def load_meta(path: str):
    return joblib.load(path)

def infer_meta(clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)[:,1].astype(np.float32)


DoD

Trained meta saved as .joblib; calibration curve looks sensible (manual spot-check).

Optional ONNX export if needed later.

7) Market Regime Tagger

Files

engine/regime/regime.py

import numpy as np

def tag_regime(index_close: np.ndarray, short=20, long=50, volw=20) -> np.ndarray:
    ma_s = np.convolve(index_close, np.ones(short)/short, mode="same")
    ma_l = np.convolve(index_close, np.ones(long)/long, mode="same")
    trend = np.sign(ma_s - ma_l)
    ret = np.diff(np.log(index_close), prepend=index_close[0])
    vol = np.sqrt(np.convolve(ret**2, np.ones(volw)/volw, mode="same"))
    # 0: consolidate, 1: bull, -1: bear, 2: oscillate (high vol, low trend)
    out = np.zeros_like(index_close, dtype=np.int8)
    out[trend>0] = 1; out[trend<0] = -1
    out[(np.abs(trend)<1e-6) & (vol>np.median(vol))] = 2
    return out


DoD

Produces market_regime int8 per day; distribution is not degenerate.

8) Backtester Hot Path (NumPy/Numba) + Async Logging

Files

engine/backtest/simple_daily.py

engine/backtest/logger.py

# engine/backtest/logger.py
import queue, threading, json
from pathlib import Path

class AsyncWriter:
    def __init__(self, outpath: str, batch_size=256):
        self.q = queue.Queue()
        self.path = Path(outpath); self.batch = []; self.batch_size = batch_size
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()
    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                if self.batch: self._flush(); break
            else:
                self.batch.append(item)
                if len(self.batch) >= self.batch_size: self._flush()
    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for x in self.batch: f.write(json.dumps(x)+"\n")
        self.batch.clear()
    def write(self, obj): self.q.put(obj)
    def close(self): self.q.put(None); self.th.join()

# engine/backtest/simple_daily.py
import numpy as np
from numba import njit
from .logger import AsyncWriter

@njit(cache=True, fastmath=True)
def apply_slippage_fee(price, weight, fee_bps=1.0):
    # toy model, replace with your fill logic
    return price * (1.0 + np.sign(weight) * fee_bps * 1e-4)

@njit(cache=True, fastmath=True)
def rebalance(weights_prev, target_weights, price):
    # simple one-shot rebalance to targets
    exec_price = apply_slippage_fee(price, target_weights - weights_prev)
    return target_weights, exec_price

def run_backtest(prices: np.ndarray, signals: np.ndarray, outlog: str):
    """
    prices: [T, N], signals: [T, N] in [0,1] probability -> map to weights.
    """
    writer = AsyncWriter(outlog)
    T, N = prices.shape
    w = np.zeros((N,), dtype=np.float32)
    for t in range(T):
        tw = (signals[t] - 0.5) * 2.0  # [-1,1]
        w, px = rebalance(w, tw, prices[t])
        writer.write({"t": int(t), "pnl": float(np.dot(w, prices[t]))})
    writer.close()


DoD

Backtest runs without DataFrame allocations inside the loop.

Logging does not stall the loop (async batch writes).

9) Jobs & Scheduler

Files

engine/jobs/historical_build_and_train.py

engine/jobs/live_paper_trade.py

# engine/jobs/historical_build_and_train.py
from engine.features.build_features import build_all
from engine.models.run_cv import cv_specialist
from engine.models.train_meta import train_meta
# glue: load Parquet → assemble X,y → run cv → train meta

# engine/jobs/live_paper_trade.py
# ingest latest bars → incremental features → specialists → meta → paper orders → log


DoD

Two entrypoints that execute without manual intervention on a small universe.

10) CLI

File

engine/cli.py

import argparse

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build-features")
    b.add_argument("--raw", required=True)
    b.add_argument("--workers", type=int, default=0)

    l = sp.add_parser("train-lstm")
    l.add_argument("--seq-len", type=int, default=128)
    l.add_argument("--epochs", type=int, default=10)
    l.add_argument("--device", choices=["cpu","gpu"], default="gpu")

    m = sp.add_parser("train-meta")
    m.add_argument("--horizon", type=int, choices=[3,7,14], default=7)

    bt = sp.add_parser("backtest")
    bt.add_argument("--log", required=True)

    args = ap.parse_args()
    # dispatch…

if __name__ == "__main__":
    main()


DoD

python -m engine.cli build-features --raw data/raw runs and writes Parquet.

Other subcommands parse flags and call corresponding modules.

11) Evaluation & Feedback Loop (3/7/14-day)

Files

engine/eval/metrics.py

engine/eval/eval_outcomes.py

# engine/eval/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def classification_report(y_true, y_prob):
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

# engine/eval/eval_outcomes.py
# join predictions at t with realized labels at t+H; compute metrics + decile returns by regime


DoD

Produces per-horizon JSON with AUC/PR-AUC/Brier and decile returns.

12) Inference Path (fast alerts)

Files

engine/serve/infer.py

# engine/serve/infer.py
# load frozen features → specialists → meta (CPU) + LSTM (GPU) → probability per horizon
# provide infer_one(symbol, now) and batch variants


DoD

infer_one() returns dict with {h3, h7, h14} probabilities and the feature snapshot used.

13) Persistence & Repro

Every module writes a JSON sidecar {git_sha, cmd, args, timings, rss_mb, rows} to reports/….

Artifacts layout:

reports/
  cv/
  meta/
  lstm/
  eval/
models/
  meta_*.joblib
  lstm_*.pt
data/
  raw/  features/  predictions/

Performance & Cache Locality (7950X3D)

Chunk symbols into ~250 trading days per worker initially; tune so each worker’s live arrays ≤ ~32 MB.

Use float32; avoid object dtypes. Convert once at ingress.

Processes at the symbol-chunk level; keep NumPy threads to 1–2 inside each worker (see env script).

If you pin affinities (Linux), keep a worker per CCD if needed; otherwise let the scheduler handle it.

AMD GPU (7900 XTX ROCm) Notes

torch.cuda.is_available() should return True on ROCm.

Enable AMP (autocast + GradScaler) and torch.compile(model, mode="max-autotune").

Set torch.set_float32_matmul_precision("high").

Use pinned DataLoader memory, num_workers=4–8, prefetch_factor=2.

Quick Acceptance Script (Smoke)

Create scripts/smoke.sh:

#!/usr/bin/env bash
set -euo pipefail
source env/openblas_threads.sh
python -m engine.cli build-features --raw data/raw
python -m engine.cli train-lstm --seq-len 64 --epochs 1 --device gpu || true
python -m engine.cli train-meta --horizon 7
python -m engine.cli backtest --log reports/backtest.jsonl


DoD

Smoke completes on toy data without errors; artifacts created.

Next Steps (You/ Codex)

Implement the stubs exactly as given above.

Replace toy features with your full specialist set (still zero-copy NumPy buffers).

Wire historical_build_and_train.py to:

build features → label → compute specialists → LSTM train (optional) → CV → train meta → save.

Wire live_paper_trade.py to:

incremental features → specialists → meta/LSTM inference → log predictions + inputs.

Add regime column into training matrices and condition meta weights by regime (e.g., include one-hot regimes in X or fit per-regime models).

Success Criteria

Feature throughput ≥ 500k rows/min on CPU with the base set.

LSTM training uses GPU (AMP + compile) and does not OOM.

Backtest loop shows no DataFrame allocations in the hot path.

All artifacts + sidecars reproducibly written under models/ and reports/.


 14) ChatGPT5's response:
 ✅ What I agree with (brief)

Pin threads at startup (avoid BLAS oversubscription vs. process pool).

Single source of truth for schema/dtypes.

Don’t bounce between polars → pandas → polars in the hot path.

Per-symbol partitioning and cache-sized chunks.

CV × specialist parallelism with shared arrays.

LSTM on ROCm with AMP + torch.compile.

Migration notes to prevent duplicated “old vs new” paths.

🔧 Improvements I recommend

Startup guard for threads + device printout
Print thread envs and CUDA/ROCm status on import so every script shows its perf posture.

# engine/__init__.py
import os, torch, warnings
def _post_init():
    env = {k: os.environ.get(k) for k in
           ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]}
    print(f"[engine] threads={env}  torch_cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[engine] device_name={torch.cuda.get_device_name(0)}")
    else:
        warnings.warn("GPU not detected; training will run on CPU.")
_post_init()


Schema first, NumPy everywhere
Make schema.py declare dtypes + field order, and write array-level label/feature functions. That lets you reuse code from polars scanners, pandas legacy, and Numba without copies.

# engine/data/schema.py
HORIZONS = (3,7,14)
DTYPES = dict(
    date="datetime64[ns]", symbol="category",
    open="float32", high="float32", low="float32", close="float32", volume="float32"
)
FEATURES_ORDER = ["f_sma20","f_rsi14"]  # extend here


Chunk with “halo” (no cross-chunk leakage)
When chunking by days, you must overlap each chunk by halo = max(rolling_windows, max(HORIZONS)) to compute rolling stats and future labels without leakage or NaNs at edges. Drop the halo when writing.

Polars scanner in, NumPy out, Parquet once
Don’t collect a whole symbol to pandas. Scan → filter by symbol → iterate (year, chunk_id) groups → materialize to NumPy arrays → compute features/labels → write Parquet. No DataFrame round-trips.

# engine/features/build_features.py (shape-safe, no pandas in hot loop)
import polars as pl, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from engine.data.schema import HORIZONS
from engine.features.vector_math import rolling_mean, rsi
from engine.data.io import write_parquet

CHUNK_DAYS = 250
HALO = max(20, max(HORIZONS))  # example: SMA20 + horizons

def _iter_symbol_chunks(lf_symbol: pl.LazyFrame, chunk_days=CHUNK_DAYS, halo=HALO):
    # Expect partitions: symbol=SYM/year=YYYY/part.parquet
    df = lf_symbol.sort("date").select(["date","symbol","open","high","low","close","volume"]).collect(streaming=True)
    # Add row index to slice in fixed day windows with halo overlap
    df = df.with_row_count("rn")
    n = df.height
    start = 0
    while start < n:
        stop = min(n, start + chunk_days + halo)
        yield df.slice(start, stop - start), start, stop
        start += chunk_days

def _build_symbol(symbol: str, lf_base: pl.LazyFrame, out_base: str) -> list[str]:
    paths = []
    lf_sym = lf_base.filter(pl.col("symbol")==symbol)
    for chunk_df, start, stop in _iter_symbol_chunks(lf_sym):
        # Extract NumPy views
        close = chunk_df["close"].to_numpy().astype(np.float32, copy=False)
        # features (Numba kernels)
        f_sma20 = rolling_mean(close, 20)
        f_rsi14  = rsi(close, 14)
        # labels (vectorized)
        labels = {}
        for h in HORIZONS:
            fut = np.full_like(close, np.nan, dtype=np.float32)
            if h < len(close): fut[:-h] = (close[h:] - close[:-h]) / close[:-h]
            labels[f"label_ret_h{h}"] = fut
            labels[f"label_bin_h{h}"] = (fut > 0.0).astype(np.int8)

        # Drop leading HALO rows except for the first chunk
        drop = HALO if start>0 else 0
        take = len(close) - drop
        frame = pl.DataFrame({
            "date": chunk_df["date"].to_numpy()[drop:],
            "symbol": np.repeat(symbol, take),
            "open":  chunk_df["open"].to_numpy()[drop:].astype(np.float32, copy=False),
            "high":  chunk_df["high"].to_numpy()[drop:].astype(np.float32, copy=False),
            "low":   chunk_df["low"].to_numpy()[drop:].astype(np.float32, copy=False),
            "close": close[drop:],
            "volume":chunk_df["volume"].to_numpy()[drop:].astype(np.float32, copy=False),
            "f_sma20": f_sma20[drop:], "f_rsi14": f_rsi14[drop:],
            **{k:v[drop:] for k,v in labels.items()}
        })
        # Write once per real year present in this chunk
        years = frame.select(pl.col("date").dt.year()).unique().to_series().to_list()
        for yr in years:
            paths.append(write_parquet(frame.filter(pl.col("date").dt.year()==yr), out_base, symbol, int(yr)))
    return paths

def build_all(raw_base: str, out_base: str, max_workers: int|None=None) -> list[str]:
    lf = pl.scan_parquet(f"{raw_base}/symbol=*/year=*/part.parquet")
    symbols = lf.select(pl.col("symbol")).unique().collect()["symbol"].to_list()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_build_symbol, s, lf, out_base) for s in symbols]
        out  = sum((f.result() for f in futs), [])
    return out


Zero-copy “panel” with shared memory (no dict thrash)
Back specialist inputs with a single C-contiguous float32 array ([T, F]) plus a column registry (name → view). Optionally place it on a numpy.memmap so workers read once (joblib memory mapping).

# engine/features/panel.py
import numpy as np
class Panel:
    def __init__(self, mat: np.ndarray, cols: list[str]):
        assert mat.flags['C_CONTIGUOUS'] and mat.dtype==np.float32
        self.mat, self.cols = mat, cols; self._idx = {c:i for i,c in enumerate(cols)}
    def col(self, name: str) -> np.ndarray: return self.mat[:, self._idx[name]]
    def view(self, names: list[str]) -> np.ndarray: return self.mat[:, [self._idx[n] for n in names]]

# engine/features/specialists.py (threads for NumPy work)
from joblib import Parallel, delayed
from .panel import Panel
import numpy as np

def spec_mean_reversion(p: Panel) -> np.ndarray:
    close, sma = p.col("close"), p.col("f_sma20")
    z = (close - sma) / (np.abs(sma) + 1e-6)
    score = np.clip(-z, -1.0, 1.0).astype(np.float32)  # keep your [-1,1] convention
    return score

REGISTRY = [("spec_mr", spec_mean_reversion)]

def run_specialists(panel: Panel, n_jobs: int=-1) -> dict[str, np.ndarray]:
    outs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(fn)(panel) for _,fn in REGISTRY)
    return {name: arr for (name,_), arr in zip(REGISTRY, outs)}


CV × specialist parallelism (explicit)
Use joblib over specialists, folds serial inside worker (or swap). Keep scores as memory-mapped arrays to avoid per-worker duplicates.

# engine/models/run_cv.py (key pattern)
from joblib import Parallel, delayed
import numpy as np

def cv_one_specialist(scores_mm: np.memmap, y_mm: np.memmap, folds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    metrics=[]
    for tr, te in folds:
        clf = LogisticRegression(max_iter=200).fit(scores_mm[tr].reshape(-1,1), y_mm[tr])
        p = clf.predict_proba(scores_mm[te].reshape(-1,1))[:,1]
        metrics.append(dict(auc=float(roc_auc_score(y_mm[te], p)),
                            brier=float(brier_score_loss(y_mm[te], p))))
    return metrics

def run_cv_all(specialist_scores: dict[str, np.memmap], y_mm: np.memmap, folds, n_jobs=-1):
    return {name: m for name, m in zip(specialist_scores.keys(),
            Parallel(n_jobs=n_jobs)(delayed(cv_one_specialist)(mm, y_mm, folds)
                                    for name, mm in specialist_scores.items()))}


ROCm notes + clean CPU fallback
Document env knobs without breaking runs:

Prefer device="cuda" (ROCm appears as CUDA in PyTorch).

Support HIP_VISIBLE_DEVICES to pick 0/1 etc.

If GPU missing, don’t crash—print fallback and continue on CPU.

Meta model options
Keep LR+isotonic as default (fast + calibrated). Add LightGBM CPU path (Histogram + GOSS) for stronger nonlinearities. Only enable GPU trees if you’ve verified ROCm for your box; otherwise it becomes a yak-shave.

Incremental/live features
Add an “incremental updater” that maintains rolling kernels with O(1) update (queue/sum) and appends just the last row for live paper trading. Avoid recomputing full windows.

Observability (cache & perf)
Log per task: input rows, bytes, array shapes, peak RSS, elapsed, workers. This is how you verify the V-Cache benefits are real.

📦 Migration Notes (prevent code drift)

Replaces:

engine/features/build_features.py (old) → new chunk/halo builder

engine/features/specialists.py (old) → panel-based specialists

engine/models/run_cv.py (old) → shared-mem, joblib CV

Delete after DoD passes: any legacy pandas groupby builders, per-row rolling ops, CSV writers.

Keep (for now): legacy pandas paths behind a --legacy flag only for regression testing.

✅ Updated “Definition of Done” per stage

Env/threads

Every script prints thread envs + device at startup; OMP_NUM_THREADS=1 (or 2) is visible.

Builder

Processes chunks with halo; writes Parquet once per (symbol, year); no pandas in hot loop; peak RSS per worker ≤ 1.5× chunk bytes.

Specialists

Inputs come from a single C-contiguous float32 matrix via Panel; no per-specialist copies; scores in [-1,1].

CV

scores & y stored as np.memmap; joblib over specialists; fold metrics JSON written; total memory nearly constant as workers scale.

LSTM

AMP + torch.compile; ROCm path works; CPU fallback tested; DataLoader uses pinned memory; no NaNs.

Meta

Baseline LR+isotonic saved; optional LightGBM CPU path; calibration plot saved.

Backtest

Hot loop has no DataFrame allocations; async logger never blocks; decisions/sec increases vs. old path.

Eval

Joins predictions at t with realized at t+H; reports AUC/PR-AUC/Brier + decile returns by regime.

✨ Small extras that pay off

Feature registry: central list of feature names + fn pointers; audit order and reuse.

Seeded reproducibility: set np.random.seed, torch.manual_seed at each entrypoint.

Calendar correctness: label horizons by trading days (use trading-calendars or your existing calendar).

Array dtypes: assert float32 at ingress; ban object dtypes with a helper assert_no_object(df) in tests.

Affinity (optional): on Linux, you can pin CV workers to a CCD to keep them on the 3D V-Cache CCD (psutil.Process().cpu_affinity([...])), but don’t overcomplicate until needed.
