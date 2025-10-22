from __future__ import annotations

"""Update YAML config paths (features, models, oof, calibrators, outputs).

Usage:
  python -m engine.tools.update_yaml_paths \
    --in engine/presets/swing_aggressive.yaml \
    --out engine/presets/swing_aggressive.local.yaml \
    --features D:\\EngineData\\datasets\\features_daily_1D.parquet \
    --meta-model D:\\EngineData\\models\\meta_lr.pkl \
    --oof D:\\EngineData\\datasets\\oof_specialists.parquet \
    --calibrators D:\\EngineData\\models\\spec_calibrators.pkl \
    --meta D:\\EngineData\\datasets\\meta_predictions.parquet \
    --sentiment D:\\EngineData\\datasets\\sentiment_finbert.parquet \
    --picks D:\\EngineData\\signals\\picks.csv \
    --universe engine/data/universe/us_all.txt \
    --meta-calibrator D:\\EngineData\\models\\meta_calibrator.pkl
"""

import argparse
from typing import Optional
import yaml  # type: ignore


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch YAML paths for presets/configs")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--features", default="")
    p.add_argument("--meta-model", dest="meta_model", default="")
    p.add_argument("--oof", default="")
    p.add_argument("--calibrators", default="")
    p.add_argument("--meta", default="")
    p.add_argument("--meta-calibrator", dest="meta_calibrator", default="")
    p.add_argument("--sentiment", default="")
    p.add_argument("--picks", default="")
    p.add_argument("--universe", default="")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    with open(args.inp, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    if args.features:
        paths["features"] = args.features
    if args.meta_model:
        paths["meta_model"] = args.meta_model
    if args.oof:
        paths["oof"] = args.oof
    if args.calibrators:
        paths["calibrators_pkl"] = args.calibrators
    if args.meta:
        paths["meta"] = args.meta
    if args.meta_calibrator:
        cal = (
            cfg.get("calibration", {})
            if isinstance(cfg.get("calibration"), dict)
            else {}
        )
        cal["meta_calibrator_pkl"] = args.meta_calibrator
        cfg["calibration"] = cal
    if args.sentiment:
        paths["sentiment_out"] = args.sentiment
    if args.picks:
        paths["picks_csv"] = args.picks
    if paths:
        cfg["paths"] = paths
    if args.universe:
        al = cfg.get("alert", {}) if isinstance(cfg.get("alert"), dict) else {}
        al["universe_file"] = args.universe
        cfg["alert"] = al
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
