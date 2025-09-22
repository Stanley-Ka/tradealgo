from __future__ import annotations

import os
from dataclasses import dataclass


try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "settings.toml")


@dataclass
class Settings:
    project_name: str
    mode: str
    timezone: str
    storage_root: str
    universe: str
    bar_size: str
    broker_default: str
    max_name_weight: float
    max_gross: float
    max_net: float
    sector_max: float
    alpha: float
    w_max: float

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        p = path or SETTINGS_PATH
        with open(p, "rb") as f:
            cfg = tomllib.load(f)
        proj = cfg.get("project", {})
        data = cfg.get("data", {})
        brokers = cfg.get("brokers", {})
        risk = cfg.get("risk", {})
        sizing = cfg.get("sizing", {})
        return cls(
            project_name=proj.get("name", "trading-engine"),
            mode=proj.get("mode", "dev"),
            timezone=proj.get("timezone", "US/Eastern"),
            storage_root=data.get("storage_root", "data"),
            universe=data.get("universe", "sp500"),
            bar_size=data.get("bar_size", "1D"),
            broker_default=brokers.get("default", "ibkr"),
            max_name_weight=float(risk.get("max_name_weight", 0.10)),
            max_gross=float(risk.get("max_gross", 1.0)),
            max_net=float(risk.get("max_net", 0.5)),
            sector_max=float(risk.get("sector_max", 0.30)),
            alpha=float(sizing.get("alpha", 1.0)),
            w_max=float(sizing.get("w_max", 0.10)),
        )

