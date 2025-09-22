"""
Backtest entry point stub.
Wire data -> features (5 specialists) -> calibration -> meta-learner -> portfolio -> orders.
"""

from __future__ import annotations

from .infra.config import Settings


def main() -> None:
    settings = Settings.load()
    print("[BACKTEST] Loaded settings:", settings.project_name, settings.mode)
    print("[BACKTEST] TODO: implement data loading, labeling, and backtest loop.")
    print("[BACKTEST] See ENGINE_CAPABILITIES.txt for supported commands/settings.")


if __name__ == "__main__":
    main()
