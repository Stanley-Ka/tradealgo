"""
Live/paper trading entry point stub.
Computes daily signals and submits orders via a broker adapter.
"""

from __future__ import annotations

from .infra.config import Settings


def main() -> None:
    settings = Settings.load()
    print("[LIVE] Loaded settings:", settings.project_name, settings.mode)
    print("[LIVE] TODO: implement signal computation and broker order routing.")


if __name__ == "__main__":
    main()

