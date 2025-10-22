from __future__ import annotations

"""Style and plan helpers.

Provides a stable mapping from a `--style` flag to a preset YAML file,
preferring a `.local.yaml` variant when present.

Also normalizes user-friendly `--plan` values to internal polygon_plan hints.
"""

from pathlib import Path
from typing import Optional


_STYLE_FILES = {
    # Swing-only presets
    "swing_aggressive": "engine/presets/swing_aggressive.yaml",
    "swing_conservative": "engine/presets/swing_conservative.yaml",
}


def resolve_style(style: Optional[str]) -> Optional[str]:
    if not style:
        return None
    key = str(style).strip().lower()
    if key not in _STYLE_FILES:
        return None
    base = Path(_STYLE_FILES[key])
    # Prefer .local.yaml if it exists alongside
    local = base.with_name(base.stem + ".local.yaml")
    if local.exists():
        return str(local)
    return str(base)


def normalize_plan(plan: Optional[str]) -> Optional[str]:
    if not plan:
        return None
    p = str(plan).strip().lower()
    # Map friendly plan names to internal polygon plan hints
    mapping = {
        "basic": "starter",
        "starter": "starter",
        "developer": "pro",
        "advanced": "enterprise",
        # pass-throughs
        "auto": "auto",
        "pro": "pro",
        "enterprise": "enterprise",
    }
    return mapping.get(p, p)
