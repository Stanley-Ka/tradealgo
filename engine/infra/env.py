from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def _parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out
    for ln in text:
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith(";"):
            continue
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k = k.strip()
        v = v.strip().strip("\"' ")
        if k:
            out[k] = v
    return out


def _find_repo_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    for p in [cur] + list(cur.parents):
        if (
            (p / ".git").exists()
            or (p / "scripts").exists()
            or (p / "pyproject.toml").exists()
        ):
            return p
    return cur.parent


def load_env_files() -> Dict[str, str]:
    """Load environment variables from scripts/api.env or .env if present.

    - Does not overwrite existing os.environ keys.
    - Returns a dict of keys that were injected into os.environ.
    """
    injected: Dict[str, str] = {}
    root = _find_repo_root()
    candidates = [root / "scripts" / "api.env", root / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        data = _parse_env_file(env_path)
        for k, v in data.items():
            if k not in os.environ and v is not None:
                os.environ[k] = v
                injected[k] = v
    return injected
