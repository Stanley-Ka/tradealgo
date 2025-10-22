from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Iterable, List, Tuple


TEXT_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".ps1",
    ".bat",
}
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "engine/data/equities",
    "mlruns",
    "docs",
}
MAX_FILE_BYTES = 1_000_000  # 1 MB


# Patterns with severity to reduce false positives in docs (.md/.txt)
# Tuple: (name, regex, severity) where severity is 'critical', 'info', or 'conditional'
PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "critical"),
    ("GitHub Token (ghp)", re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"), "critical"),
    (
        "GitHub Token (github_pat)",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}_[A-Za-z0-9]{59,}\b"),
        "critical",
    ),
    ("Slack Token", re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}\b"), "critical"),
    (
        "Private Key Block",
        re.compile(r"-----BEGIN (?:RSA |EC |)PRIVATE KEY-----"),
        "critical",
    ),
    # Broad base64-like strings cause false positives; raise threshold and mark as info
    ("Azure Key", re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"), "info"),
    # Generic API key: critical in code, info in docs
    (
        "Generic API key",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?"
        ),
        "conditional",
    ),
]


def is_text_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in TEXT_EXTS


def is_doc_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in {".md", ".txt"}


def list_staged_files() -> List[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], text=True
        )
        files = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return files
    except Exception:
        return []


def list_all_files(root: str = ".") -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        # filter skip dirs
        parts = set(dirpath.replace("\\", "/").split("/"))
        if parts & SKIP_DIRS:
            continue
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def scan_file(path: str) -> List[Tuple[str, int, str, str]]:
    res: List[Tuple[str, int, str, str]] = []
    if not is_text_file(path):
        return res
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return res
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            doc = is_doc_file(path)
            for i, line in enumerate(f, 1):
                for name, rx, sev in PATTERNS:
                    if rx.search(line):
                        eff = sev
                        if sev == "conditional":
                            eff = "info" if doc else "critical"
                        res.append((name, i, line.rstrip("\n"), eff))
    except Exception:
        pass
    return res


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan repository for potential secrets")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--staged", action="store_true", help="Scan only staged files")
    g.add_argument("--all", action="store_true", help="Scan all repo files")
    p.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit non-zero if any findings are detected",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    files: List[str]
    if args.staged:
        files = list_staged_files()
    else:
        files = list(list_all_files(".")) if args.all else list_staged_files()
        if not files:
            files = list(list_all_files("."))

    findings_crit: List[Tuple[str, str, int, str]] = []
    findings_info: List[Tuple[str, str, int, str]] = []
    for path in files:
        for name, ln, text, sev in scan_file(path):
            if sev == "critical":
                findings_crit.append((path, name, ln, text))
            else:
                findings_info.append((path, name, ln, text))

    def _safe_print(s: str) -> None:
        try:
            print(s)
        except UnicodeEncodeError:
            try:
                sys.stdout.write(
                    s.encode("ascii", errors="replace").decode("ascii") + "\n"
                )
            except Exception:
                pass

    if findings_info or findings_crit:
        _safe_print("[secrets] Potential secret findings:")
        for path, name, ln, text in findings_info:
            _safe_print(f" - {path}:{ln} [INFO:{name}] {text[:120]}")
        for path, name, ln, text in findings_crit:
            _safe_print(f" - {path}:{ln} [CRIT:{name}] {text[:120]}")
        if (args.fail_on_findings or args.staged) and findings_crit:
            # Fail only if critical findings are present
            return 1
    else:
        _safe_print("[secrets] No findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
