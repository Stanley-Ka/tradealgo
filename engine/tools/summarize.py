from __future__ import annotations

"""Heuristic preset summarizer.

Generates a concise natural-language summary of a preset/config including
specialists enabled, intended timeframe, and basic position sizing guidance.
"""

from typing import Dict, Any, List


def _enabled(d: Any) -> bool:
    return not (isinstance(d, dict) and d.get("enabled") is False)


def _list_enabled_specs(spec_cfg: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for k in [
        "patterns",
        "technicals",
        "sequence",
        "breakout",
        "flow",
        "adx",
        "stoch_rsi",
        "williams_r",
        "cci",
        "lstm",
        "nlp",
    ]:
        if k in spec_cfg and _enabled(spec_cfg[k]):
            names.append(k)
    return names


def _spec_briefs(names: List[str]) -> List[str]:
    m = {
        "patterns": "candlestick/structure; short-term bar patterns",
        "technicals": "trend + momentum (SMA, RSI, MACD, BB guard)",
        "sequence": "EMA-of-returns over vol (drift/trend proxy)",
        "breakout": "Donchian + ATR breakout (trend continuation)",
        "flow": "OBV slope + MFI (volume accumulation/distribution)",
        "adx": "directional movement strength/trend filter",
        "stoch_rsi": "overbought/oversold oscillator (mean-reversion timing)",
        "williams_r": "overbought/oversold oscillator (fast MR)",
        "cci": "deviation from typical price (channel extremes)",
        "lstm": "sequence model (optional; if model provided)",
        "nlp": "news sentiment (requires provider/api)",
    }
    return [f"- {n}: {m.get(n, '')}".strip() for n in names]


def _timeframe_hint(cfg: Dict[str, Any]) -> str:
    # Heuristic: daily data implies swing/position; weekly rebalance -> swing/position; daily rebalance -> swing/short swing.
    rebalance = cfg.get("rebalance", "daily")
    top_k = int(cfg.get("top_k", cfg.get("alert", {}).get("top_k", 20)))
    risk = cfg.get("risk", {})
    max_atr = float(risk.get("max_atr_pct", 0.05) or 0.05)
    if rebalance == "weekly":
        return "Swing/position trading over days to weeks."
    if rebalance == "monthly":
        return "Position trades over weeks to months."
    # daily
    if max_atr <= 0.03 and top_k <= 20:
        return "Lower-volatility swing setups (multi-day)."
    if max_atr >= 0.06 and top_k >= 30:
        return "Broader momentum basket; active swing (multi-day)."
    return "General daily swing trading (multi-day)."


def _position_sizing_hint(cfg: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    risk = cfg.get("risk", {})
    max_atr = float(risk.get("max_atr_pct", 0.05) or 0.05)
    top_k = int(cfg.get("top_k", cfg.get("alert", {}).get("top_k", 20)))
    lines.append(f"Equal-weight across top-{top_k} picks is a simple baseline.")
    lines.append(
        "ATR sizing: weight ∝ 1/ATR% with caps; target per-trade risk 0.25–0.50% using 1.0–1.5× ATR stops."
    )
    lines.append(
        "Liquidity gate enforces ADV>min; keep single-name weight ≤ 10% ADV for execution."
    )
    if max_atr <= 0.03:
        lines.append(
            "Tighter stops (≈1× ATR) fit lower ATR caps; allow slightly larger weights."
        )
    else:
        lines.append(
            "Looser stops (≈1.5× ATR) for higher-volatility names; reduce per-name weight accordingly."
        )
    return lines


def summarize_preset(cfg: Dict[str, Any]) -> str:
    spec_cfg = cfg.get(
        "specialists", cfg.get("sensitivity", {})
    )  # backward compat with example file
    enabled = _list_enabled_specs(spec_cfg)
    briefs = _spec_briefs(enabled)
    risk = cfg.get("risk", {})
    alert = cfg.get("alert", {})
    lines: List[str] = []
    lines.append("Preset Summary")
    lines.append(f"- Universe: {alert.get('universe_file', 'custom')}")
    lines.append(f"- Top-K: {alert.get('top_k', cfg.get('top_k', 20))}")
    lines.append(f"- Rebalance: {cfg.get('rebalance', 'daily')}")
    lines.append(
        f"- Risk gates: min ADV ${risk.get('min_adv_usd', 1e7):,.0f}, max ATR% {100*float(risk.get('max_atr_pct', 0.05)):0.1f}, earnings blackout ±{risk.get('earnings_blackout', 2)}d"
    )
    if enabled:
        lines.append("- Specialists:")
        lines.extend(briefs)
    lines.append(f"- Intended use: {_timeframe_hint({**cfg, **alert})}")
    lines.append("- Position sizing notes:")
    lines.extend([f"  {t}" for t in _position_sizing_hint({**cfg, **alert})])
    return "\n".join(lines)
