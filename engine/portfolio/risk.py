"""Basic risk checks (stubs)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderIntent:
    symbol: str
    target_weight: float
    notional: float
    spread_bps: float | None = None


def is_order_safe(order: OrderIntent, max_spread_bps: float = 50.0) -> bool:
    """Example safety check using quoted spread and weight bounds."""
    if abs(order.target_weight) > 1.0:
        return False
    if order.spread_bps is not None and order.spread_bps > max_spread_bps:
        return False
    return True
