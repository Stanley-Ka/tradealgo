"""Tradier REST adapter (stub)."""

from __future__ import annotations

from typing import Any

from .broker_base import Broker


class TradierBroker(Broker):
    def __init__(self, token: str | None = None, account_id: str | None = None) -> None:
        self.token = token
        self.account_id = account_id

    def get_positions(self) -> Any:
        return []

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        tif: str = "day",
    ) -> Any:
        _ = (symbol, qty, side, order_type, limit_price, tif)
        # TODO: implement REST call
        return {"status": "submitted"}

    def cancel_all(self) -> Any:
        return {"status": "ok"}
