"""Interactive Brokers adapter (stub).

Implement using `ib_insync` when ready.
"""

from __future__ import annotations

from typing import Any

from .broker_base import Broker


class IBKRBroker(Broker):
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        # TODO: connect via ib_insync.IB()

    def get_positions(self) -> Any:
        return []

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "MKT",
        limit_price: float | None = None,
        tif: str = "DAY",
    ) -> Any:
        _ = (symbol, qty, side, order_type, limit_price, tif)
        # TODO: implement real order routing
        return {"status": "submitted"}

    def cancel_all(self) -> Any:
        return {"status": "ok"}

