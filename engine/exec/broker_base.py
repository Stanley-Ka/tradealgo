from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Broker(ABC):
    """Abstract broker interface for placing and managing orders."""

    @abstractmethod
    def get_positions(self) -> Any:  # narrow later
        raise NotImplementedError

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "MKT",
        limit_price: float | None = None,
        tif: str = "DAY",
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def cancel_all(self) -> Any:
        raise NotImplementedError

