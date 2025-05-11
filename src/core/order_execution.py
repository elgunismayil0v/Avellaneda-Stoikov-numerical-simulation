# core/order_execution.py
from abc import ABC, abstractmethod

class OrderExecution(ABC):
    @abstractmethod
    def execute_orders(
        self,
        bid_price: float,
        ask_price: float,
        inventory: int,
        cash: float,
        dt: float
    ) -> tuple[int, float]:
        """Return updated (inventory, cash) after order execution."""
        pass
