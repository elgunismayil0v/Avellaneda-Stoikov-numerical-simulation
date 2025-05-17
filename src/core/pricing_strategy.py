# core/pricing_strategy.py
from abc import ABC, abstractmethod


class PricingStrategy(ABC):
    @abstractmethod
    def calculate_reservation_price(self, current_price: float, inventory: int, time_remaining: float) -> float:
        pass

    @abstractmethod
    def calculate_spread(self, current_price: float, inventory: int, time_remaining: float) -> tuple[float, float]:
        """Return (bid_spread, ask_spread) relative to reservation price."""
        pass
