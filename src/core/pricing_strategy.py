# core/pricing_strategy.py
from abc import ABC, abstractmethod
import numpy as np


class PricingStrategy(ABC):
    @abstractmethod
    def calculate_reservation_price(self) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_spread(self) -> np.ndarray:
        """Return (bid_spread, ask_spread) relative to reservation price."""
        pass
    @abstractmethod
    def calculate_bid_ask(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes bid and ask prices at each time step.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays of bid and ask prices.
        """
        pass

