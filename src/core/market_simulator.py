# core/market_simulator.py
from abc import ABC, abstractmethod
import numpy as np

class MarketSimulator(ABC):
    @abstractmethod
    def simulate(self, steps: int) -> np.ndarray:
        """Simulate mid-prices and return an array of prices."""
        pass
