from src.core.pricing_strategy import PricingStrategy
import numpy as np

class AvellanedaStoikovStrategyAbm(PricingStrategy):
    def __init__(self, gamma: float, sigma: float, k: float):
        """
        Args:
            gamma (float): Risk aversion coefficient.
            sigma (float): Volatility.
            k (float): Market depth parameter.
        """
        self.gamma = gamma
        self.k = k
        self.sigma = sigma


    def calculate_reservation_price(self, current_price: float, inventory: int, time_remaining: float) -> float:
        return current_price - inventory * self.gamma * self.sigma**2 * time_remaining

    def calculate_spread(self, current_price: float, inventory: int, time_remaining: float) -> tuple[float, float]:
        spread = self.gamma * self.sigma**2 * time_remaining
        spread += (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return spread / 2, spread / 2  # bid_spread, ask_spread 
