# strategies/avellaneda_stoikov.py

from src.core.pricing_strategy import PricingStrategy
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
import numpy as np

class AvellanedaStoikovStrategy(PricingStrategy):
    def __init__(self, gamma: float, sigma: float, k: float, model_type: str = "ABM"):
        """
        Args:
            gamma (float): Risk aversion coefficient.
            sigma (float): Volatility.
            k (float): Market depth parameter.
            model_type (str): "ABM" or "GBM" to choose price dynamics.
        """
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.model_type = model_type.upper()

    def calculate_reservation_price(self, current_price, inventory, time_remaining):
        if self.model_type == "GBM":
            return current_price - inventory * self.gamma * current_price**2 * (1 - np.exp(-self.sigma**2 * time_remaining))
        else:  # ABM
            return current_price - inventory * self.gamma * self.sigma**2 * time_remaining

    def calculate_spread(self, current_price, inventory, time_remaining):
        if self.model_type == "GBM":
            spread = self.gamma * current_price**2 * (1 - np.exp(-self.sigma**2 * time_remaining))
        else:  # ABM
            spread = self.gamma * self.sigma**2 * time_remaining

        spread += (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return spread / 2, spread / 2




