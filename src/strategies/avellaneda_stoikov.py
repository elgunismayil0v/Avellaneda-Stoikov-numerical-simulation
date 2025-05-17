# strategies/avellaneda_stoikov.py

from src.core.pricing_strategy import PricingStrategy
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
import numpy as np

class AvellanedaStoikovStrategy(PricingStrategy):
    def __init__(self, gamma: float, sigma: float ,k: float):
        self.gamma = gamma
        self.k = k
        self.sigma = sigma

    def calculate_reservation_price(self, current_price, inventory, time_remaining):
        return current_price - inventory * self.gamma * self.sigma**2 * time_remaining

    def calculate_spread(self, current_price, inventory, time_remaining):
        spread = self.gamma * self.sigma**2 * time_remaining + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return spread / 2, spread / 2



