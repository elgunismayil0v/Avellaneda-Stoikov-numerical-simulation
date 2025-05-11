# strategies/avellaneda_stoikov.py
from core.pricing_strategy import PricingStrategy

class AvellanedaStoikovStrategy(PricingStrategy):
    def __init__(self, gamma: float, sigma: float, k: float):
        self.gamma = gamma
        self.sigma = sigma
        self.k = k

    def calculate_reservation_price(self, current_price: float, inventory: int, time_remaining: float) -> float:
        return current_price - inventory * self.gamma * (self.sigma ** 2) * time_remaining

    def calculate_spread(self, current_price: float, inventory: int, time_remaining: float) -> tuple[float, float]:
        spread = self.gamma * (self.sigma ** 2) * time_remaining + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return (spread / 2, spread / 2)

