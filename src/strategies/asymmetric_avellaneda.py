import numpy as np
from core.pricing_strategy import PricingStrategy


class AsymmetricAvellanedaStoikovStrategy(PricingStrategy):
    def __init__(self, gamma: float, sigma: float, kappa: float):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa

    def calculate_reservation_price(self,
                                    current_price: float,
                                    inventory: int,
                                    time_remaining: float) -> float:
        return current_price - inventory * self.gamma * (self.sigma ** 2) * time_remaining

    def calculate_spreads(self,
                          current_price: float,
                          inventory: int,
                          time_remaining: float) -> tuple[float, float]:
        # Inventory risk component
        inventory_risk = self.gamma * (self.sigma ** 2) * time_remaining

        # Execution probability component
        execution_term = (1 / self.gamma) * np.log(1 + self.gamma / self.kappa)

        # Asymmetric spread calculation
        delta_bid = ((1 + 2 * inventory) * inventory_risk) / 2 + execution_term
        delta_ask = ((1 - 2 * inventory) * inventory_risk) / 2 + execution_term

        return delta_bid, delta_ask