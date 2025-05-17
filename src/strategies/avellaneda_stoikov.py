from src.core.pricing_strategy import PricingStrategy
import numpy as np

class AvellanedaStoikovStrategy(PricingStrategy):
    def __init__(self, gamma: float, sigma: float, k: float, model_type: str = "ABM"):
        """
        Args:
            gamma (float): Risk aversion coefficient.
            sigma (float): Volatility.
            k (float): Market depth parameter.
            model_type (str): Either "ABM" or "GBM".
        """
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.model_type = model_type.upper()
        assert self.model_type in ["ABM", "GBM"], "model_type must be 'ABM' or 'GBM'"

    def calculate_reservation_price(self, current_price: float, inventory: int, time_remaining: float) -> float:
        if self.model_type == "GBM":
            return current_price - inventory * self.gamma * current_price**2 * (1 - np.exp(-self.sigma**2 * time_remaining))
        else:  # ABM
            return current_price - inventory * self.gamma * self.sigma**2 * time_remaining

    def calculate_spread(self, current_price: float, inventory: int, time_remaining: float) -> tuple[float, float]:
        if self.model_type == "GBM":
            spread = self.gamma * current_price**2 * (1 - np.exp(-self.sigma**2 * time_remaining))
        else:  # ABM
            spread = self.gamma * self.sigma**2 * time_remaining

        spread += (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return spread / 2, spread / 2  # bid_spread, ask_spread
