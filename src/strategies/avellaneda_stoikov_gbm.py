from src.core.pricing_strategy import PricingStrategy
from src.simulations.geometric_browian import GeometricBrowianMotion
import numpy as np
import pandas as pd

class AvellanedaStoikovGBM(PricingStrategy):
    def __init__(self, gamma: float, inventory: int, k: float, gbm: GeometricBrowianMotion):
        self.gamma = gamma
        self.q = inventory
        self.sigma = gbm.sigma
        self.T = gbm.NoOfStep
        self.S = gbm.simulate()
        self.dt = 1 / self.T
        self.k = k
        
    def calculate_reservation_price(self) -> np.ndarray:
        t = 0
        reservation_price = np.zeros(self.T)
        for i in range(self.T):
            reservation_price[i] = self.S[i] - self.q * self.gamma * self.S[i] ** 2 * (1 - np.exp(-self.sigma**2 * (t)))
            t += self.dt
        return reservation_price
    
    def calculate_spread(self) -> np.ndarray:
        t = 0
        spread = np.zeros(self.T)
        for i in range(self.T):
            spread[i] = self.gamma * self.S[i] ** 2 * (1 - np.exp(-self.sigma**2 * (t))) + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
            t += self.dt
        return spread / 2
    
    def calculate_bid_ask(self) -> tuple[np.ndarray, np.ndarray]:
        bid = self.calculate_reservation_price() - self.calculate_spread()
        ask = self.calculate_reservation_price() + self.calculate_spread()
        return  bid, ask
    
        
        
  
  
