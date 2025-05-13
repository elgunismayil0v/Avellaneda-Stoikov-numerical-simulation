# strategies/avellaneda_stoikov.py
from core.pricing_strategy import PricingStrategy
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
import numpy as np

class AvellanedaStoikovStrategyAbm(PricingStrategy):
    def __init__(self, gamma: float, k: float, inventory: int, abm:ArithmeticBrownianMotion):
        self.gamma = gamma
        self.k = k
        self.q = inventory
        self.sigma = abm.sigma
        self.T = abm.NoOfStep
        self.dt = 1 / self.T
        self.S = abm.simulate() 

    def calculate_reservation_price(self) -> float:
        t = 0
        reservation_price = np.zeros(self.T)
        for i in range(self.T):
            reservation_price[i] = self.S[i] - self.q * self.gamma * (self.sigma ** 2) * t
            t += self.dt
        return reservation_price

    def calculate_spread(self) -> np.ndarray:
        t = 0
        spread = np.zeros(self.T)
        for i in range(self.T):
            spread[i] = self.gamma * (self.sigma ** 2) * t + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
            t += self.dt
        return spread / 2
    
    def calculate_bid_ask(self) -> tuple[np.ndarray, np.ndarray]:
        bid = self.calculate_reservation_price() - self.calculate_spread()
        ask = self.calculate_reservation_price() + self.calculate_spread()
        return  bid, ask

