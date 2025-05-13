# simulations/arithmetic_brownian.py
from core.market_simulator import MarketSimulator
import numpy as np

class ArithmeticBrownianMotion(MarketSimulator):
    def __init__(self, NoOfStep: int, S0: float, sigma: float, seed: int):
        self.NoOfStep = NoOfStep
        self.S0 = S0
        self.sigma = sigma
        self.seed = seed

    def simulate(self) -> np.ndarray:
        np.random.seed(self.seed)
        dt = 1 / self.NoOfStep
        S = np.zeros(self.NoOfStep + 1)
        S[0] = self.S0
        Z = np.random.normal(0,1,self.NoOfStep)
        for i in range(1, self.NoOfStep + 1):
            S[i] = S[i - 1] + self.sigma * np.sqrt(dt) * Z[i - 1]
        
        return S
