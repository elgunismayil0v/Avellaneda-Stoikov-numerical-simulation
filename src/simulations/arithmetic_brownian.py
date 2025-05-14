# simulations/arithmetic_brownian.py
from src.core.market_simulator import MarketSimulator
import numpy as np

import numpy as np
from src.core.market_simulator import MarketSimulator

class ArithmeticBrownianMotion(MarketSimulator):
    def __init__(self, S0: float, sigma: float, seed: int):
        self.S0 = S0
        self.sigma = sigma
        self.seed = seed

    def simulate(self, steps: int) -> np.ndarray:
        np.random.seed(self.seed)
        dt = 1 / steps
        S = np.zeros(steps + 1)
        S[0] = self.S0
        Z = np.random.normal(0, 1, steps)

        for i in range(1, steps + 1):
            S[i] = S[i - 1] + self.sigma * np.sqrt(dt) * Z[i - 1]

        return S


