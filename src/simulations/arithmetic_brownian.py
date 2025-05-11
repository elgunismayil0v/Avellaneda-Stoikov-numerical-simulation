# simulations/arithmetic_brownian.py
from core.market_simulator import MarketSimulator

class ArithmeticBrownianMotion(MarketSimulator):
    def __init__(self, S0: float, sigma: float):
        self.S0 = S0
        self.sigma = sigma

    def simulate(self, steps: int, dt: float) -> np.ndarray:
        dW = np.random.normal(0, np.sqrt(dt), steps)
        return self.S0 + self.sigma * np.cumsum(dW)
