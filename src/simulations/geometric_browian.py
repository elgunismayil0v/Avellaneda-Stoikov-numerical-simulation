from src.core.market_simulator import MarketSimulator
import numpy as np

class GeometricBrowianMotion(MarketSimulator):
    def __init__(self, NoOfStep : int, S0 : float, sigma: float, seed: int):
        self.NoOfStep = NoOfStep
        self.S0 = S0
        self.seed = seed
        self.sigma = sigma
        
    def simulate(self) -> np.ndarray:
        np.random.seed(self.seed)
        S = np.zeros(self.NoOfStep + 1)
        S[0] = self.S0
        dt = 1 / self.NoOfStep
        dW = np.random.normal(0, np.sqrt(dt), self.NoOfStep)

        for i in range(1, self.NoOfStep + 1):
            S[i] = S[i - 1] * np.exp(
            (0.5 * self.sigma**2) * dt + self.sigma * dW[i - 1])

        return S
    


            
        
        
    