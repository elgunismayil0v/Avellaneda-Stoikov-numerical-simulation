from src.core.market_simulator import MarketSimulator
import numpy as np

class GeometricBrownianMotion(MarketSimulator):
    """
    A class to simulate asset prices using Geometric Brownian Motion (GBM).

    GBM models the asset price evolution using the exponential of a Brownian motion:
        S(t+1) = S(t) * exp((μ - 0.5 * σ²) * dt + σ * sqrt(dt) * Z)
    Here, μ is assumed to be 0 for simplicity (no drift).

    Attributes:
        NoOfStep (int): Number of time steps in the simulation.
        S0 (float): Initial asset price.
        sigma (float): Volatility of the asset.
        seed (int): Seed for random number generator for reproducibility.
    """

    def __init__(self, NoOfStep: int, S0: float, sigma: float, seed: int):
        """
        Initializes the GBM simulator with the provided parameters.

        Args:
            NoOfStep (int): Number of discrete time steps.
            S0 (float): Initial price of the asset.
            sigma (float): Volatility (standard deviation of returns).
            seed (int): Seed for random number generator.
        """
        self.NoOfStep = NoOfStep
        self.S0 = S0
        self.sigma = sigma
        self.seed = seed

    def simulate(self) -> np.ndarray:
        """
        Runs the Geometric Brownian Motion simulation.

        Returns:
            np.ndarray: Simulated price path as a NumPy array.
        """
        np.random.seed(self.seed)  # Set seed for reproducibility
        S = np.zeros(self.NoOfStep + 1)  # Array to store simulated prices
        S[0] = self.S0  # Set initial price
        dt = 1 / self.NoOfStep  # Time step size
        Z = np.random.normal(0, 1, self.NoOfStep)  # Standard normal random variables

        # Generate price path
        for i in range(1, self.NoOfStep + 1):
            # GBM price update formula
            S[i] = S[i - 1] * np.exp(
                (-0.5 * self.sigma**2) * dt + self.sigma * Z[i - 1] * np.sqrt(dt)
            )

        return S

    


            
        
        
    