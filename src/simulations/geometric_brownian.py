from src.core.market_simulator import MarketSimulator
import numpy as np
from typing import Optional

class GeometricBrownianMotion(MarketSimulator):
    """ A class to simulate asset prices using Geometric Brownian Motion (GBM).
    GBM models the asset price evolution using the exponential of a Brownian motion:
    S(t+1) = S(t) * exp((μ - 0.5 * σ²) * dt + σ * sqrt(dt) * Z)
    Here, μ is assumed to be 0 for simplicity (no drift).
    Attributes:
        NoOfSteps (int): Number of time steps in the simulation.
    S0 (float): Initial asset price.
    sigma (float): Volatility of the asset.
    """
    def __init__(self, S0: float, sigma: float, NoOfSteps: Optional[int] = None):
        """
        Initializes the GBM simulator with the provided parameters.
        Args : 
        NoOfSteps (int): Number of discrete time steps.
        S0 (float): Initial price of the asset.
        sigma (float): Volatility (standard deviation of returns).
        """ 
        self.NoOfStep = NoOfSteps
        self.S0 = S0
        self.sigma = sigma
        
    def simulate(self) -> np.ndarray:
        """
        Runs the Geometric Brownian Motion simulation.
        Returns:
        np.ndarray: Simulated price path as a NumPy array. 
        """
        S = np.zeros(self.NoOfStep + 1) # Array to store simulated prices
        S[0] = self.S0 # Set initial price
        dt = 1 / self.NoOfStep # Time step size
        Z = np.random.normal(0, 1, self.NoOfStep) # Standard normal random variables

        # Generate price path
        for i in range(1, self.NoOfStep + 1):
            # GBM price update formula
            S[i] = S[i - 1] * np.exp((-0.5 * self.sigma**2) * dt + self.sigma * Z[i - 1] * np.sqrt(dt))
        return S

    def simulate(self, steps: int) -> np.ndarray:
        """
        Runs the Geometric Brownian Motion simulation.
        Returns:
        np.ndarray: Simulated price path as a NumPy array.
        """
        S = np.zeros(steps + 1) # Array to store simulated prices
        S[0] = self.S0 # Set initial price
        dt = 1 / steps # Time step size
        Z = np.random.normal(0, 1, steps) # Standard normal random variables

        # Generate price path
        for i in range(1, steps + 1):
            # GBM price update formula
            S[i] = S[i - 1] * np.exp((-0.5 * self.sigma**2) * dt + self.sigma * Z[i - 1] * np.sqrt(dt))
        return S
    


    


            
        
        

