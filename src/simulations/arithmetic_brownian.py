# simulations/arithmetic_brownian.py

from src.core.market_simulator import MarketSimulator
import numpy as np

class ArithmeticBrownianMotion(MarketSimulator):
    """
    A class to simulate a price path using Arithmetic Brownian Motion (ABM).

    ABM models the evolution of asset prices using a linear stochastic differential equation:
    S(t+1) = S(t) + sigma * sqrt(dt) * Z,
    where Z is a standard normal random variable.

    Attributes:
        NoOfSteps (int): Number of time steps in the simulation.
        S0 (float): Initial asset price.
        sigma (float): Volatility coefficient (standard deviation).
    """

    def __init__(self, S0: float, sigma: float, NoOfSteps: int):
        """
        Initializes the ABM simulator with the given parameters.

        Args:
            NoOfSteps (int): Number of steps in the simulation.
            S0 (float): Initial price level.
            sigma (float): Volatility of the price process.
        """
        self.S0 = S0
        self.sigma = sigma
        self.NoOfSteps = NoOfSteps

    def simulate(self) -> np.ndarray:
        """
        Runs the simulation of Arithmetic Brownian Motion.

        Returns:
            np.ndarray: Simulated path of asset prices as a NumPy array.
        """
        dt = 1 / self.NoOfSteps
        S = np.zeros(self.NoOfSteps + 1)  # Initialize price array
        S[0] = self.S0  # Set initial price
        Z = np.random.normal(0, 1, self.NoOfSteps)  # Generate standard normal random variables

        # Generate the price path
        for i in range(1, self.NoOfSteps + 1):
            S[i] = S[i - 1] + self.sigma * np.sqrt(dt) * Z[i - 1]

        return S


