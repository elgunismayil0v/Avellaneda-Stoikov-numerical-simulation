from core.pricing_strategy import PricingStrategy
from src.simulations.geometric_brownian import GeometricBrownianMotion
import numpy as np

class AvellanedaStoikovGBM(PricingStrategy):
    """
    Avellaneda-Stoikov market-making strategy using Geometric Brownian Motion (GBM) for price simulation.

    This model adjusts the bid and ask prices based on inventory risk and market volatility.

    Attributes:
        gamma (float): Risk aversion parameter.
        q (int): Current inventory level.
        k (float): Market depth parameter.
        sigma (float): Volatility of the asset.
        T (int): Number of time steps.
        S (np.ndarray): Simulated asset prices using GBM.
        dt (float): Time step size.
    """

    def __init__(self, gamma: float, inventory: int, k: float, gbm: GeometricBrownianMotion):
        """
        Initializes the pricing strategy with model parameters and simulated price path.

        Args:
            gamma (float): Trader's risk aversion.
            inventory (int): Current inventory position.
            k (float): Liquidity parameter (depth of order book).
            gbm (GeometricBrowianMotion): GBM simulator instance.
        """
        self.gamma = gamma
        self.q = inventory
        self.sigma = gbm.sigma
        self.T = gbm.NoOfStep
        self.S = gbm.simulate()
        self.dt = 1 / self.T
        self.k = k

    def calculate_reservation_price(self) -> np.ndarray:
        """
        Calculates the reservation price at each time step.

        Returns:
            np.ndarray: Reservation price series.
        """
        t = 0
        reservation_price = np.zeros(self.T)
        for i in range(self.T):
            # Reservation price includes inventory risk adjustment
            reservation_price[i] = self.S[i] - self.q * self.gamma * self.S[i] ** 2 * (1 - np.exp(-self.sigma**2 * t))
            t += self.dt
        return reservation_price

    def calculate_spread(self) -> np.ndarray:
        """
        Calculates the spread at each time step.

        Returns:
            np.ndarray: Half-spread series (spread / 2).
        """
        t = 0
        spread = np.zeros(self.T)
        for i in range(self.T):
            # Spread increases with risk aversion and volatility
            spread[i] = self.gamma * self.S[i] ** 2 * (1 - np.exp(-self.sigma**2 * t)) + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
            t += self.dt
        return spread / 2  # Return half-spread to compute bid/ask

    def calculate_bid_ask(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes bid and ask prices using reservation price and half-spread.

        Returns:
            tuple[np.ndarray, np.ndarray]: Bid and ask price series.
        """
        reservation_price = self.calculate_reservation_price()
        spread = self.calculate_spread()
        bid = reservation_price - spread
        ask = reservation_price + spread
        return bid, ask

    
        
        
  
  
