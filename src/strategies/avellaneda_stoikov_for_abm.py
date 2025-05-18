# strategies/avellaneda_stoikov.py

from core.pricing_strategy import PricingStrategy
from simulations.arithmetic_brownian import ArithmeticBrownianMotion
import numpy as np

class AvellanedaStoikovStrategyAbm(PricingStrategy):
    """
    Avellaneda-Stoikov market-making strategy using Arithmetic Brownian Motion (ABM) for price simulation.

    This strategy sets bid and ask prices dynamically based on inventory levels and simulated market volatility,
    adapting the original Avellaneda-Stoikov framework to an ABM price process.

    Attributes:
        gamma (float): Risk aversion coefficient.
        k (float): Market depth parameter.
        q (int): Current inventory.
        sigma (float): Volatility from ABM.
        T (int): Number of time steps.
        dt (float): Time increment per step.
        S (np.ndarray): Simulated asset price path.
    """

    def __init__(self, gamma: float, k: float, inventory: int, abm: ArithmeticBrownianMotion):
        """
        Initializes the strategy with risk preferences, inventory, and a simulated ABM path.

        Args:
            gamma (float): Trader's risk aversion.
            k (float): Market liquidity depth.
            inventory (int): Current inventory position.
            abm (ArithmeticBrownianMotion): ABM simulator instance.
        """
        self.gamma = gamma
        self.k = k
        self.q = inventory
        self.sigma = abm.sigma
        self.T = abm.NoOfStep
        self.dt = 1 / self.T
        self.S = abm.simulate()

    def calculate_reservation_price(self) -> np.ndarray:
        """
        Calculates the reservation price at each time step.

        Returns:
            np.ndarray: Series of reservation prices.
        """
        t = 0
        reservation_price = np.zeros(self.T)
        for i in range(self.T):
            # Reservation price adjusted by inventory risk over time
            reservation_price[i] = self.S[i] - self.q * self.gamma * (self.sigma ** 2) * t
            t += self.dt
        return reservation_price

    def calculate_spread(self) -> np.ndarray:
        """
        Calculates the half-spread at each time step.

        Returns:
            np.ndarray: Series of half-spreads.
        """
        t = 0
        spread = np.zeros(self.T)
        for i in range(self.T):
            # Spread is based on risk aversion, time, and market depth
            spread[i] = self.gamma * (self.sigma ** 2) * t + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
            t += self.dt
        return spread / 2  # Return half-spread

    def calculate_bid_ask(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes bid and ask prices at each time step.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays of bid and ask prices.
        """
        reservation_price = self.calculate_reservation_price()
        spread = self.calculate_spread()
        bid = reservation_price - spread
        ask = reservation_price + spread
        return bid, ask


