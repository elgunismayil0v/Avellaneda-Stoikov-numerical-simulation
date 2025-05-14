from src.core.order_execution import OrderExecution
from src.strategies.avellaneda_stoikov_for_gbm import AvellanedaStoikovGBM
from src.simulations.geometric_browian import GeometricBrownianMotion
import numpy as np

class PoissonExecutionForGbm(OrderExecution):
    """
    Executes market-making orders based on a Poisson process using the Avellaneda-Stoikov strategy
    with Geometric Brownian Motion (GBM) price simulation.

    Attributes:
        A (float): Baseline intensity of order arrivals.
        dt (float): Time step size.
        spread (np.ndarray): Spread at each time step.
        T (int): Number of simulation steps.
        inventory (int): Current inventory level.
        k (float): Market depth parameter.
        bid_price (np.ndarray): Bid prices over time.
        ask_price (np.ndarray): Ask prices over time.
        cash (float): Trader's cash position.
    """

    def __init__(self, A: float, cash: float, strategy: AvellanedaStoikovGBM):
        """
        Initialize the Poisson-based order execution engine.

        Args:
            A (float): Base order arrival rate.
            cash (float): Initial cash balance.
            strategy (AvellanedaStoikovGBM): Pricing strategy instance.
        """
        self.A = A
        self.dt = strategy.dt
        self.spread = strategy.calculate_spread()
        self.T = strategy.T
        self.inventory = strategy.q
        self.k = strategy.k
        self.bid_price, self.ask_price = strategy.calculate_bid_ask()
        self.cash = cash

    def execute_orders(self) -> tuple[float, float]:
        """
        Executes trades over the time horizon using Poisson-distributed order arrivals.

        Returns:
            tuple[float, float]: Final inventory and cash positions.
        """
        # Calculate execution intensities
        lambda_ask = self.A * np.exp(-self.k * self.spread)
        lambda_bid = self.A * np.exp(-self.k * self.spread)

        for i in range(self.T):
            # Simulate buy order fill (market sells to us at bid)
            if np.random.rand() < lambda_bid[i] * self.dt:
                self.cash -= self.bid_price[i]
                self.inventory += 1

            # Simulate sell order fill (market buys from us at ask)
            if np.random.rand() < lambda_ask[i] * self.dt:
                self.cash += self.ask_price[i]
                self.inventory -= 1

        return self.inventory, self.cash

 


        