from src.core.order_execution import OrderExecution
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.strategies.avellaneda_stoikov_for_abm import AvellanedaStoikovStrategyAbm
import numpy as np

class PoissonExecutionForAbm(OrderExecution):
    """
    Executes orders based on a Poisson process, using prices from the Avellaneda-Stoikov strategy
    and a simplified market impact model.

    Attributes:
        A (float): Baseline intensity for order arrivals.
        dt (float): Time increment per step.
        spread (np.ndarray): Spread values at each time step.
        T (int): Total number of time steps.
        inventory (int): Current inventory.
        k (float): Market depth parameter.
        bid_price (np.ndarray): Bid prices from pricing strategy.
        ask_price (np.ndarray): Ask prices from pricing strategy.
        cash (float): Trader's cash position.
    """

    def __init__(self, A: float, cash: float, strategy: AvellanedaStoikovStrategyAbm):
        """
        Initializes the order execution logic.

        Args:
            A (float): Base intensity of order arrivals.
            cash (float): Initial cash position.
            strategy (AvellanedaStoikovStrategyAbm): Pricing strategy instance.
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
        Executes buy/sell orders using a Poisson process model.

        Returns:
            tuple[float, float]: Final inventory and cash positions.
        """
        # Calculate Poisson intensities based on spread
        lambda_bid = self.A * np.exp(-self.k * self.spread)
        lambda_ask = self.A * np.exp(-self.k * self.spread)

        for i in range(self.T):
            # Simulate bid order fill
            if np.random.rand() < lambda_bid[i] * self.dt:
                self.cash -= self.bid_price[i]
                self.inventory += 1

            # Simulate ask order fill
            if np.random.rand() < lambda_ask[i] * self.dt:
                self.cash += self.ask_price[i]
                self.inventory -= 1

        return self.inventory, self.cash

    



