# src/executions/poisson_execution.py

from src.core.order_execution import OrderExecution
import numpy as np

class AsymPoissonOrderExecution(OrderExecution):
    """
    Executes a single step of order execution using asymmetric Poisson processes
    for bid and ask sides, based on quoted prices and spread-dependent intensity.

    Attributes:
        A_bid (float): Base intensity for market-sell orders (hit our bid).
        A_ask (float): Base intensity for market-buy orders (lift our ask).
        k_bid (float): Decay factor for bid-side intensity.
        k_ask (float): Decay factor for ask-side intensity.
    """

    def __init__(
        self,
        A_bid: float,
        A_ask: float,
        k_bid: float,
        k_ask: float,
        cash: float,  # not used inside the class, but kept for compatibility
        strategy=None  # optional: for future extension
    ):
        self.A_bid = A_bid
        self.A_ask = A_ask
        self.k_bid = k_bid
        self.k_ask = k_ask
        self.strategy = strategy  # optional reference
        self.cash = cash  # unused, can be removed if not needed

    def execute_orders(
        self,
        bid_price: float,
        ask_price: float,
        inventory: int,
        cash: float,
        dt: float
    ) -> tuple[int, float]:
        """
        Executes buy/sell orders using asymmetric Poisson arrival intensities.

        Args:
            bid_price (float): Current bid quote.
            ask_price (float): Current ask quote.
            inventory (int): Current inventory level.
            cash (float): Current cash position.
            dt (float): Time increment (e.g., 0.005)

        Returns:
            (updated_inventory, updated_cash)
        """
        # Compute bid/ask spread (as a proxy for execution distance)
        bid_spread = abs(bid_price)  # Adjust logic if needed
        ask_spread = abs(ask_price)

        # Compute fill intensities
        lambda_bid = self.A_bid * np.exp(-self.k_bid * bid_spread)
        lambda_ask = self.A_ask * np.exp(-self.k_ask * ask_spread)

        # Simulate bid-side execution
        if np.random.rand() < lambda_bid * dt:
            cash -= bid_price
            inventory += 1

        # Simulate ask-side execution
        if np.random.rand() < lambda_ask * dt:
            cash += ask_price
            inventory -= 1

        return inventory, cash
