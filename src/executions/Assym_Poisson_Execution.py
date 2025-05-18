# src/executions/poisson_execution.py

from src.core.order_execution import OrderExecution
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
import numpy as np
from src.strategies.avellaneda_stoikov_abm import AvellanedaStoikovStrategyAbm

class AsymPoissonOrderExecution(OrderExecution):
    """
    Executes orders based on independent Poisson processes for bid vs ask,
    using prices from the Avellaneda-Stoikov strategy.

    Attributes:
        A_bid (float): Base intensity for sells hitting our bid.
        A_ask (float): Base intensity for buys lifting our ask.
        dt (float): Time increment per step.
        spread (np.ndarray): Half‐spread values at each time step.
        T (int): Total number of time steps.
        inventory (int): Current inventory.
        k_bid (float): Decay rate for bid intensity.
        k_ask (float): Decay rate for ask intensity.
        bid_price (np.ndarray): Bid quotes from strategy.
        ask_price (np.ndarray): Ask quotes from strategy.
        cash (float): Trader's cash position.
    """

    def __init__(
        self,
        A_bid: float,
        A_ask: float,
        k_bid: float,
        k_ask: float,
        cash: float,
        strategy: AvellanedaStoikovStrategyAbm
    ):
        """
        Initializes the order execution logic with asymmetric parameters.

        Args:
            A_bid (float): Base intensity of market‐sell arrivals (hitting bid).
            A_ask (float): Base intensity of market‐buy arrivals (lifting ask).
            k_bid (float): Decay rate for bid execution intensity λ_b(δ)=A_bid e^(−k_bid δ).
            k_ask (float): Decay rate for ask execution intensity λ_a(δ)=A_ask e^(−k_ask δ).
            cash (float): Initial cash position.
            strategy (AvellanedaStoikovStrategyAbm): Pricing strategy instance.
        """
        # store asymmetric parameters
        self.A_bid = A_bid
        self.A_ask = A_ask
        self.k_bid = k_bid
        self.k_ask = k_ask

        # pull timing and quotes from the strategy
        self.dt = strategy.dt
        self.spread = strategy.calculate_spread()
        self.T = strategy.T
        self.inventory = strategy.q
        self.bid_price, self.ask_price = strategy.calculate_bid_ask()

        self.cash = cash

    def execute_orders(self) -> tuple[float, float]:
        """
        Executes buy/sell orders via two independent Poisson processes.

        Returns:
            (inventory, cash)
        """
        for i in range(self.T):
            # asymmetric Poisson intensities
            lambda_bid = self.A_bid * np.exp(-self.k_bid * self.spread[i])
            lambda_ask = self.A_ask * np.exp(-self.k_ask * self.spread[i])

            # bid‐side fill
            if np.random.rand() < lambda_bid * self.dt:
                self.cash -= self.bid_price[i]
                self.inventory += 1

            # ask‐side fill
            if np.random.rand() < lambda_ask * self.dt:
                self.cash += self.ask_price[i]
                self.inventory -= 1

        return self.inventory, self.cash
