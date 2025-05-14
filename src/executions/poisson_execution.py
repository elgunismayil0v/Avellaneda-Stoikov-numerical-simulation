from src.core.order_execution import OrderExecution
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.strategies.avellaneda_stoikov import AvellanedaStoikovStrategyAbm
import numpy as np

# executions/stateless_poisson_execution.py

import numpy as np
from src.core.order_execution import OrderExecution

class PoissonExecution(OrderExecution):
    def __init__(self, A: float, k: float):
        self.A = A  # intensity base
        self.k = k  # market depth coefficient

    def execute_orders(self, bid_price: float, ask_price: float, inventory: int, cash: float, dt: float) -> tuple[int, float]:
        """
        Executes orders at bid and ask prices based on Poisson arrivals.
        
        Args:
            bid_price (float): Bid quote
            ask_price (float): Ask quote
            inventory (int): Current inventory
            cash (float): Current cash
            dt (float): Time delta

        Returns:
            (new_inventory, new_cash)
        """
        # Calculate symmetric half-spread
        spread = (ask_price - bid_price) / 2

        # Arrival intensities
        lambda_bid = self.A * np.exp(-self.k * spread)
        lambda_ask = self.A * np.exp(-self.k * spread)

        # Simulate order arrivals
        if np.random.rand() < lambda_bid * dt:
            inventory += 1
            cash -= bid_price

        if np.random.rand() < lambda_ask * dt:
            inventory -= 1
            cash += ask_price

        return inventory, cash


    



