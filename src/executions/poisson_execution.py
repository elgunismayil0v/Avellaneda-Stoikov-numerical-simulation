# executions/poisson_execution.py
from core.order_execution import OrderExecution
import numpy as np
class PoissonOrderExecution(OrderExecution):
    def __init__(self, A: float, k: float):
        self.A = A
        self.k = k

    def execute_orders(self, bid_price: float, ask_price: float, inventory: int, cash: float, dt: float) -> tuple[
        int, float]:
        # Simplified Poisson execution logic
        lambda_bid = self.A * np.exp(-self.k * (ask_price - bid_price) / 2)
        lambda_ask = self.A * np.exp(-self.k * (ask_price - bid_price) / 2)

        if np.random.rand() < lambda_bid * dt:
            cash -= bid_price
            inventory += 1
        if np.random.rand() < lambda_ask * dt:
            cash += ask_price
            inventory -= 1

        return inventory, cash
