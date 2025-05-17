import numpy as np
from src.core.order_execution import OrderExecution

class PoissonExecutionForAbm(OrderExecution):
    def __init__(self, A: float, k: float):
        self.A = A
        self.k = k

    def execute_orders(
        self,
        bid_price: float,
        ask_price: float,
        inventory: int,
        cash: float,
        dt: float
    ) -> tuple[int, float]:
        spread = ask_price - bid_price
        lambda_bid = self.A * np.exp(-self.k * spread)
        lambda_ask = self.A * np.exp(-self.k * spread)

        if np.random.rand() < lambda_bid * dt:
            cash -= bid_price
            inventory += 1

        if np.random.rand() < lambda_ask * dt:
            cash += ask_price
            inventory -= 1

        return inventory, cash
