# executions/poisson_execution.py
from core.order_execution import OrderExecution
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
import numpy as np


class PoissonOrderExecution(OrderExecution):
    def __init__(self, A: float, cash: float, abm : ArithmeticBrownianMotion):
        self.A = A
        self.dt = abm.dt
        self.spread = abm.calculate_spread()
        self.T = abm.T
        self.inventory = abm.q
        self.k = abm.k
        self.bid_price, self.ask_price = abm.calculate_bid_ask()
        self.cash = cash

    def execute_orders(self) -> tuple[
        int, float]:
        # Simplified Poisson execution logic
        lambda_bid = self.A * np.exp(-self.k * self.spread)
        lambda_ask = self.A * np.exp(-self.k * self.spread)

        for i in range(self.T):
            if np.random.rand() < lambda_bid[i] * self.dt:
                self.cash -= self.bid_price[i]
                self.inventory += 1
        if np.random.rand() < lambda_ask[i] * self.dt:
            self.cash += self.ask_price[i]
            self.inventory -= 1


        return self.inventory, self.cash
