from src.core.order_execution import OrderExecution
from src.strategies.avellaneda_stoikov_gbm import AvellanedaStoikovGBM
from src.simulations.geometric_browian import GeometricBrowianMotion
import numpy as np

class PoissonExecutionForPaths(OrderExecution):
    def __init__(self, A: float, cash: float, gbm : AvellanedaStoikovGBM):
        self.A = A
        self.dt = gbm.dt
        self.spread = gbm.calculate_spread()
        self.T = gbm.T
        self.inventory = gbm.q
        self.k = gbm.k
        self.bid_price, self.ask_price = gbm.calculate_bid_ask()
        self.cash = cash
        
    def execute_orders(self):
        lambda_ask = self.A * np.exp(-self.k * self.spread)
        lambda_bid = self.A * np.exp(-self.k * self.spread)
        for i in range(self.T):
            if np.random.rand() < lambda_bid[i] * self.dt:
                self.cash -= self.bid_price[i]
                self.inventory += 1
        if np.random.rand() < lambda_ask[i] * self.dt:
            self.cash += self.ask_price[i]
            self.inventory -= 1


        return self.inventory, self.cash
 
x = GeometricBrowianMotion(5,3,0.2,42)
y = AvellanedaStoikovGBM(1,1,1,x)
c = PoissonExecutionForPaths(120, 120,y)
print(c.execute_orders())

        