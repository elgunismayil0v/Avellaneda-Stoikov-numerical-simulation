# core/simulation_runner.py

from core.data_logger import DataLogger
from core.inventory_manager import InventoryManager
from core.market_simulator import MarketSimulator
from core.order_execution import OrderExecution
from core.pricing_strategy import PricingStrategy
import matplotlib.pyplot as plt

# core/simulation_runner.py

class SimulationRunner:
    def __init__(self, market, pricing_strategy, order_execution, inventory, logger, dt: float, T: float):
        self.market = market
        self.strategy = pricing_strategy
        self.execution = order_execution
        self.inventory = inventory
        self.logger = logger
        self.dt = dt
        self.T = T
        self.steps = int(T / dt)

    def run(self):


        mid_prices = self.market.simulate(self.steps)


        for i in range(self.steps):
            t = i * self.dt
            time_remaining = self.T - t
            current_price = mid_prices[i]
            inventory_level = self.inventory.inventory
            cash = self.inventory.cash


            # Get pricing from strategy
            reservation_price = self.strategy.calculate_reservation_price(
                current_price=mid_prices[i],
                inventory=self.inventory.inventory,
                time_remaining=time_remaining
            )

            bid_spread, ask_spread = self.strategy.calculate_spread(
                current_price=mid_prices[i],
                inventory=self.inventory.inventory,
                time_remaining=time_remaining
            )

            # Calculate absolute prices

            bid_price = reservation_price - bid_spread
            ask_price = reservation_price + ask_spread

            # Execution
            new_inventory, new_cash = self.execution.execute_orders(
                bid_price=bid_price,
                ask_price=ask_price,
                inventory=inventory_level,
                cash=cash,
                dt=self.dt
            )
            self.inventory.update(new_inventory - inventory_level, new_cash - cash)


            

            wealth = self.inventory.cash + self.inventory.inventory*mid_prices[i]

            # Log data
            self.logger.log('mid_prices', mid_prices[i])
            self.logger.log('reservation_prices', reservation_price)

            self.logger.log('bid_prices', bid_price)
            self.logger.log('ask_prices', ask_price)
            self.logger.log('inventory', self.inventory.inventory)
            self.logger.log('cash', self.inventory.cash)
            self.logger.log('wealth', wealth)

