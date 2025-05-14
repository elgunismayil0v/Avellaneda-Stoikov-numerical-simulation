# core/simulation_runner.py

from src.core.data_logger import DataLogger
from src.core.inventory_manager import InventoryManager
from src.core.market_simulator import MarketSimulator
from src.core.order_execution import OrderExecution
from src.core.pricing_strategy import PricingStrategy


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

            # Strategy
            reservation_price = self.strategy.calculate_reservation_price(
                current_price, inventory_level, time_remaining
            )
            bid_spread, ask_spread = self.strategy.calculate_spread(
                current_price, inventory_level, time_remaining
            )
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

            # Logging
            self.logger.log('mid_prices', current_price)
            self.logger.log('reservation_prices', reservation_price)
            self.logger.log('bid_prices', bid_price)
            self.logger.log('ask_prices', ask_price)
            self.logger.log('inventory', self.inventory.inventory)
            self.logger.log('cash', self.inventory.cash)


