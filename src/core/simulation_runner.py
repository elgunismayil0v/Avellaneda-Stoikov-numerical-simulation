# core/simulation_runner.py

from src.core.data_logger import DataLogger
from src.core.inventory_manager import InventoryManager
from src.core.market_simulator import MarketSimulator
from src.core.order_execution import OrderExecution
from src.core.pricing_strategy import PricingStrategy


class SimulationRunner:
    def __init__(
            self,
            market: MarketSimulator,
            pricing_strategy: PricingStrategy,
            order_execution: OrderExecution,
            inventory: InventoryManager,
            logger: DataLogger,
            dt: float,
            T: float
    ):
        self.market = market
        self.strategy = pricing_strategy
        self.execution = order_execution
        self.inventory = inventory
        self.logger = logger
        self.dt = dt
        self.T = T
        self.steps = int(T / dt)

    def run(self):
        mid_prices = self.market.simulate(self.steps, self.dt)

        for i in range(self.steps):
            t = i * self.dt
            time_remaining = self.T - t

            # Calculate quotes
            reservation_price = self.strategy.calculate_reservation_price(
                mid_prices[i], self.inventory.inventory, time_remaining
            )
            bid_spread, ask_spread = self.strategy.calculate_spread(
                mid_prices[i], self.inventory.inventory, time_remaining
            )
            bid_price = reservation_price - bid_spread
            ask_price = reservation_price + ask_spread

            # Execute orders
            new_inventory, new_cash = self.execution.execute_orders(
                bid_price, ask_price, self.inventory.inventory, self.inventory.cash, self.dt
            )
            self.inventory.update(new_inventory - self.inventory.inventory, new_cash - self.inventory.cash)

            # Log data
            self.logger.log('mid_prices', mid_prices[i])
            self.logger.log('bid_prices', bid_price)
            self.logger.log('ask_prices', ask_price)
            self.logger.log('reservation_prices', reservation_price)
            self.logger.log('inventory', self.inventory.inventory)
            self.logger.log('cash', self.inventory.cash)
