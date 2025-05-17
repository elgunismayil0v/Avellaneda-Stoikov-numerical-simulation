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
        mid_prices = self.market.simulate()

        for i in range(self.steps):
            t = i * self.dt
            time_remaining = self.T - t

            # Calculate quotes
            reservation_price = self.strategy.calculate_reservation_price()
            spread = self.strategy.calculate_spread()
            bid_price, ask_price = self.strategy.calculate_bid_ask()

            # Execute orders
            new_inventory, new_cash = self.execution.execute_orders()
            self.inventory.update(new_inventory - self.inventory.inventory, new_cash - self.inventory.cash)

            wealth = self.inventory.cash + self.inventory.inventory*mid_prices[i]

            # Log data
            self.logger.log('mid_prices', mid_prices[i])
            self.logger.log('bid_prices', bid_price)
            self.logger.log('ask_prices', ask_price)
            self.logger.log('reservation_prices', reservation_price)
            self.logger.log('inventory', self.inventory.inventory)
            self.logger.log('cash', self.inventory.cash)
            self.logger.log('wealth', wealth)
