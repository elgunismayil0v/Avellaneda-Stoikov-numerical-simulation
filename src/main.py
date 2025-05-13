# main.py
from simulations.arithmetic_brownian import ArithmeticBrownianMotion
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from executions.poisson_execution import PoissonOrderExecution
from core.inventory_manager import InventoryManager
from core.data_logger import DataLogger
from core.simulation_runner import SimulationRunner

import matplotlib.pyplot as plt

def main():
    # Initialize components
    market = ArithmeticBrownianMotion(S0=100, sigma=2)
    strategy = AvellanedaStoikovStrategy(gamma=0.1, sigma=2, k=1.5)
    execution = PoissonOrderExecution(A=140, k=1.5)
    inventory = InventoryManager(initial_cash=0, initial_inventory=0)
    logger = DataLogger()

    # Configure simulation
    runner = SimulationRunner(
        market=market,
        pricing_strategy=strategy,
        order_execution=execution,
        inventory=inventory,
        logger=logger,
        dt=0.005,
        T=1.0
    )

    # Run and analyze
    runner.run()
    df = logger.get_dataframe()
    df.to_csv('simulation_results.csv', index=False)

    # Plot prices
    df.plot(y=['mid_prices', 'reservation_prices', 'bid_prices', 'ask_prices'])
    plt.title('Price Dynamics')
    plt.show()


if __name__ == "__main__":
    main()
