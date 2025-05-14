# main.py
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.strategies.avellaneda_stoikov_for_abm import AvellanedaStoikovStrategyAbm
from src.executions.poisson_execution_for_abm import PoissonExecutionForAbm
from src.executions.Assym_Poisson_Execution import AsymPoissonOrderExecution
from src.core.inventory_manager import InventoryManager
from src.core.data_logger import DataLogger
from src.core.simulation_runner import SimulationRunner

import matplotlib.pyplot as plt

def main():
    # Initialize components
    market = ArithmeticBrownianMotion(300,100,0.2,42)
    strategy = AvellanedaStoikovStrategyAbm(gamma=1.5, k=1, inventory=0,abm=market)
    #execution = PoissonExecutionForAbm(A=100,cash=0,strategy=strategy)
    execution = AsymPoissonOrderExecution(A_bid = 160.0, A_ask = 120.0, k_bid = 1.2, k_ask = 1.8, cash = 0.0, strategy=strategy)
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
