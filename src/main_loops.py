import itertools
import pandas as pd
import matplotlib.pyplot as plt

from simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.strategies.avellaneda_stoikov_abm import AvellanedaStoikovStrategyABM
from src.executions.poisson_execution_abm import PoissonOrderExecution
from core.inventory_manager import InventoryManager
from core.data_logger import DataLogger
from core.simulation_runner import SimulationRunner

def main():
    # Define parameter grids
    sigma_values = [1, 2, 3]
    gamma_values = [0.05, 0.1, 0.2]
    k_values = [1.0, 1.5, 2.0]
    A_values = [100, 140, 180]

    # Initialize list to collect results
    results = []

    # Iterate over all combinations of parameters
    for sigma, gamma, k, A in itertools.product(sigma_values, gamma_values, k_values, A_values):
        # Initialize components with current parameters
        market = ArithmeticBrownianMotion(S0=100, sigma=sigma)
        strategy = AvellanedaStoikovStrategyABM(gamma=gamma, sigma=sigma, k=k)
        execution = PoissonOrderExecution(A=A, k=k)
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

        # Run simulation
        runner.run()
        df = logger.get_dataframe()

        # Compute summary statistics
        final_cash = inventory.cash
        final_inventory = inventory.inventory
        final_mid_price = df['mid_prices'].iloc[-1]
        final_pnl = final_cash + final_inventory * final_mid_price
        avg_spread = (df['ask_prices'] - df['bid_prices']).mean()
        trades_executed = len(logger.trades)

        # Append results
        results.append({
            'sigma': sigma,
            'gamma': gamma,
            'k': k,
            'A': A,
            'final_pnl': final_pnl,
            'avg_spread': avg_spread,
            'trades_executed': trades_executed
        })

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('simulation_results.csv', index=False)

    # Optional: Plotting for a specific parameter set
    subset = results_df[(results_df['sigma'] == 2) & (results_df['k'] == 1.5) & (results_df['A'] == 140)]
    plt.plot(subset['gamma'], subset['final_pnl'], marker='o')
    plt.xlabel('Gamma')
    plt.ylabel('Final PnL')
    plt.title('Sensitivity of PnL to Risk Aversion (gamma)')
    plt.show()

if __name__ == "__main__":
    main()
