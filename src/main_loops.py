import itertools
import pandas as pd
import matplotlib.pyplot as plt

from simulations.arithmetic_brownian import ArithmeticBrownianMotion
from strategies.avellaneda_stoikov_for_gbm import AvellanedaStoikovGBM
from strategies.symmetric_strategy import SymmetricStategy
from executions.poisson_execution import PoissonOrderExecution
from core.inventory_manager import InventoryManager
from core.data_logger import DataLogger
from core.simulation_runner import SimulationRunner


def run_simulations(strategy_class, strategy_name, sigma_values, gamma_values, k_values, num_runs=1000):
    """
    Runs num_runs simulations for each combination of sigma, gamma, and k using the given strategy_class.
    Returns a DataFrame with summary metrics.
    """
    results = []
    # Fixed parameters per paper
    S0 = 100
    A = 140
    dt = 0.005
    T = 1.0

    for sigma, gamma, k in itertools.product(sigma_values, gamma_values, k_values):
        profits = []
        final_qs = []
        spreads = []
        for _ in range(num_runs):
            # Initialize components
            market = ArithmeticBrownianMotion(S0=S0, sigma=sigma)
            strategy = SymmetricStategy(gamma=gamma, sigma=sigma, k=k)
            execution = PoissonOrderExecution(A=A, k=k)
            inventory = InventoryManager(initial_cash=0, initial_inventory=0)
            logger = DataLogger()

            # Configure and run
            runner = SimulationRunner(
                market=market,
                pricing_strategy=strategy,
                order_execution=execution,
                inventory=inventory,
                logger=logger,
                dt=dt,
                T=T
            )
            runner.run()
            df = logger.get_dataframe()

            # Compute metrics
            mid_price = df['mid_prices'].iloc[-1]
            pnl = inventory.cash + inventory.inventory * mid_price
            profits.append(pnl)
            final_qs.append(inventory.inventory)
            spread = (df['ask_prices'] - df['bid_prices']).mean()
            spreads.append(spread)

        # summarize per parameter tuple
        results.append({
            'strategy': strategy_name,
            'sigma': sigma,
            'gamma': gamma,
            'k': k,
            'spread': sum(spreads)/len(spreads),
            'mean_profit': pd.Series(profits).mean(),
            'std_profit': pd.Series(profits).std(),
            'mean_q': pd.Series(final_qs).mean(),
            'std_q': pd.Series(final_qs).std()
        })

    return pd.DataFrame(results)


def main():
    # Parameter grids as requested
    sigma_values = [1, 2, 3]
    gamma_values = [0.05, 0.1, 0.2]
    k_values = [1.0, 1.5, 2.0]
    num_runs = 1000

    # Run inventory (Avellaneda-Stoikov) strategy
    inv_df = run_simulations(AvellanedaStoikovGBM, 'Inventory', sigma_values, gamma_values, k_values, num_runs)

    # Run symmetric strategy
    sym_df = run_simulations(SymmetricStategy, 'Symmetric', sigma_values, gamma_values, k_values, num_runs)

    # Combine results and save
    results_df = pd.concat([inv_df, sym_df], ignore_index=True)
    results_df.to_csv('simulation_results.csv', index=False)

    # Read back CSV
    df = pd.read_csv('simulation_results.csv')
    print("Loaded results from simulation_results.csv")

    # Example plotting: mean profit vs gamma for each (sigma, k) in inventory strategy
    plt.figure(figsize=(8, 5))
    for sigma in sigma_values:
        for k in k_values:
            subset = df[(df['strategy'] == 'Inventory') &
                        (df['sigma'] == sigma) &
                        (df['k'] == k)]
            plt.plot(subset['gamma'], subset['mean_profit'], marker='o',
                     label=f"σ={sigma}, k={k}")

    plt.xlabel('Gamma')
    plt.ylabel('Mean Profit')
    plt.title('Inventory Strategy: Mean Profit vs Gamma')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
