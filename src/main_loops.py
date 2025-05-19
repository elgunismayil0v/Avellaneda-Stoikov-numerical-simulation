import matplotlib.pyplot as plt
import pandas as pd

from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.strategies.avellaneda_stoikov_gbm import AvellanedaStoikovStrategyGbm
from src.strategies.symmetric_strategy import SymmetricStrategy
from src.executions.poisson_execution import PoissonOrderExecution
from src.core.inventory_manager import InventoryManager
from src.core.data_logger import DataLogger
from src.core.simulation_runner import SimulationRunner
from src.utils.simulation_helpers import run_strategy

import itertools

import itertools
import pandas as pd

def run_simulations(
    simulator,             # Class or function to simulate mid-price (e.g., GBM or ABM)
    strategy_class,        # Class that implements quoting strategy
    execution_class,       # Class that models execution logic
    strategy_name,         # Label for strategy (for plots)
    market_name,           # Label for the market model (ABM/GBM)
    sigma_values,          # List of volatilities
    gamma_values,          # List of risk aversion parameters
    k_values,              # List of market depth parameters
    steps_values,          # List of step counts
    dt_values,             # List of time increments
    seed: int = 42         # Random seed
):
    all_results = []

    for sigma, gamma, k, steps, dt in itertools.product(
        sigma_values, gamma_values, k_values, steps_values, dt_values
    ):
        df = run_strategy(
            simulator=simulator,
            strategy_class=strategy_class,
            execution_class=execution_class,
            strategy_name=strategy_name,
            market_name=market_name,
            steps=steps,
            dt=dt,
            gamma=gamma,
            k=k,
            sigma=sigma,
            seed=seed
        )
        df["sigma"] = sigma
        df["gamma"] = gamma
        df["k"] = k
        df["steps"] = steps
        df["dt"] = dt
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)



def main():
    sigma_values = [1, 2, 3]
    gamma_values = [0.05, 0.1, 0.2]
    k_values = [1.0, 1.5, 2.0]
    steps_values = [300, 400]
    dt_values = [0.005, 0.01]
    num_runs = 1000

    # Corrected: now includes steps_values and dt_values
    inv_df = run_simulations(
        AvellanedaStoikovStrategyGbm, 
        'Inventory',
        sigma_values, 
        gamma_values, 
        k_values, 
        steps_values,      
        dt_values,          
        num_runs            
    )

    sym_df = run_simulations(
        SymmetricStrategy, 
        'Symmetric',
        sigma_values, 
        gamma_values, 
        k_values, 
        steps_values,       
        dt_values,          
        num_runs            
    )

    results_df = pd.concat([inv_df, sym_df], ignore_index=True)
    results_df.to_csv('simulation_results.csv', index=False)
    print("Saved results to simulation_results.csv")

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
            plt.plot(subset['gamma'], subset['avg_profit'], marker='o',
                     label=f"σ={sigma}, k={k}")

    plt.xlabel('Gamma')
    plt.ylabel('Mean Profit')
    plt.title('Inventory Strategy: Mean Profit vs Gamma')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
