import matplotlib.pyplot as plt
import pandas as pd

from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.simulations.geometric_browian import GeometricBrownianMotion
from src.core.simulation_runner import SimulationRunner
from src.core.inventory_manager import InventoryManager
from src.core.data_logger import DataLogger




def run_strategy(
    simulator, # Class or function to simulate mid-price (ABM or GBM)
    strategy_class, # Class that implements the quoting strategy
    execution_class, # Class that models execution probability (Poisson-based)
    strategy_name, # Label for strategy (used for logging or plotting)
    market_name, # Label for market process (ABM or GBM)
    steps,  # Number of discrete time steps in the simulation
    dt, # Time increment (Δt)
    gamma, # Risk aversion parameter
    k, # Market depth (for execution intensity λ(δ))
    sigma,  # Volatility of the mid-price process
    seed: int = 42
):
    T = steps * dt 
    # Initialize the mid-price process (ABM or GBM) and simulate a price path over steps
    market = simulator(S0=100, sigma=sigma, seed=seed)
    prices = market.simulate(steps)
    
    # Instantiate the quoting strategy (e.g., Avellaneda-Stoikov), passing in model parameters.
    strategy = strategy_class(gamma=gamma, sigma=sigma, k=k)

    # Create the Poisson-based trade execution model with a base intensity A and decay rate k
    execution = execution_class(A=100, k=k)

    # Start with zero inventory and zero cash. Prepare the logging system to track simulation data.
    inventory = InventoryManager(initial_cash=0, initial_inventory=0)
    logger = DataLogger()

    # Initialize the simulation engine and run it. 
    # This simulates quote placements, executions, and inventory updates over time.
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
    df["pnl"] = df["cash"] + df["inventory"] * df["mid_prices"]
    df["strategy"] = f"{strategy_name} ({market_name})"
    df.plot(y=['mid_prices', 'reservation_prices', 'bid_prices', 'ask_prices'])
    plt.title('Price Dynamics')
    plt.show()
    return df

def run_monte_carlo(
    simulator, # Class or function to simulate mid-price (ABM or GBM)
    strategy_class, # Class that implements the quoting strategy
    execution_class, # Class that models execution probability (Poisson-based)
    strategy_name, # Label for strategy (used for logging or plotting)
    market_name, # Label for market process (ABM or GBM)
    steps,  # Number of discrete time steps in the simulation
    dt, # Time increment (Δt)
    gamma, # Risk aversion parameter
    k, # Market depth (for execution intensity λ(δ))
    sigma,  # Volatility of the mid-price process
    n_simulations # Number of Monte Carlo simulations to run
):
    all_results = []

    for i in range(n_simulations):
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
            seed=i  # use different seed
        )
        final = df.iloc[-1]  # take last row (final pnl, inventory, etc.)
        all_results.append(final)

    return pd.DataFrame(all_results)

def main():
    steps = 300
    dt = 0.005
    gamma = 1.5
    k = 1.0
    sigma = 0.2
    n_simulations = 1

    from src.strategies.avellaneda_stoikov_abm import AvellanedaStoikovStrategyAbm
    from src.strategies.avellaneda_stoikov_gbm import AvellanedaStoikovStrategyGbm
    from src.strategies.symmetric_strategy import SymmetricStrategy
    from src.executions.poisson_execution_abm import PoissonExecutionAbm
    from src.executions.poisson_execution_gbm import PoissonExecutionGbm

    dfs = []

    # Avellaneda ABM
    dfs.append(run_monte_carlo(
        ArithmeticBrownianMotion, AvellanedaStoikovStrategyAbm, PoissonExecutionAbm,
        "Avellaneda", "ABM", steps, dt, gamma, k, sigma, n_simulations
    ))

    # Avellaneda GBM
    dfs.append(run_monte_carlo(
        GeometricBrownianMotion, AvellanedaStoikovStrategyGbm, PoissonExecutionGbm,
        "Avellaneda", "GBM", steps, dt, gamma, k, sigma, n_simulations
    ))

    # Symmetric ABM
    dfs.append(run_monte_carlo(
        ArithmeticBrownianMotion, SymmetricStrategy, PoissonExecutionAbm,
        "Symmetric", "ABM", steps, dt, gamma, k, sigma, n_simulations
    ))

    df_all = pd.concat(dfs, ignore_index=True)
    
    # Plot histogram of final PnLs
    plt.figure(figsize=(10, 6))
    for label, group in df_all.groupby("strategy"):
        plt.hist(group["pnl"], alpha=0.6, label=label)
    plt.title("Final PnL Distribution (1000 simulations)")
    plt.xlabel("PnL")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(True)

    # Print summary statistics
    summary = df_all.groupby("strategy")["pnl"].agg(["mean", "std"])
    summary["sharpe"] = summary["mean"] / summary["std"]
    print("\nSummary Statistics (Final PnL):")
    print(summary)

    plt.show()

    


if __name__ == "__main__":
    main()
