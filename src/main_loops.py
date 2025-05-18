import matplotlib.pyplot as plt
import pandas as pd

from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.simulations.geometric_browian import GeometricBrownianMotion

from src.strategies.avellaneda_stoikov_abm import AvellanedaStoikovStrategyAbm
from src.strategies.avellaneda_stoikov_gbm import AvellanedaStoikovStrategyGbm
from src.strategies.symmetric_strategy import SymmetricStrategy

from src.executions.poisson_execution_abm import PoissonExecutionAbm
from src.executions.poisson_execution_gbm import PoissonExecutionGbm
from src.core.simulation_runner import SimulationRunner
from src.core.inventory_manager import InventoryManager
from src.core.data_logger import DataLogger
from itertools import product


def run_param_grid(
    simulator,
    strategy_class,
    execution_class,
    strategy_name,
    market_name,
    steps,
    dt,
    gammas,
    ks,
    sigmas,
    n_simulations
):
    results = []

    for gamma, k, sigma in product(gammas, ks, sigmas):
        df = run_monte_carlo(
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
            n_simulations=n_simulations
        )

        mean_pnl = df["pnl"].mean()
        std_pnl = df["pnl"].std()
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0

        results.append({
            "strategy": f"{strategy_name} ({market_name})",
            "gamma": gamma,
            "k": k,
            "sigma": sigma,
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "sharpe": sharpe
        })

    return pd.DataFrame(results)

def run_strategy(
    simulator,
    strategy_class,
    execution_class,
    strategy_name,
    market_name,
    steps,
    dt,
    gamma,
    k,
    sigma,
    seed: int = 42
):
    T = steps * dt
    market = simulator(S0=100, sigma=sigma, seed=seed)
    prices = market.simulate(steps)

    # Instantiate the strategy
    strategy = strategy_class(gamma=gamma, sigma=sigma, k=k)

    # Instantiate the matching execution class
    execution = execution_class(A=100, k=k)

    # Set up simulation infrastructure
    inventory = InventoryManager(initial_cash=0, initial_inventory=0)
    logger = DataLogger()

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
    return df

def run_monte_carlo(
    simulator,
    strategy_class,
    execution_class,
    strategy_name,
    market_name,
    steps,
    dt,
    gamma,
    k,
    sigma,
    n_simulations
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
    n_simulations = 300  # reduce if slow

    gammas = [0.5, 1.0, 1.5, 2.0]
    ks = [0.5, 1.0, 1.5]
    sigmas = [0.1, 0.2, 0.3]

    from src.strategies.avellaneda_stoikov_gbm import AvellanedaStoikovStrategyGbm
    from src.executions.poisson_execution_gbm import PoissonExecutionGbm
    from src.simulations.geometric_browian import GeometricBrownianMotion

    df_grid = run_param_grid(
        simulator=GeometricBrownianMotion,
        strategy_class=AvellanedaStoikovStrategyGbm,
        execution_class=PoissonExecutionGbm,
        strategy_name="Avellaneda",
        market_name="GBM",
        steps=steps,
        dt=dt,
        gammas=gammas,
        ks=ks,
        sigmas=sigmas,
        n_simulations=n_simulations
    )

    print("\nGrid Search Results (sorted by Sharpe):")
    print(df_grid.sort_values(by="sharpe", ascending=False).head(10))

    # Optional: save results
    df_grid.to_csv("grid_search_results.csv", index=False)


if __name__ == "__main__":
    main()
