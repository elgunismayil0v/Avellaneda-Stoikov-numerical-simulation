import matplotlib.pyplot as plt
import pandas as pd

from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.simulations.geometric_browian import GeometricBrownianMotion

from src.strategies.avellaneda_stoikov_abm import AvellanedaStoikovStrategyABM
from src.strategies.symmetric_strategy import SymmetricStrategy

from src.executions.poisson_execution_abm import PoissonExecution
from src.core.simulation_runner import SimulationRunner
from src.core.inventory_manager import InventoryManager
from src.core.data_logger import DataLogger


def run_strategy(simulator, strategy_class, strategy_name, market_name, steps, dt, gamma, k, sigma):
    T = steps * dt
    market = simulator(S0=100, sigma=sigma, seed=42)

    # Strategy: Avellaneda or Symmetric

    strategy = strategy_class(gamma=gamma, sigma=sigma, k=k)


    execution = PoissonExecution(A=100, k=k)
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


def main():
    steps = 300
    dt = 0.005
    gamma = 1.5
    k = 1.0
    sigma = 0.2

    dfs = []

    # Run 3 configs
    dfs.append(run_strategy(ArithmeticBrownianMotion, AvellanedaStoikovStrategy, "Avellaneda", "ABM", steps, dt, gamma, k, sigma))
    dfs.append(run_strategy(GeometricBrownianMotion, AvellanedaStoikovStrategy, "Avellaneda", "GBM", steps, dt, gamma, k, sigma))
    dfs.append(run_strategy(ArithmeticBrownianMotion, SymmetricStrategy, "Symmetric", "ABM", steps, dt, gamma, k, sigma))

    df_all = pd.concat(dfs, ignore_index=True)

    # Plot inventory
    plt.figure()
    for label, group in df_all.groupby("strategy"):
        plt.plot(group["inventory"].values, label=label)
    plt.title("Inventory Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Inventory")
    plt.legend()
    plt.grid(True)

    # Plot PnL
    plt.figure()
    for label, group in df_all.groupby("strategy"):
        plt.plot(group["pnl"].values, label=label)
    plt.title("PnL Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("PnL")
    plt.legend()
    plt.grid(True)

    # Plot Mid Prices
    plt.figure()
    for label, group in df_all.groupby("strategy"):
        plt.plot(group["mid_prices"].values, label=label)
    plt.title("Mid Price Paths")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
