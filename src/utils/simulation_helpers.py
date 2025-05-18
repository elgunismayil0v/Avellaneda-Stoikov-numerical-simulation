from src.core.simulation_runner import SimulationRunner
from src.core.inventory_manager import InventoryManager
from src.core.data_logger import DataLogger
import matplotlib.pyplot as plt
import pandas as pd


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
    

    return df

def plot_strategy_diagnostics(df, title="Strategy Behavior"):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Price and Quotes
    axs[0].plot(df["mid_prices"], label="Mid Price")
    axs[0].plot(df["reservation_prices"], label="Reservation Price")
    axs[0].plot(df["bid_prices"], label="Bid Price", linestyle="--")
    axs[0].plot(df["ask_prices"], label="Ask Price", linestyle="--")
    axs[0].set_ylabel("Price")
    axs[0].set_title(f"{title} – Quotes vs Price")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Inventory
    axs[1].plot(df["inventory"], label="Inventory", color="orange")
    axs[1].set_ylabel("Inventory")
    axs[1].set_title("Inventory Over Time")
    axs[1].legend()
    axs[1].grid(True)

    # 3. PnL and Cash
    axs[2].plot(df["pnl"], label="PnL", color="green")
    axs[2].plot(df["cash"], label="Cash", color="purple", linestyle="--")
    axs[2].set_ylabel("PnL / Cash")
    axs[2].set_xlabel("Time Step")
    axs[2].set_title("PnL and Cash Over Time")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

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
            seed=i  
        )
        final = df.iloc[-1]  # take last row (final pnl, inventory, etc.)
        all_results.append(final)

    return pd.DataFrame(all_results)