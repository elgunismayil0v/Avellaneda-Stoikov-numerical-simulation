import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.strategies.avellaneda_stoikov_abm import AvellanedaStoikovStrategyAbm
from src.strategies.avellaneda_stoikov_gbm import AvellanedaStoikovStrategyGbm
from src.strategies.symmetric_strategy import SymmetricStrategy
from src.executions.poisson_execution_abm import PoissonExecutionAbm
from src.executions.poisson_execution_gbm import PoissonExecutionGbm
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.simulations.geometric_brownian import GeometricBrownianMotion
# Import helper functions (single run, monte carlo, plotting)
from src.utils.simulation_helpers import run_strategy, run_monte_carlo, plot_strategy_diagnostics


def main():
    # Simulation parameters
    steps = 300            # Number of time steps per simulation
    dt = 0.005             # Time increment (Δt)
    gamma = 1.5            # Risk aversion coefficient
    k = 1.0                # Market depth parameter
    sigma = 0.2            # Volatility of the mid-price process
    n_simulations = 1000   # Number of Monte Carlo simulation runs

    # Flags
    show_plots = True   # Whether to show detailed single-run plots
    run_mc = False      # Whether to run Monte Carlo simulation

    # Define different strategies and market models to evaluate
    strategy_configs = [
        {
            "name": "Avellaneda",
            "market": "ABM",
            "simulator": ArithmeticBrownianMotion,
            "strategy_class": AvellanedaStoikovStrategyAbm,
            "execution_class": PoissonExecutionAbm
        },
        {
            "name": "Avellaneda",
            "market": "GBM",
            "simulator": GeometricBrownianMotion,
            "strategy_class": AvellanedaStoikovStrategyGbm,
            "execution_class": PoissonExecutionGbm
        },
        {
            "name": "Symmetric",
            "market": "ABM",
            "simulator": ArithmeticBrownianMotion,
            "strategy_class": SymmetricStrategy,
            "execution_class": PoissonExecutionAbm
        }
    ]

    # ----------------------
    # SINGLE-RUN SIMULATION
    # ----------------------
    if show_plots:
        for config in strategy_configs:
            df = run_strategy(
                simulator=config["simulator"],
                strategy_class=config["strategy_class"],
                execution_class=config["execution_class"],
                strategy_name=config["name"],
                market_name=config["market"],
                steps=steps,
                dt=dt,
                gamma=gamma,
                k=k,
                sigma=sigma,
                seed=0
            )
            plot_title = f"{config['name']} ({config['market']})"
            plot_strategy_diagnostics(df, plot_title)

    # ----------------------
    # MONTE CARLO SIMULATION
    # ----------------------
    if run_mc:
        mc_results = []
        for config in strategy_configs:
            # Run N simulations and collect final PnL statistics
            df_mc = run_monte_carlo(
                simulator=config["simulator"],
                strategy_class=config["strategy_class"],
                execution_class=config["execution_class"],
                strategy_name=config["name"],
                market_name=config["market"],
                steps=steps,
                dt=dt,
                gamma=gamma,
                k=k,
                sigma=sigma,
                n_simulations=n_simulations
            )
            mc_results.append(df_mc)
        # Combine results from all strategy configurations
        df_all = pd.concat(mc_results, ignore_index=True)
        # Save the full simulation results (optional)
        df_all.to_csv("Monte_Carlo_Simulation.csv")
        # Plot density curves of final PnLs (one per strategy)
        plt.figure(figsize=(10, 6))
        for label, group in df_all.groupby("strategy"):
            sns.kdeplot(
                group["pnl"],
                label=label,
                fill=False,
                linewidth=2.0,
                alpha=0.4,
                bw_adjust=0.5
            )
        plt.title("Final PnL Distribution (Monte Carlo)")
        plt.xlabel("PnL")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        # Print summary statistics (mean, std, Sharpe ratio)
        summary = df_all.groupby("strategy")["pnl"].agg(["mean", "std"])
        summary["sharpe"] = summary["mean"] / summary["std"]
        print("\nSummary Statistics (Final PnL):")
        print(summary)

# Entry point of the script
if __name__ == "__main__":
    main()
