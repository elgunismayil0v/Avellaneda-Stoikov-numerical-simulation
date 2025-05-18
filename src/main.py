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
from src.utils.simulation_helpers import run_strategy, run_monte_carlo, plot_strategy_diagnostics


def main():
    # Simulation parameters
    steps = 300
    dt = 0.005
    gamma = 1.5
    k = 1.0
    sigma = 0.2
    n_simulations = 1000

    # Flags
    show_plots = True
    run_mc = False

    # Strategy configurations
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

    # Visualize one run per strategy
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

    # Run Monte Carlo for final PnL distribution
    if run_mc:
        mc_results = []
        for config in strategy_configs:
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

        df_all = pd.concat(mc_results, ignore_index=True)
        df_all.to_csv("Monte_Carlo_Simulation.csv")
        # Histogram of final PnLs
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


        # Summary stats
        summary = df_all.groupby("strategy")["pnl"].agg(["mean", "std"])
        summary["sharpe"] = summary["mean"] / summary["std"]
        print("\nSummary Statistics (Final PnL):")
        print(summary)


if __name__ == "__main__":
    main()
