import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product

from src.strategies.avellaneda_stoikov_abm import AvellanedaStoikovStrategyAbm
from src.strategies.avellaneda_stoikov_gbm import AvellanedaStoikovStrategyGbm
from src.strategies.symmetric_strategy import SymmetricStrategy
from src.strategies.asymmetric_avellaneda import AsymmetricAvellanedaStoikovStrategy
from src.executions.poisson_execution_abm import PoissonExecutionAbm
from src.executions.poisson_execution_gbm import PoissonExecutionGbm
from src.executions.asym_poisson_execution import AsymPoissonOrderExecution
from src.simulations.arithmetic_brownian import ArithmeticBrownianMotion
from src.simulations.geometric_brownian import GeometricBrownianMotion

from src.utils.simulation_helpers import run_strategy, run_monte_carlo, plot_strategy_diagnostics

# -------------------------------
# Strategy Configurations (shared)
# -------------------------------
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
    },
    {
        "name": "Asymmetric",
        "market": "ABM",
        "simulator": ArithmeticBrownianMotion,
        "strategy_class": AsymmetricAvellanedaStoikovStrategy,
        "execution_class": AsymPoissonOrderExecution
    }
]

# -----------------------
# Original main() logic
# -----------------------
def main():
    steps = 300
    dt = 0.005
    gamma = 1.5
    k = 1.0
    sigma = 0.2
    n_simulations = 1000

    show_plots = True
    run_mc = False

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

        plt.figure(figsize=(10, 6))
        for label, group in df_all.groupby("strategy"):
            sns.kdeplot(group["pnl"], label=label, fill=False, linewidth=2.0, alpha=0.4, bw_adjust=0.5)
        plt.title("Final PnL Distribution (Monte Carlo)")
        plt.xlabel("PnL")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        summary = df_all.groupby("strategy")["pnl"].agg(["mean", "std"])
        summary["sharpe"] = summary["mean"] / summary["std"]
        print("\nSummary Statistics (Final PnL):")
        print(summary)

# ------------------------------------------
# Sensitivity analysis across 27 combinations
# ------------------------------------------
def sensitivity_analysis():
    sigma_values = [1, 2, 3]
    gamma_values = [0.05, 0.1, 0.5]
    k_values = [1.0, 1.5, 2.0]

    steps = 300
    dt = 0.005
    n_simulations = 500  # Adjust as needed

    all_results = []

    for sigma, gamma, k in product(sigma_values, gamma_values, k_values):
        print(f"\nRunning: sigma={sigma}, gamma={gamma}, k={k}")
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
            mean_pnl = df_mc["pnl"].mean()
            print(f"  Strategy={config['name']} ({config['market']}), Mean PnL={mean_pnl:.2f}")

            df_mc["sigma"] = sigma
            df_mc["gamma"] = gamma
            df_mc["k"] = k
            all_results.append(df_mc)

    df_sensitivity = pd.concat(all_results, ignore_index=True)
    df_sensitivity.to_csv("Sensitivity_Analysis_Results.csv", index=False)
    print("\nSaved sensitivity analysis results to 'Sensitivity_Analysis_Results.csv'.")

# -----------------------------------------
# Plot Mean Profit vs Gamma for (σ, k) pairs
# -----------------------------------------
def plot_mean_pnl_vs_gamma(csv_file="Sensitivity_Analysis_Results.csv", strategy_filter=None):
    df = pd.read_csv(csv_file)

    if strategy_filter:
        df = df[df["strategy"] == strategy_filter]

    if df.empty:
        print("No data found for the given strategy filter.")
        return

    summary = (
        df.groupby(["sigma", "gamma", "k"])["pnl"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    found = False
    for (sigma, k), group in summary.groupby(["sigma", "k"]):
        if not group.empty:
            label = f"σ={sigma}, k={k}"
            plt.plot(group["gamma"], group["pnl"], marker="o", label=label)
            found = True

    if not found:
        print("No valid data to plot.")
        return

    plt.title("Inventory Strategy: Mean Profit vs Gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Mean Profit")
    plt.legend(title="(σ, k)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Entry point: choose what to run
# -----------------------------
if __name__ == "__main__":
    # Uncomment one at a time

    # main()  # Single-run plots and/or original Monte Carlo

    sensitivity_analysis()  # Run all 27 combinations
    plot_mean_pnl_vs_gamma(strategy_filter="Symmetric (ABM)")  # Plot only for 'Symmetric' strategy (change if needed)
    plot_mean_pnl_vs_gamma(strategy_filter="Avellaneda (ABM)")  # Plot only for 'Avellaneda' strategy (change if needed)
    plot_mean_pnl_vs_gamma(strategy_filter="Avellaneda (GBM)")  # Plot only for 'Avellaneda' strategy (change if needed)
    plot_mean_pnl_vs_gamma(strategy_filter="Asymmetric (ABM)")  # Plot only for 'Avellaneda' strategy (change if needed)