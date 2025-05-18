import pandas as pd
from itertools import product

def run_sensitivity_analysis(
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
    n_simulations,
    run_strategy_fn,
    run_monte_carlo_fn
):
    results = []
    param_grid = list(product(gammas, ks, sigmas))

    for idx, (gamma, k, sigma) in enumerate(param_grid, start=1):
        print(f"Running ({idx}/{len(param_grid)}): gamma={gamma}, k={k}, sigma={sigma}")

        df_mc = run_monte_carlo_fn(
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

        summary = df_mc["pnl"].agg(["mean", "std"])
        summary["sharpe"] = summary["mean"] / summary["std"]
        summary["gamma"] = gamma
        summary["k"] = k
        summary["sigma"] = sigma
        summary["strategy"] = f"{strategy_name} ({market_name})"

        results.append(summary)

    return pd.DataFrame(results)
