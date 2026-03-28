"""Comparison experiment: GP-WLS vs baselines on Hartmann-6D.

Runs four methods with N_SEEDS different random seeds and plots
the mean simple regret curve for each method:

    simple_regret(k) = f(x*) - best_value_after_k_evals

Lower is better. The plot shows mean ± 1 standard deviation.

Methods compared:
    - GP-WLS:      GP posterior mean gradient ascent + Wolfe line search
    - Vanilla BO:  Same GP, UCB acquisition, random candidate search
    - ARS:         Augmented Random Search (gradient-free finite differences)
    - Random:      Uniform random sampling (lower bound baseline)

Usage:
    cd gp-fatigue-optimizer
    python -m experiments.compare
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import matplotlib.pyplot as plt
import matplotlib

from ground_truth.hartmann import hartmann, GLOBAL_MAX_VALUE
from optimizer.gp_optimizer import run as run_gibo
from baselines.vanilla_bo import run_vanilla_bo
from baselines.ars import run_ars
from baselines.random_search import run_random_search


#Experiment hyperparameters
N_EVAL  = 60   #Total function evaluation budget per method
N_INIT  = 10   #Shared random initialisation evaluations
N_SEEDS = 20   #Number of independent random seeds

BOUNDS_LOW  = torch.zeros(6)
BOUNDS_HIGH = torch.ones(6)

#Hartmann lives in [0, 1]^6 already so bounds are trivially [0, 1]
#but we keep bounds explicit so the optimizer interface is consistent


def _to_regret(history: torch.Tensor) -> torch.Tensor:
    """Convert running-best history to simple regret.

    simple_regret[k] = f(x*) - history[k]

    Args:
        history: (n_eval,) running-best tensor.

    Returns:
        (n_eval,) simple regret tensor.
    """
    return GLOBAL_MAX_VALUE - history


def run_all_seeds() -> dict[str, torch.Tensor]:
    """Run all four methods across N_SEEDS seeds.

    Returns:
        Dict mapping method name to (N_SEEDS, N_EVAL) regret tensor.
    """
    #Pre allocate result storage--> one row per seed, one column per eval
    results = {
        "GP-WLS":       torch.zeros(N_SEEDS, N_EVAL),
        "Vanilla BO": torch.zeros(N_SEEDS, N_EVAL),
        "ARS":        torch.zeros(N_SEEDS, N_EVAL),
        "Random":     torch.zeros(N_SEEDS, N_EVAL),
    }

    for seed in range(N_SEEDS):
        print(f"Seed {seed + 1:02d}/{N_SEEDS}", flush=True)

        #--- GIBO ---
        #run() uses n_init=10 random points then n_iter=50 BO iterations
        #history has length n_init + n_iter = N_EVAL
        _, _, history_gibo = run_gibo(
            ground_truth=hartmann,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_init=N_INIT,
            n_iter=N_EVAL - N_INIT,
            n_restarts=3,
            noisy_obs=False,
            verbose=False,
            seed=seed,
        )
        results["GP-WLS"][seed] = _to_regret(history_gibo)

        #--- Vanilla BO ---
        history_vbo = run_vanilla_bo(
            ground_truth=hartmann,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_init=N_INIT,
            n_eval=N_EVAL,
            noisy_obs=False,
            seed=seed,
        )
        results["Vanilla BO"][seed] = _to_regret(history_vbo)

        #--- ARS ---
        history_ars = run_ars(
            ground_truth=hartmann,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_init=N_INIT,
            n_eval=N_EVAL,
            noisy_obs=False,
            seed=seed,
        )
        results["ARS"][seed] = _to_regret(history_ars)

        #--- Random Search ---
        history_rand = run_random_search(
            ground_truth=hartmann,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_eval=N_EVAL,
            noisy_obs=False,
            seed=seed,
        )
        results["Random"][seed] = _to_regret(history_rand)

    return results


def plot_regret(results: dict[str, torch.Tensor], save_path: str | None = None) -> None:
    """Plot mean ± std simple regret curves for all methods.

    Args:
        results: Dict mapping method name to (N_SEEDS, N_EVAL) regret tensor.
        save_path: If given, save the figure to this path instead of showing it.
    """
    #Color palette — each method gets a distinct color
    colors = {
        "GP-WLS":       "#1f77b4",  #blue
        "Vanilla BO": "#ff7f0e",  #orange
        "ARS":        "#2ca02c",  #green
        "Random":     "#d62728",  #red
    }

    x_axis = torch.arange(1, N_EVAL + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, regret_matrix in results.items():
        # regret_matrix: (N_SEEDS, N_EVAL)
        mean = regret_matrix.mean(dim=0)   # (N_EVAL,)
        std  = regret_matrix.std(dim=0)    # (N_EVAL,)

        ax.plot(x_axis.numpy(), mean.numpy(), label=name, color=colors[name], linewidth=2)
        ax.fill_between(
            x_axis.numpy(),
            (mean - std).numpy(),
            (mean + std).numpy(),
            alpha=0.2,
            color=colors[name],
        )

    ax.set_xlabel("Number of function evaluations", fontsize=12)
    ax.set_ylabel("Simple regret  f(x*) − best so far", fontsize=12)
    ax.set_title(
        f"Hartmann-6D: mean simple regret ± 1 std  ({N_SEEDS} seeds)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # log scale shows early differences more clearly

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    print(f"Running comparison: {N_SEEDS} seeds × 4 methods × {N_EVAL} evals each")
    print(f"Total function evaluations: {N_SEEDS * 4 * N_EVAL}\n")

    results = run_all_seeds()

    #Print final mean regret for each method
    print("\nFinal mean simple regret (lower is better):")
    for name, regret_matrix in results.items():
        final = regret_matrix[:, -1]  # regret at last evaluation
        print(f"  {name:<12}: {final.mean():.4f}  ±  {final.std():.4f}")

    save_path = os.path.join(
        os.path.dirname(__file__), "results", "hartmann_regret.png"
    )
    plot_regret(results, save_path=save_path)
