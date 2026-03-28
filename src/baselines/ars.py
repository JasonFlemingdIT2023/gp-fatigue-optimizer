from typing import Callable
import torch


#ARS hyperparameters (Mania et al., 2018)
STEP_SIZE = 0.02   #eta--> how far to move per update step
NOISE_STD = 0.03   #alpha--> perturbation size for finite-difference gradient
N_DIRS    = 4      #N--> random directions sampled per step (uses 2*N evaluations)
TOP_B     = 2      #b--> top-b directions kept for the update (b <= N)


def run_ars(
    ground_truth: Callable,
    bounds_low: torch.Tensor,
    bounds_high: torch.Tensor,
    n_init: int = 10,
    n_eval: int = 60,
    noisy_obs: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Augmented Random Search (Mania et al., 2018) baseline.

    Estimates the gradient of f by finite differences along random directions,
    without any surrogate model:

        delta_k ~ N(0, I)
        x <- x + (eta / (b * sigma_R)) * sum_{top b} [f(x+a*d_k) - f(x-a*d_k)] * d_k

    sigma_R= std of all 2*N observed rewards---> normalizes the update adaptively.

    Each ARS step uses 2 * N_DIRS evaluations.
    The first n_init evaluations are random, shared with other methods via seed.

    Args:
        ground_truth: f(x, noisy)--> (N,) tensor, x is (N, d) in physical units.
        bounds_low: (d,) lower bounds.
        bounds_high: (d,) upper bounds.
        n_init: Random initial observations shared with other methods.
        n_eval: Total function evaluation budget (including n_init).
        noisy_obs: If True, adds noise to evaluations.
        seed: Random seed for reproducibility.

    Returns:
        history: (n_eval,) tensor where history[k] = best value after k+1 evals.
    """
    torch.manual_seed(seed)

    dim = bounds_low.shape[0]

    # Initialisation--> same random points as other methods (same seed)
    X_norm = torch.rand(n_init, dim)
    X_phys = X_norm * (bounds_high - bounds_low) + bounds_low
    y_init = ground_truth(X_phys, noisy=noisy_obs) #(n_init,)

    history = y_init.clone().tolist()
    best_so_far = y_init.max().item()

    #Start ARS from the best initial point
    x = X_norm[y_init.argmax()].clone() # (d,) in [0, 1]^d

    #ARS update steps--> each step uses exactly 2 * N_DIRS evaluations
    remaining = n_eval - n_init

    while remaining >= 2 * N_DIRS:

        #Sample N_DIRS random directions from standard normal
        directions = torch.randn(N_DIRS, dim) #(N_DIRS, d)

        rewards_pos = torch.zeros(N_DIRS)
        rewards_neg = torch.zeros(N_DIRS)

        for k in range(N_DIRS):
            #Perturbate in both directions, clamp to stay in [0, 1]^d
            x_pos = (x + NOISE_STD * directions[k]).clamp(0.0, 1.0)
            x_neg = (x - NOISE_STD * directions[k]).clamp(0.0, 1.0)

            x_pos_phys = x_pos * (bounds_high - bounds_low) + bounds_low
            x_neg_phys = x_neg * (bounds_high - bounds_low) + bounds_low

            rewards_pos[k] = ground_truth(x_pos_phys.unsqueeze(0), noisy=noisy_obs)[0]
            rewards_neg[k] = ground_truth(x_neg_phys.unsqueeze(0), noisy=noisy_obs)[0]

            best_so_far = max(best_so_far, rewards_pos[k].item(), rewards_neg[k].item())
            #Two evaluations per direction-->append best_so_far twice
            history.append(best_so_far)
            history.append(best_so_far)

        remaining = remaining - 2 * N_DIRS

        #sigma_R--> std of all 2*N rewards-->normalizes step size
        all_rewards = torch.cat([rewards_pos, rewards_neg])
        sigma_r = all_rewards.std().clamp(min=1e-8)

        #Keep top-b directions ranked by |f(x+) - f(x-)|
        scores = (rewards_pos - rewards_neg).abs()
        top_indices = scores.topk(TOP_B).indices

        #Gradient estimate--> weighted sum of top-b directions
        grad_estimate = torch.zeros(dim)
        for k in top_indices:
            grad_estimate = grad_estimate + (rewards_pos[k] - rewards_neg[k]) * directions[k]

        #Update x and clamp to [0, 1]^d
        x = (x + (STEP_SIZE / (TOP_B * sigma_r)) * grad_estimate).clamp(0.0, 1.0)

    #Pad to exactly n_eval if remaining budget < 2*N_DIRS
    while len(history) < n_eval:
        history.append(best_so_far)

    history_tensor = torch.tensor(history[:n_eval])
    return torch.cummax(history_tensor, dim=0).values
