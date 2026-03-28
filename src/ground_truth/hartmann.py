import torch


# ---------------------------------------------------------------------------
# Hartmann-6D constants
# ---------------------------------------------------------------------------

# Weights for the four Gaussian bells -- larger alpha = taller bell
ALPHA = torch.tensor([1.0, 1.2, 3.0, 3.2])

# A (4 x 6): controls the width of each bell along each dimension.
# Large A_ij means the bell is narrow in dimension j for component i.
A = torch.tensor([
    [10.0,  3.0,  17.0,  3.5,  1.7,  8.0],
    [ 0.05, 10.0,  17.0,  0.1,  8.0, 14.0],
    [ 3.0,   3.5,   1.7, 10.0, 17.0,  8.0],
    [17.0,   8.0,   0.05, 10.0,  0.1, 14.0],
])

# P (4 x 6): centres of the four Gaussian bells in [0, 1]^6.
# Given as integers in the literature, scaled by 1e-4 to get the actual centres.
P = 1e-4 * torch.tensor([
    [1312.0, 1696.0, 5569.0,  124.0, 8283.0, 5886.0],
    [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
    [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
    [4047.0, 8828.0, 8732.0, 5743.0, 1091.0,  381.0],
])

# Known global maximum for validation
GLOBAL_MAX_VALUE = 3.32237
GLOBAL_MAX_X = torch.tensor([0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.65731])


def hartmann(x: torch.Tensor, noisy: bool = False) -> torch.Tensor:
    """Evaluate the Hartmann-6D benchmark function.

    Standard BO benchmark with multiple local maxima and a known global maximum
    at f(x*) ≈ 3.322, used to validate the optimizer before applying it to the
    engineering problem.

    Input space: x ∈ [0, 1]^6  (no normalisation needed).

    Args:
        x: (N x 6) tensor of input points. Each row is one evaluation point.
        noisy: If True, adds small Gaussian noise (sigma=0.01) to the output.
            Kept small so convergence to the known optimum is still measurable.

    Returns:
        (N,) tensor of function values. Higher is better.
    """
    # x:     (N, 6)
    # P:     (4, 6)  ->  (1, 4, 6)  after unsqueeze
    # x[:, None, :]: (N, 1, 6)
    # diff:  (N, 4, 6)  -- distance from each point to each bell centre
    diff = x[:, None, :] - P[None, :, :]       # (N, 4, 6)

    # A[None, :, :]: (1, 4, 6)
    # inner sum: sum over dimensions j=1..6 of A_ij * (x_j - P_ij)^2
    # result: (N, 4)
    inner = (A[None, :, :] * diff ** 2).sum(dim=-1)   # (N, 4)

    # Weighted sum of Gaussian bells: sum_i alpha_i * exp(-inner_i)
    # ALPHA[None, :]: (1, 4)
    f = (ALPHA[None, :] * torch.exp(-inner)).sum(dim=-1)   # (N,)

    if noisy:
        f = f + torch.randn_like(f) * 0.01

    return f
