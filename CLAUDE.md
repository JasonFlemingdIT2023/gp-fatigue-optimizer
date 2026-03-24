# GP Fatigue Optimizer – Project Instructions

## Project Overview

This project implements Bayesian optimization of steel (S355) fatigue life using a
Gaussian Process (GP) surrogate model with a Matérn kernel and a Wolfe condition
line search as the inner loop termination criterion. It serves as a standalone
engineering application and as preparation for a Bachelor's thesis on optimizing
the GIBO algorithm via adaptive line search.

## Motivation

Fatigue life testing is expensive — each experiment takes hours to weeks. The GP
surrogate learns the fatigue life function from few observations and guides the
search for optimal operating conditions via line search on the GP posterior.

## Objective

Maximize log10 fatigue life over the 5-dimensional input space:

| Parameter         | Symbol    | Range        | Unit |
|-------------------|-----------|--------------|------|
| Stress amplitude  | sigma_a   | [100, 500]   | MPa  |
| Mean stress       | sigma_m   | [-200, 300]  | MPa  |
| Temperature       | T         | [20, 300]    | °C   |
| Stress ratio      | R         | [-1, 0.5]    | –    |
| Surface roughness | k_s       | [0.5, 1.0]   | –    |

## Tech Stack

- **Python 3.11**
- **PyTorch only** – no NumPy in core implementation. PyTorch is used for all
  tensor operations, Cholesky decomposition, and autograd for posterior gradients.
- **No GPyTorch, no botorch** – everything is implemented from scratch for
  learning purposes.
- **Matplotlib** for visualization.
- **Scipy** only allowed for hyperparameter optimization (L-BFGS).

## Project Structure
```
gp-fatigue-optimizer/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── src/
│   ├── ground_truth.py        # Synthetic fatigue life function (S355 steel)
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── matern.py          # Matérn kernel (nu = 0.5, 1.5, 2.5) in PyTorch
│   ├── gp/
│   │   ├── __init__.py
│   │   ├── cholesky.py        # Cholesky decomposition + triangular solvers from scratch
│   │   ├── gaussian_process.py  # GP posterior (fit, predict) using Cholesky
│   │   └── gradients.py       # Posterior gradients via PyTorch autograd
│   ├── linesearch/
│   │   ├── __init__.py
│   │   └── wolfe.py           # Wolfe condition line search on GP posterior
│   └── optimizer/
│       ├── __init__.py
│       └── gp_optimizer.py    # Main optimization loop
├── notebooks/
│   ├── 01_background.ipynb    # Fatigue life theory + Wöhler curve
│   ├── 02_gp_and_kernel.ipynb # GP theory + Matérn kernel
│   ├── 03_optimization.ipynb  # Line search + Wolfe conditions
│   └── 04_experiments.ipynb   # Results + comparison
└── experiments/
    └── results/
```

## Implementation Decisions

### Why Matérn and not RBF?
Real fatigue behavior is not infinitely smooth — the Matérn kernel allows
explicit control over smoothness via nu. We use nu = 2.5 (twice differentiable)
as default, which is the standard choice for physical problems.

### Why custom Cholesky?
GP inference requires solving (K + sigma_n^2 * I) @ alpha = y. Instead of
inverting the matrix directly (numerically unstable), we use Cholesky
decomposition (L @ L.T = K + sigma_n^2 * I) and solve two triangular systems.
This is O(n^3) for the factorization but O(n^2) for each subsequent solve.

### Why Wolfe conditions for line search?
GIBO uses a fixed hyperparameter M for inner loop termination. The thesis
contribution replaces M with Wolfe condition checks on the GP posterior —
terminating when sufficient decrease and curvature conditions are satisfied.
This project applies the same idea in the fatigue optimization context.

### Why PyTorch autograd for gradients?
Instead of deriving the analytical gradient of the GP posterior mean manually,
we use PyTorch autograd. This avoids error-prone manual derivation and makes
the code cleaner. The gradient flows through: kernel → Cholesky → posterior mean.

## Current Implementation Status

- [x] `src/ground_truth.py` — fatigue life function in PyTorch, tested
- [x] `src/kernels/matern.py` — Matérn kernel (nu = 0.5, 1.5, 2.5), tested
- [ ] `src/gp/cholesky.py` — naive Cholesky + triangular solvers, in progress
- [ ] `src/gp/gaussian_process.py` — GP fit + predict
- [ ] `src/gp/gradients.py` — posterior gradients via autograd
- [ ] `src/linesearch/wolfe.py` — Wolfe condition line search
- [ ] `src/optimizer/gp_optimizer.py` — main loop
- [ ] notebooks

## Learning Approach

**This is a learning project.** The user wants to understand every component
deeply before moving on. Follow these guidelines:

- Always explain the mathematics before showing code
- Show complete implementations, not partial snippets
- Go step by step — one function at a time
- Explain every line of code when introducing something new
- Ask for confirmation that the user understood before moving to the next step
- When the user tests code, wait for their output before continuing
- Prefer clarity over cleverness — no one-liners that sacrifice readability

## Coding Standards

- All code in **English** — variable names, comments, docstrings
- Type hints on all function signatures
- Docstrings in Google style (Args / Returns)
- No in-place PyTorch operations (+=, *=) — use explicit assignment to preserve autograd
- Constants in UPPER_SNAKE_CASE at module level
- Private helper methods prefixed with underscore

## Key Mathematical Concepts

- **GP Posterior**: conditioned joint Gaussian, gives mu(x*) and sigma^2(x*)
- **Matérn Kernel**: controls smoothness via nu, uses lengthscale and output variance
- **Cholesky**: A = L @ L.T, solve via forward/backward substitution
- **Log Marginal Likelihood**: used to train kernel hyperparameters
- **Wolfe Conditions**: sufficient decrease (Armijo) + curvature condition
- **Posterior Gradient**: nabla mu(x*) via autograd, used for line search direction