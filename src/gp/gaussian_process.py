import torch

from .cholesky import cholesky, solve_cholesky


class GaussianProcess:
    """Gaussian Process surrogate model with custom Cholesky-based inference.

    Args:
        kernel: Callable kernel object, e.g. MaternKernel. Must accept two
            tensors (n x d) and (m x d) and return an (n x m) matrix.
        noise_var: Observation noise variance sigma_n^2. Added to the diagonal
            of the training kernel matrix for numerical stability and to model
            measurement noise.
    """

    def __init__(self, kernel, noise_var: float = 1e-4) -> None:
        self.kernel = kernel
        # noise_var also stored in log-space so it stays positive during optimization
        self.log_noise_var = torch.tensor(
            torch.log(torch.tensor(noise_var)).item(), requires_grad=True
        )

        # Set during fit()
        self._X_train: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None  # needed for LML recomputation
        self._L: torch.Tensor | None = None        # Cholesky factor of K_y
        self._alpha: torch.Tensor | None = None    # K_y^{-1} y

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Condition the GP on training data.

        Computes K_y = K(X, X) + sigma_n^2 * I, then factorizes it via
        Cholesky and solves for alpha = K_y^{-1} y. Both L and alpha are
        stored for use in predict().

        Args:
            X_train: (n x d) training inputs.
            y_train: (n,) training targets.
        """
        self._X_train = X_train
        self._y_train = y_train
        n = X_train.shape[0]

        # Build training kernel matrix: K_ij = k(x_i, x_j)
        K = self.kernel(X_train, X_train)  # (n, n) and positive definite

        # Add noise variance to diagonal: K_y = K + sigma_n^2 * I
        # Use exp(log_noise_var) so the value stays positive
        K_y = K + torch.exp(self.log_noise_var) * torch.eye(n, dtype=X_train.dtype)

        # Cholesky factorization: L @ L.T = K_y
        self._L = cholesky(K_y)

        # Solve K_y @ alpha = y  <=>  L @ L.T @ alpha = y
        self._alpha = solve_cholesky(self._L, y_train)

    def predict(self, X_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GP posterior mean and variance at test points.

        For each x* in X_test:
            mu(x*)    = k(x*, X)^T @ alpha
            sigma^2(x*) = k(x*, x*) - || L^{-1} k(x*, X) ||^2

        The mean computation is autograd-compatible: gradients flow through
        the kernel into x* (required by gradients.py for line search).

        For the variance, we use torch.linalg.solve_triangular instead of our
        custom solver because the loop-based solver breaks the autograd graph.

        Args:
            X_test: (m x d) test inputs.

        Returns:
            mu:  (m,) posterior mean at each test point.
            var: (m,) posterior variance at each test point.
        """
        if self._L is None or self._alpha is None or self._X_train is None:
            raise RuntimeError("Call fit() before predict().")

        # Cross-covariance between test and training points: (m, n)
        K_star = self.kernel(X_test, self._X_train)

        # Posterior mean: mu = K_star @ alpha,  shape (m,)
        mu = K_star @ self._alpha

        # Prior variance at test points: diagonal of k(X_test, X_test), shape (m,)
        K_star_diag = self.kernel(X_test, X_test).diagonal()

        # Solve L @ V = K_star.T  =>  V = L^{-1} K_star.T,  shape (n, m)
        # Uses torch.linalg for autograd compatibility through K_star.
        V = torch.linalg.solve_triangular(
            self._L, K_star.T, upper=False
        )

        # Posterior variance: sigma^2 = k(x*,x*) - ||v_star||^2,  shape (m,)
        var = K_star_diag - (V ** 2).sum(dim=0)

        return mu, var


