import torch


def cholesky(A: torch.Tensor) -> torch.Tensor:
    
    """Compute Cholesky factorization A = L @ L.T from scratch.

    Args:
        A: (n x n) symmetric positive definite matrix.

    Returns:
        L: (n x n) lower triangular matrix such that L @ L.T = A.
    """
    n = A.shape[0]
    L = torch.zeros_like(A)

    for j in range(n):
        # Diagonal entry: l_jj = sqrt(a_jj - sum(l_jk^2) for k < j)
        L[j, j] = torch.sqrt(A[j, j] - torch.sum(L[j, :j] ** 2))

        # Entries below diagonal: l_ij = (a_ij - sum(l_ik * l_jk) for k < j) / l_jj
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - torch.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L


def solve_triangular(L: torch.Tensor, b: torch.Tensor, upper: bool = False) -> torch.Tensor:
    
    """Solve a triangular system via forward or backward substitution.

    Args:
        L: (n x n) triangular matrix.
        b: (n,) right hand side vector.
        upper: If False (default), solve lower triangular L @ x = b via forward
            substitution. If True, solve upper triangular U @ x = b via backward
            substitution.

    Returns:
        x: (n,) solution vector.
    """
    n = len(b)
    x = torch.zeros(n, dtype=b.dtype)

    if not upper:
        # Forward substitution: x_i = (b_i - sum(L_ik * x_k) for k < i) / L_ii
        for i in range(n):
            x[i] = (b[i] - torch.sum(L[i, :i] * x[:i])) / L[i, i]
    else:
        # Backward substitution: x_i = (b_i - sum(U_ik * x_k) for k > i) / U_ii
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - torch.sum(L[i, i + 1:] * x[i + 1:])) / L[i, i]

    return x


def solve_cholesky(L: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve A @ x = b given Cholesky factor L where A = L @ L.T.

    Args:
        L: (n x n) lower triangular Cholesky factor.
        b: (n,) right hand side vector.

    Returns:
        x: (n,) solution vector.
    """
    # Step 1: solve L @ v = b via forward substitution
    v = solve_triangular(L, b, upper=False)
    # Step 2: solve L.T @ x = v via backward substitution
    x = solve_triangular(L.T, v, upper=True)
    return x




