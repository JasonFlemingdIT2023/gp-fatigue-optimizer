import math
import torch

class MaternKernel:

    def __init__(self, length_scale: float=1.0, output_variance: float=1.0, nu: float=2.5):
        self.nu = nu

        # Store hyperparameters in log-space so exp() keeps them strictly positive.
        # requires_grad=True allows torch.optim.LBFGS to compute d(LML)/d(theta)
        # via autograd -- same trick as GPyTorch's raw_lengthscale.
        self.log_length_scale = torch.tensor(math.log(length_scale), requires_grad=True)
        self.log_output_variance = torch.tensor(math.log(output_variance), requires_grad=True)
        
    def _compute_distance(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        '''
        X1 = (n,d), X2 = (m,d), n = number of points, d= dimension
        Returns: (n,m) distance matrix
        
        none for boradcasting
        Example:
        
        X1[:, none, :].shape = (2, 1, 5)
        X2[none, :, :].shape = (1, 3, 5)
        
        Problem in first 2 dimensions
        
        '''
        diff = X1[:, None, :] - X2[None, :, :] #(n,m,d)
        # 1e-10 prevents sqrt(0) whose gradient is inf -- happens on diagonal
        # of kernel(X_train, X_train) where every point is compared with itself.
        return torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-10) #(n,m)
    
    def _matern_12(self, r: torch.Tensor) -> torch.Tensor:
        ls = torch.exp(self.log_length_scale)
        ov = torch.exp(self.log_output_variance)
        return ov * torch.exp(-r / ls)

    def _matern_32(self, r: torch.Tensor) -> torch.Tensor:
        ls = torch.exp(self.log_length_scale)
        ov = torch.exp(self.log_output_variance)
        sqrt3r = (3 ** 0.5) * r / ls
        return ov * (1 + sqrt3r) * torch.exp(-sqrt3r)

    def _matern_52(self, r: torch.Tensor) -> torch.Tensor:
        ls = torch.exp(self.log_length_scale)
        ov = torch.exp(self.log_output_variance)
        sqrt5r = (5 ** 0.5) * r / ls
        return ov * (1 + sqrt5r + (5 * r**2) / (3 * ls**2)) * torch.exp(-sqrt5r)
    
    def __call__(self, X1: torch.Tensor,X2: torch.Tensor) -> torch.Tensor:
       r = self._compute_distance(X1, X2)
    
       if self.nu == 0.5:
           return self._matern_12(r)
       elif self.nu == 1.5:
           return self._matern_32(r)
       elif self.nu == 2.5:
           return self._matern_52(r)
       else:
           raise ValueError(f"nu must be 0.5, 1.5, or 2.5 - got {self.nu}")
    


        
    
    