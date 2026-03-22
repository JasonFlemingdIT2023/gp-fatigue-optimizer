import numpy as np

class MaternKernel:
    
    def __init__(self, length_scale=1.0,output_variance=1.0, nu=2.5):
        self.length_scale = length_scale
        self.output_variance = output_variance
        self.nu = nu
        
    def _compute_distance(self, X1, X2):
        '''
        X1 = (n,d), X2 = (m,d), n = number of points, d= dimension
        Returns: (n,m) distance matrix
        
        newaxis for boradcasting
        Example:
        
        X1[:, np.newaxis, :].shape = (2, 1, 5)
        X2[np.newaxis, :, :].shape = (1, 3, 5)
        
        Problem in first 3 dimensions
        
        '''
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :] #(n,m,d)
        return np.sqrt(np.sum(diff**2, axis=-1)) #(n,m)
    
    def _matern_12(self, r):
         return self.output_variance * np.exp(-r / self.length_scale)


    def _matern_32(self, r):
        sqrt3r = np.sqrt(3) * r / self.length_scale
        return self.output_variance * (1 + sqrt3r) * np.exp(-sqrt3r)
    

    def _matern_52(self, r):
        sqrt5r = np.sqrt(5) * r / self.length_scale
        return self.output_variance * (1 + sqrt5r + (5 * r**2) / (3* self.length_scale**2)) * np.exp(-sqrt5r)
    
    def __call__(self, X1,X2):
       r = self._compute_distance(X1, X2)
    
       if self.nu == 0.5:
           return self._matern_12(r)
       elif self.nu == 1.5:
           return self._matern_32(r)
       elif self.nu == 2.5:
           return self._matern_52(r)
       else:
           raise ValueError(f"nu must be 0.5, 1.5, or 2.5 - got {self.nu}")
    




        
        
    
    