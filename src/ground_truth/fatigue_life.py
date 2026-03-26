import torch

#Material constants for S355

SIGMA_U = 510  #Ultimte tensile strength in MPa
B = 8          #Fatigue Exponent
C = 2.56e24    #Basquin coefficient
GAMMA = 0.5    #Walker exponent
ALPHA = 0.0003 #Temperature coefficient
BETA = 1.2     #Temperature exponent
T0 = 20        #Reference temperature in °C
SIGMA_N = 0.1  #Noise standard deviation


def fatigue_life(x:torch.Tensor, noisy: bool = True) -> torch.Tensor:
     
    """Compute log10 fatigue life for S355 steel.
    
    Args:
        x: (N x 5) tensor with columns [sigma_a, sigma_m, T, R, k_s]
        noisy: If True, adds gaussian noise to the output.
    
    Returns:
        (N,) tensor of log10 fatigue life values.
    """
    
    sigma_a = x[:, 0]
    sigma_m = x[:, 1]
    T       = x[:, 2]
    R       = x[:, 3]
    k_s     = x[:, 4]
    
    #Goodman correction
    sigma_eff = sigma_a / (1 - sigma_m / SIGMA_U)
    #Temperature correction
    sigma_eff = sigma_eff * (1 + ALPHA * (T - T0)**BETA)
    #Walker correction
    sigma_eff = sigma_eff * (1 - R)**GAMMA
    #Surface roughness correction
    sigma_eff = sigma_eff * (1 / k_s)
    
    #Basquin
    N = C * sigma_eff**(-B)
    
    N_log = torch.log10(N)
    
    #Noise
    if noisy:
        N_log = N_log + torch.randn_like(N_log)*SIGMA_N
    
    return N_log
    



