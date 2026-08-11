import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict

try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError("The Neural ODE model requires 'torchdiffeq'. Please install it using: pip install torchdiffeq")


class LatentAssemblyModel(nn.Module):
    """
    Discrete-Time Latent Microbiome Assembly Model.
    Predicts x_{T+1} given x_T using latent environmental variables.
    """
    def __init__(self, num_species, num_latent_dims, num_conditions=1):
        super(LatentAssemblyModel, self).__init__()
        self.N = num_species
        self.K = num_latent_dims
        
        # --- Model Parameters ---
        # Theta (N x K): The latent metabolic trait matrix for each species
        self.Theta = nn.Parameter(torch.randn(self.N, self.K) * 0.1)
        
        # W (K x K): The metabolic cross-feeding/pleiotropy matrix
        self.W = nn.Parameter(torch.randn(self.K, self.K) * 0.1)
        
        # a (num_conditions x K): The baseline environmental states 
        self.a = nn.Parameter(torch.randn(num_conditions, self.K) * 0.1)

    def forward(self, x_T, condition_idx):
        # 1. Community Aggregate Trait Expression
        community_traits = torch.matmul(x_T, self.Theta)
        
        # 2. Metabolic Modification (f)
        f = torch.matmul(community_traits, self.W.t())
        
        # 3. Final Environmental State (z_final)
        a_batch = self.a[condition_idx]
        z_final = a_batch - f
        
        # 4. Per-species fitness
        base_logits = -torch.matmul(z_final, self.Theta.t())
        
        # 5. Scores: Removed the clamp. If x_T is 0, score is exactly 0.
        x_T = torch.clamp(x_T, min=0)
        scores = x_T * torch.exp(base_logits)
        
        # 6. Normalize to get relative abundances
        x_T_plus_1 = scores / (scores.sum(dim=-1, keepdim=True))
        
        # 7. Safe log for KLDivLoss
        # Clamp ONLY for the log calculation, leaving the returned x_T_plus_1 with exact zeros
        safe_x_T_plus_1 = torch.clamp(x_T_plus_1, min=1e-9)
        log_x_T_plus_1 = torch.log(safe_x_T_plus_1)
        
        return log_x_T_plus_1, x_T_plus_1

    def find_steady_state(self, x_0, condition_idx, max_passages=2000, tol=1e-7):
        """
        Iterates the discrete map to find the steady-state relative abundances
        starting from an initial inoculum x_0.
        
        Args:
            x_0 (Tensor): Initial relative abundance (batch_size, N)
            condition_idx (Tensor): Environmental conditions (batch_size,)
            max_passages (int): Cutoff to prevent infinite loops
            tol (float): The tolerance for convergence (change in relative abundance)
            
        Returns:
            x_steady (Tensor): The final relative abundances.
            converged (bool): True if change fell below tol, False if it hit max_passages.
            passages (int): Number of passages simulated.
        """
        # Disable gradient tracking to prevent out-of-memory errors over 1000+ passages
        with torch.no_grad():
            x_T = x_0.clone()
            
            for passage in range(max_passages):
                _, x_T_plus_1 = self.forward(x_T, condition_idx)
                
                # Check maximum absolute difference across all species and batches
                max_diff = torch.max(torch.abs(x_T_plus_1 - x_T))
                
                if max_diff < tol:
                    return x_T_plus_1, True, passage + 1
                    
                x_T = x_T_plus_1
                
            return x_T, False, max_passages

class ConsumerResourceODEFunc(nn.Module):
    """
    The explicit ordinary differential equation (ODE) governing the
    coupled continuous-time consumer-resource dynamics using mass-action kinetics.
    """
    def __init__(self, num_species, num_latent_dims):
        super(ConsumerResourceODEFunc, self).__init__()
        self.N = num_species
        self.K = num_latent_dims
        
        # Theta (N x K): The latent metabolic trait matrix for each species
        self.Theta = nn.Parameter(torch.randn(self.N, self.K) * 0.1)
        
        # W (K x K): The metabolic cross-feeding/pleiotropy matrix
        self.W = nn.Parameter(torch.randn(self.K, self.K) * 0.1)

    def forward(self, t, S):
        # S is the concatenated state vector [N(t), R(t)]
        # Extract Absolute Abundances N(t) and Resources R(t)
        N_t = S[..., :self.N]
        R_t = S[..., self.N:]
        
        # ODE 1: dN_i/dt = N_i(t) * sum_k( Theta_ki * R_k(t) )
        # Shape manipulation: R_t is (..., K), Theta.t() is (K, N) -> (..., N)
        dN_dt = N_t * torch.matmul(R_t, self.Theta.t())
        
        # ODE 2: dR_k/dt = -R_k(t) * sum_l sum_j( W_kl * Theta_jl * N_j(t) )
        # community_traits = N(t) @ Theta -> (..., K)
        # metabolic_flux = community_traits @ W^T -> (..., K)
        community_traits = torch.matmul(N_t, self.Theta)
        metabolic_flux = torch.matmul(community_traits, self.W.t())
        
        # Multiply by -R_t to enforce mass-action kinetics (exponential decay)
        dR_dt = -R_t * metabolic_flux
        
        # Return concatenated derivatives
        return torch.cat([dN_dt, dR_dt], dim=-1)


class LatentAssemblyNeuralODE(nn.Module):
    """
    Continuous-Time Latent Microbiome Assembly Model via Neural ODEs.
    Predicts absolute final abundance N(T_end) and relative abundance x_{T+1} 
    by numerically integrating the exact coupled consumer-resource ODEs.
    """
    def __init__(self, num_species, num_latent_dims, num_conditions=1, default_dilution_factor=100.0):
        super(LatentAssemblyNeuralODE, self).__init__()
        self.N = num_species
        self.K = num_latent_dims
        self.default_dilution_factor = default_dilution_factor
        
        # The ODE function holds the core biological parameters (Theta and W)
        self.ode_func = ConsumerResourceODEFunc(self.N, self.K)
        
        # a (num_conditions x K): The baseline initial resources (R(0)) for fresh media
        self.a = nn.Parameter(torch.randn(num_conditions, self.K) * 0.1)

    def forward(self, N_prev, condition_idx, T_end=1.0, dilution_factor=None):
        """
        N_prev: Absolute final abundances from the previous passage (batch_size, N)
        condition_idx: Index of the environmental media (batch_size,)
        T_end: Duration of the serial dilution passage
        dilution_factor: Experimental dilution factor (e.g., 100 for a 1:100 dilution)
        """
        if dilution_factor is None:
            dilution_factor = self.default_dilution_factor
            
        # 0. Apply dilution to the previous saturated culture to get the inoculum
        N_0 = N_prev / dilution_factor

        # 1. Get initial resources for the given condition
        a_batch = self.a[condition_idx]
        
        # Use softplus to ensure strictly positive initial resources (biologically required)
        R_0 = F.softplus(a_batch) 
        
        # Ensure N_0 is strictly non-negative
        N_0_safe = torch.clamp(N_0, min=0.0)
        
        # 2. Concatenate into initial state vector S_0
        S_0 = torch.cat([N_0_safe, R_0], dim=-1)
        
        # 3. Define time span for the ODE solver [0, T_end]
        t_span = torch.tensor([0.0, float(T_end)], device=S_0.device, dtype=S_0.dtype)
        
        # 4. Integrate forward in continuous time using dopri5 (Runge-Kutta 4(5))
        # S_t shape will be (len(t_span), batch_size, N + K)
        S_t = odeint(self.ode_func, S_0, t_span, method='dopri5')
        
        # 5. Extract final state at T_end
        S_final = S_t[-1]
        N_final = S_final[..., :self.N]
        R_final = S_final[..., self.N:]
        
        # 6. Convert final absolute abundances back to relative abundances (x_{T+1})
        # Safety clamp to handle minor numerical deviations near zero from the ODE solver
        N_final_clamped = torch.clamp(N_final, min=0.0) 
        x_T_plus_1 = N_final_clamped / (N_final_clamped.sum(dim=-1, keepdim=True))
        
        # 7. Safe log for KLDivLoss
        safe_x_T_plus_1 = torch.clamp(x_T_plus_1, min=1e-9)
        log_x_T_plus_1 = torch.log(safe_x_T_plus_1)
        
        return log_x_T_plus_1, x_T_plus_1, N_final, R_final