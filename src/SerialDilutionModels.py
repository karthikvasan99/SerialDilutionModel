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


class LinearSoftmaxModel(nn.Module):
    """
    Linear Markov / Transition Baseline Model with Softmax.
    Ablates the latent metabolic bottleneck (Theta, W, a) in favor of a direct full-rank
    transition matrix A (N x N) and condition-specific bias vectors b (num_conditions x N).
    
    If enforce_inoculum_mask=True (default), computes scores = x_T * exp(logits), preserving
    the ecological invariant that absent species cannot spontaneously emerge.
    If enforce_inoculum_mask=False, computes pure softmax(logits).
    """
    def __init__(self, num_species, num_conditions=1, enforce_inoculum_mask=True):
        super(LinearSoftmaxModel, self).__init__()
        self.N = num_species
        self.num_conditions = num_conditions
        self.enforce_inoculum_mask = enforce_inoculum_mask
        
        # Direct species-to-species interaction/transition matrix A (N x N)
        self.A = nn.Parameter(torch.randn(self.N, self.N) * 0.01)
        
        # Condition-specific baseline fitness/growth vector b (num_conditions x N)
        self.b = nn.Parameter(torch.randn(num_conditions, self.N) * 0.01)

    def forward(self, x_T, condition_idx):
        # 1. Linear state transition: x_T @ A + b[condition_idx]
        b_batch = self.b[condition_idx]
        logits = torch.matmul(x_T, self.A) + b_batch
        
        # 2. Compute probabilities / relative abundances
        if self.enforce_inoculum_mask:
            x_T_clamped = torch.clamp(x_T, min=0.0)
            scores = x_T_clamped * torch.exp(logits)
            x_T_plus_1 = scores / (scores.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            x_T_plus_1 = F.softmax(logits, dim=-1)
            
        # 3. Safe log for KLDivLoss
        safe_x_T_plus_1 = torch.clamp(x_T_plus_1, min=1e-9)
        log_x_T_plus_1 = torch.log(safe_x_T_plus_1)
        
        return log_x_T_plus_1, x_T_plus_1

    def find_steady_state(self, x_0, condition_idx, max_passages=2000, tol=1e-7):
        """
        Iterates the discrete map to find the steady-state relative abundances
        starting from an initial inoculum x_0.
        """
        with torch.no_grad():
            x_T = x_0.clone()
            for passage in range(max_passages):
                _, x_T_plus_1 = self.forward(x_T, condition_idx)
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
        
        # alpha (N): Intraspecific competition (self-limitation) coefficient
        self.alpha_raw = nn.Parameter(torch.randn(self.N) * 0.01)

    def forward(self, t, S):
        # S is the concatenated state vector [N(t), R(t)]
        # Extract Absolute Abundances N(t) and Resources R(t)
        N_t = S[..., :self.N]
        R_t = S[..., self.N:]
        
        # Enforce strict positivity for alpha using softplus
        alpha = torch.nn.functional.softplus(self.alpha_raw)
        
        # Clamp to prevent numerical negatives during dopri5 integration
        N_safe = torch.clamp(N_t, min=0.0)
        R_safe = torch.clamp(R_t, min=0.0)
        
        # ODE 1: dN_i/dt = N_i(t) * sum_k( Theta_ki * R_k(t) ) - alpha_i * N_i(t)^2
        # Shape manipulation: R_t is (..., K), Theta.t() is (K, N) -> (..., N)
        growth_from_resources = torch.matmul(R_safe, self.Theta.t())
        dN_dt = (N_safe * growth_from_resources) - (alpha * (N_safe ** 2))
        
        # ODE 2: dR_k/dt = -R_k(t) * sum_l sum_j( W_kl * Theta_jl * N_j(t) )
        # community_traits = N(t) @ Theta -> (..., K)
        # metabolic_flux = community_traits @ W^T -> (..., K)
        community_traits = torch.matmul(N_safe, self.Theta)
        metabolic_flux = torch.matmul(community_traits, self.W.t())
        
        # Multiply by -R_safe to enforce mass-action kinetics (exponential decay)
        dR_dt = -R_safe * metabolic_flux
        
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

    def forward(self, N_prev, condition_idx, T_end=1.0, dilution_factor=None, ode_method='rk4', ode_step_size=0.05):
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
        
        # 4. Integrate forward in continuous time using chosen solver
        if ode_method in ['dopri5', 'bdf', 'adams']:
            S_t = odeint(self.ode_func, S_0, t_span, method=ode_method)
        else:
            S_t = odeint(self.ode_func, S_0, t_span, method=ode_method, options={'step_size': ode_step_size})
        
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


class PurePhysicalConsumerResourceNeuralODE(nn.Module):
    """
    Consolidated Physical Consumer-Resource Neural ODE Framework.
    Guarantees:
      1. Mass Conservation: Cross-feeding matrix W >= 0.
      2. Intra-specific Self-Limitation: alpha_i = softplus(alpha_raw) > 0.
      3. Non-negative Resource Utilization Traits: Theta >= 0.
      4. Freely Learned Environmental Resources: R0 = softplus(a) > 0 (no artificial scaling constraints).
      5. Unnormalized Physical Biomass Propagation: Serial dilution divides true unnormalized
         biomass by the dilution factor (N_0 = N_tau / D + dispersal) without simplex resets.
    """
    def __init__(self, num_species, num_latent_dims, num_conditions=1, default_dilution_factor=100.0, dispersal_rate=1e-6):
        super(PurePhysicalConsumerResourceNeuralODE, self).__init__()
        self.N = num_species
        self.K = num_latent_dims
        self.num_conditions = num_conditions
        self.default_dilution_factor = default_dilution_factor
        self.dispersal_rate = dispersal_rate

        # Unconstrained environmental baseline resources (R0 = softplus(a))
        self.a = nn.Parameter(torch.randn(num_conditions, num_latent_dims) * 0.5 + 2.0)
        # Metabolic traits Theta >= 0
        self.theta_raw = nn.Parameter(torch.randn(num_species, num_latent_dims) * 0.1)
        # Cross-feeding W >= 0
        self.w_raw = nn.Parameter(torch.randn(num_latent_dims, num_latent_dims) * 0.1)
        # Self-limitation alpha > 0
        self.alpha_raw = nn.Parameter(torch.randn(num_species) * 0.05 - 1.0)
        # Initial unnormalized biomass scale
        self.b0_raw = nn.Parameter(torch.tensor(2.0))

    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        W = F.softplus(self.w_raw)
        alpha = F.softplus(self.alpha_raw)
        R0 = F.softplus(self.a)
        B0 = F.softplus(self.b0_raw)
        return Theta, W, alpha, R0, B0

    def ode_deriv(self, t, S, Theta, W, alpha):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        
        # dN/dt = N * (R @ Theta^T) - alpha * N^2
        growth = torch.matmul(R_t, Theta.t())
        dN_dt = N_t * growth - alpha * (N_t ** 2)
        
        # dR/dt = - R * ((N @ Theta) @ W^T)
        flux = torch.matmul(torch.matmul(N_t, Theta), W.t())
        dR_dt = - R_t * flux
        
        return torch.cat([dN_dt, dR_dt], dim=-1)

    def forward(self, N_prev, condition_idx, T_end=1.0, dilution_factor=None, ode_step_size=0.05):
        """
        Executes one unnormalized physical dilution cycle.
        N_prev: unnormalized biomass from previous cycle (or initial biomass).
        """
        if dilution_factor is None:
            dilution_factor = self.default_dilution_factor
            
        Theta, W, alpha, R0_table, B0 = self.get_params()
        
        # Physical serial dilution transfer: N_0 = N_prev / D + dispersal (NO SIMPLEX NORMALIZATION)
        N_0 = torch.clamp(N_prev / dilution_factor + self.dispersal_rate, min=0.0)
        R_0 = R0_table[condition_idx]
        
        S_0 = torch.cat([N_0, R_0], dim=-1)
        t_span = torch.tensor([0.0, float(T_end)], device=S_0.device, dtype=S_0.dtype)
        
        def f(t, S):
            return self.ode_deriv(t, S, Theta, W, alpha)
            
        S_t = odeint(f, S_0, t_span, method='euler', options={'step_size': ode_step_size})
        S_final = S_t[-1]
        N_final = torch.clamp(S_final[..., :self.N], min=0.0)
        R_final = S_final[..., self.N:]
        
        # Relative abundance projection (for loss against sequencing reads)
        tot_bio = N_final.sum(dim=-1, keepdim=True) + 1e-8
        x_pred = N_final / tot_bio
        safe_x_pred = torch.clamp(x_pred, min=1e-9)
        log_x_pred = torch.log(safe_x_pred)
        
        return log_x_pred, x_pred, N_final, R_final
