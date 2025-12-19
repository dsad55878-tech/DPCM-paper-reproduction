# -*- coding: utf-8 -*-
"""
dpcm_core.py
Core logic for the Dynamic PsyCap Coupling Model (DPCM).
Contains the stochastic differential equation solver and measurement filter.
"""

import numpy as np
import pandas as pd

# Standard small epsilon for numerical stability
EPS = 1e-12

def check_stability(A_matrix, fix=True):
    """
    Checks if the spectral radius of the state transition matrix A is < 1.
    If fix=True, it scales the matrix to ensure stability (rho = 0.95).
    """
    vals = np.linalg.eigvals(A_matrix)
    rho = np.max(np.abs(vals))
    
    if rho >= 1.0 and fix:
        scale_factor = 0.95 / rho
        return A_matrix * scale_factor
    return A_matrix

def get_sampling_indices(total_steps, delta_t):
    """Returns indices for weekly (1), biweekly (2), or pre-post (end-points) sampling."""
    if delta_t >= 5: 
        # Pre-post design: only first and last time points
        return [0, total_steps - 1]
    return list(range(0, total_steps, delta_t))

def generate_trajectory(N, T, A, B, sigma_noise, rng):
    """
    Generates latent continuous-time trajectories for N subjects over T time steps.
    """
    # 1. Intervention Input (X): Decays over time
    ai_baseline = rng.normal(0, 1, size=N)
    input_stream = np.zeros((N, T))
    decay_rate = 0.10
    
    for t in range(T):
        input_stream[:, t] = ai_baseline * np.exp(-decay_rate * t)
        
    # 2. Latent State Update (The Dynamic Process)
    # State dimension = 4 (Hope, Efficacy, Resilience, Optimism)
    dim = A.shape[0]
    states = np.zeros((N, T, dim))
    current_state = np.zeros((N, dim)) # Start at 0
    
    for t in range(T):
        if t > 0:
            # SDE: x(t) = A * x(t-1) + B * u(t) + noise
            autoregressive = current_state @ A.T
            # Intervention effect
            forced_response = np.outer(input_stream[:, t], B)
            # Stochastic shock
            innovation = rng.multivariate_normal(np.zeros(dim), sigma_noise, size=N)
            
            current_state = autoregressive + forced_response + innovation
            
        states[:, t, :] = current_state
        
    return states, input_stream.mean(axis=1)

def apply_measurement_model(latent_states, input_vec, N, delta_t, alpha, rng):
    """
    Applies the observation filter: Subsampling (Delta t) + Measurement Error (Alpha).
    Returns a DataFrame ready for regression/mediation analysis.
    """
    # 1. Determine Sampling Schedule
    T_total = latent_states.shape[1]
    obs_idx = get_sampling_indices(T_total, delta_t)
    
    # 2. Add Measurement Error based on Reliability (Alpha)
    # var_error = (1 - alpha) / alpha
    error_var = (1.0 - alpha) / (alpha + EPS)
    sigma_error = np.sqrt(error_var)
    
    observed_data = np.zeros((N, len(obs_idx), 4))
    
    for i, t in enumerate(obs_idx):
        measurement_noise = rng.normal(0, sigma_error, size=(N, 4))
        observed_data[:, i, :] = latent_states[:, t, :] + measurement_noise
        
    # 3. Aggregate for Analysis (Mean score across observed time points)
    # Mediator (M) is the average of the 4 PsyCap components across observed times
    psycap_composite = observed_data.mean(axis=2) # Average across 4 dimensions
    M_raw = psycap_composite.mean(axis=1)         # Average across time points
    
    # Standardize M
    M_z = (M_raw - M_raw.mean()) / (M_raw.std() + EPS)
    
    # Standardize X (Input)
    X_z = (input_vec - input_vec.mean()) / (input_vec.std() + EPS)
    
    # 4. Generate Outcome (Y)
    # Y depends on the FINAL latent state (accumulated growth) + direct effect
    final_latent_avg = latent_states[:, -1, :].mean(axis=1)
    
    # Mechanism: Y = 0.3*M_final + 0.05*X + Noise
    Y_raw = 0.3 * final_latent_avg + 0.05 * input_vec + rng.normal(0, 0.5, size=N)
    Y_z = (Y_raw - Y_raw.mean()) / (Y_raw.std() + EPS)
    
    return pd.DataFrame({'X': X_z, 'M': M_z, 'Y': Y_z})