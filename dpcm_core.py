# -*- coding: utf-8 -*-
"""
dpcm_core.py
Core logic for the Dynamic PsyCap Coupling Model (DPCM).
Contains the stochastic differential equation solver and measurement filter.
Refined to match the 'Continuous vs Discrete' logic in the paper.
"""

import numpy as np
import pandas as pd

# Standard small epsilon for numerical stability
EPS = 1e-12

def check_stability(A_matrix, fix=True):
    """
    Checks if the spectral radius of the state transition matrix A is < 1.
    Matches Appendix A.1 in the paper.
    """
    vals = np.linalg.eigvals(A_matrix)
    rho = np.max(np.abs(vals))
    
    if rho >= 1.0 and fix:
        scale_factor = 0.95 / rho
        return A_matrix * scale_factor
    return A_matrix

def generate_trajectory(N, weeks, A, B, sigma_noise, rng):
    """
    Generates latent trajectories. 
    CRITICAL CHANGE: To mimic 'Continuous Time' vs 'Discrete Measurement',
    we simulate 7 steps per week (Daily dynamics) but will sample sparsely later.
    """
    steps_per_week = 7
    T_steps = weeks * steps_per_week
    
    # 1. Intervention Input (X): Decays over time (e.g. novelty effect of AI)
    # Scaled to be daily input
    ai_baseline = rng.normal(0, 1, size=N)
    input_stream = np.zeros((N, T_steps))
    decay_rate = 0.05 / steps_per_week # Decay is slower per day
    
    for t in range(T_steps):
        input_stream[:, t] = ai_baseline * np.exp(-decay_rate * t)
        
    # 2. Latent State Update (The Dynamic Process)
    dim = A.shape[0]
    states = np.zeros((N, T_steps, dim))
    current_state = np.zeros((N, dim)) 
    
    # Scale A matrix for daily steps (Micro-dynamics)
    # If A is weekly inertia, daily inertia is A^(1/7), roughly speaking.
    # For simulation stability, we keep A but scale the noise.
    
    for t in range(T_steps):
        if t > 0:
            # SDE approximation
            autoregressive = current_state @ A.T
            forced_response = np.outer(input_stream[:, t], B)
            innovation = rng.multivariate_normal(np.zeros(dim), sigma_noise, size=N)
            
            # Update
            current_state = autoregressive + forced_response + innovation
            
        states[:, t, :] = current_state
        
    # Return states and the average Input level (as a trait-like predictor)
    return states, input_stream.mean(axis=1)

def apply_measurement_model(latent_states, input_vec, N, delta_t_weeks, alpha, rng):
    """
    Applies the observation filter.
    latent_states shape: [N, Total_Days, 4]
    delta_t_weeks: 1, 2, or 5
    """
    steps_per_week = 7
    total_days = latent_states.shape[1]
    
    # Determine sampling indices (Day 0, Day 7, Day 14...)
    interval_days = delta_t_weeks * steps_per_week
    
    if delta_t_weeks >= 5:
        # Pre-Post Design: Only Start (Day 0) and End (Last Day)
        # Matches 'Pre-post' description in Table 2
        obs_idx = [0, total_days - 1]
    else:
        # Weekly or Biweekly tracking
        obs_idx = list(range(0, total_days, interval_days))
    
    # Measurement Error Calculation
    # Reliability alpha = Var(T) / (Var(T) + Var(E))
    # We approximate Var(T) = 1 (Standardized construct) for calibration
    error_var = (1.0 - alpha) / (alpha + EPS)
    sigma_error = np.sqrt(error_var)
    
    # Extract observed values
    observed_M_list = []
    
    for t in obs_idx:
        if t < total_days:
            # Get true state at that day
            true_state = latent_states[:, t, :]
            # Add noise
            noise = rng.normal(0, sigma_error, size=(N, 4))
            obs = true_state + noise
            # Composite score (Average of 4 sub-scales)
            observed_M_list.append(obs.mean(axis=1))
            
    # Convert to array [N, TimePoints]
    observed_M_array = np.array(observed_M_list).T 
    
    # CALCULATE MEDIATOR (M): Average of the observed snapshots
    # This is where Aliasing happens: Sparse sampling misses the peaks/valleys
    M_observed_mean = observed_M_array.mean(axis=1)
    
    # Standardize M
    M_z = (M_observed_mean - M_observed_mean.mean()) / (M_observed_mean.std() + EPS)
    X_z = (input_vec - input_vec.mean()) / (input_vec.std() + EPS)
    
    # Generate Outcome (Y)
    # The outcome is driven by the TRUE latent accumulation (not the observed one)
    # This creates the "Signal Loss" when we try to predict Y using observed M
    true_cumulative_M = latent_states.mean(axis=1).mean(axis=1) # True average over all days
    
    # Structural Model: Y = 0.4 * True_M + 0.1 * X + Noise
    Y_raw = 0.4 * true_cumulative_M + 0.1 * input_vec + rng.normal(0, 0.5, size=N)
    Y_z = (Y_raw - Y_raw.mean()) / (Y_raw.std() + EPS)
    
    return pd.DataFrame({'X': X_z, 'M': M_z, 'Y': Y_z})