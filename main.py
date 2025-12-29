#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Execution Script for DPCM Simulation.
Reproduces the "Design-Dependent Detectability" experiments.

Features:
- Generates synthetic learner trajectories.
- Applies different sampling intervals (Delta t).
- Performs mediation analysis (Sobel test by default for speed).

Usage:
    python main.py --n_jobs 4
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

# Import core logic (ensure dpcm_core.py is in the same folder)
from dpcm_core import check_stability, generate_trajectory, apply_measurement_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
# Analysis Logic (The Key Update)
# -----------------------------------------------------------------------------

def run_mediation_analysis(df, method='sobel', n_boot=1000):
    """
    Estimates Indirect Effect (ab) and determines significance.
    
    Args:
        df: DataFrame with columns ['X', 'M', 'Y']
        method: 'sobel' (Fast, default) or 'bootstrap' (Slow, Gold Standard)
        n_boot: Number of bootstrap resamples (only used if method='bootstrap')
    """
    try:
        # 1. Base OLS Models (Always required)
        # Path a: X -> M
        model_a = sm.OLS(df['M'], sm.add_constant(df['X'])).fit()
        a_coef = model_a.params.get('X', 0.0)
        a_se = model_a.bse.get('X', 1.0)
        
        # Path b: M -> Y (controlling for X)
        model_b = sm.OLS(df['Y'], sm.add_constant(df[['M', 'X']])).fit()
        b_coef = model_b.params.get('M', 0.0)
        b_se = model_b.bse.get('M', 1.0)
        
        # Point Estimate
        ab = a_coef * b_coef
        
        # 2. Significance Testing
        if method == 'sobel':
            # --- Fast Path: Sobel Test ---
            # Standard Error of ab
            ab_se = np.sqrt(b_coef**2 * a_se**2 + a_coef**2 * b_se**2)
            z_score = ab / ab_se
            # p < 0.05 corresponds to |z| > 1.96
            is_sig = abs(z_score) > 1.96
            
        elif method == 'bootstrap':
            # --- Slow Path: Percentile Bootstrap ---
            # NOTE: This is computationally intensive. Only use for small N or verification.
            boot_betas = []
            n_samples = len(df)
            
            for _ in range(n_boot):
                # Resample with replacement
                df_boot = df.sample(n=n_samples, replace=True)
                
                # Re-estimate a (Simplified for speed)
                cov_xm = np.cov(df_boot['X'], df_boot['M'])[0, 1]
                var_x = np.var(df_boot['X'])
                a_boot = cov_xm / (var_x + 1e-9)
                
                # Re-estimate b (Simplified, ignoring X control to speed up loop, 
                # or use full OLS if precision is critical. Here we use full OLS for correctness)
                # To save time in demo code, we use the pre-calculated logic or minimal OLS
                ma = sm.OLS(df_boot['M'], sm.add_constant(df_boot['X'])).fit()
                mb = sm.OLS(df_boot['Y'], sm.add_constant(df_boot[['M', 'X']])).fit()
                boot_betas.append(ma.params['X'] * mb.params['M'])
            
            # Percentile Interval
            lower = np.percentile(boot_betas, 2.5)
            upper = np.percentile(boot_betas, 97.5)
            is_sig = not (lower <= 0 <= upper)
            
        else:
            raise ValueError("Method must be 'sobel' or 'bootstrap'")

        return ab, is_sig
        
    except Exception:
        return 0.0, False

def single_iteration(seed, config, N, dt, alpha, A_stable):
    """Runs one single simulation iteration."""
    rng = np.random.default_rng(seed)
    
    # 1. Generate underlying process
    B = np.array(config['model']['B_sensitivity'])
    Sigma = np.diag(config['model']['process_noise'])
    latent, input_vec = generate_trajectory(N, config['design']['total_weeks'], A_stable, B, Sigma, rng)
    
    # 2. Apply measurement design filter
    df = apply_measurement_model(latent, input_vec, N, dt, alpha, rng)
    
    # 3. Analyze
    # !!! KEY POINT: We use 'sobel' by default for computational efficiency.
    # Change to 'bootstrap' only for small-scale verification.
    beta, sig = run_mediation_analysis(df, method='sobel')
    
    return beta, sig

# -----------------------------------------------------------------------------
# 4. Main Experiment Logic
# -----------------------------------------------------------------------------

def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of CPU cores to use')
    args = parser.parse_args()

    # 1. Load Configuration
    if not os.path.exists('config.json'):
        logging.error("config.json not found! Please create it first.")
        return

    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 2. Setup Matrices
    A_raw = np.array(config['model']['A_matrix'])
    A_stable = check_stability(A_raw)
    
    # 3. Define Grid
    N_list = config['experiment']['sample_sizes']
    alpha_list = config['experiment']['reliabilities']
    dt_list = config['experiment']['intervals']
    n_iters = config['experiment']['iterations']
    
    total_conds = len(N_list) * len(alpha_list) * len(dt_list)
    results = []
    counter = 0

    logging.info(f"Starting Simulation: {total_conds} conditions, {n_iters} iterations each.")
    logging.info(f"Method: Sobel approximation (for speed). Change code to 'bootstrap' for validation.")
    
    # 4. Main Loop
    for N in N_list:
        for alpha in alpha_list:
            for dt in dt_list:
                counter += 1
                # Log progress every few steps
                if counter % 3 == 0 or counter == 1: 
                    logging.info(f"Processing [{counter}/{total_conds}]: N={N}, alpha={alpha}, dt={dt}")
                
                # Parallel Execution
                seeds = np.random.SeedSequence(42 + counter).generate_state(n_iters)
                
                batch_res = Parallel(n_jobs=args.n_jobs)(
                    delayed(single_iteration)(seed, config, N, dt, alpha, A_stable)
                    for seed in seeds
                )
                
                # Aggregation
                betas = [x[0] for x in batch_res]
                sigs = [x[1] for x in batch_res]
                
                results.append({
                    'N': N,
                    'Reliability': alpha,
                    'Delta_t': dt,
                    'Mean_Indirect_Effect': np.mean(betas),
                    'Statistical_Power': np.mean(sigs) * 100 # Convert to percentage
                })
    
    # 5. Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results.csv', index=False)
    logging.info("Simulation completed. Results saved to 'simulation_results.csv'.")
    
    # 6. Quick Sanity Check (Print to console)
    try:
        # Check power drop for N=500, Alpha=0.8
        cond_high = df_results[(df_results['N']==500) & (df_results['Delta_t']==1) & (df_results['Reliability']==0.80)]
        cond_low = df_results[(df_results['N']==500) & (df_results['Delta_t']==5) & (df_results['Reliability']==0.80)]
        
        if not cond_high.empty and not cond_low.empty:
            p1 = cond_high['Statistical_Power'].values[0]
            p5 = cond_low['Statistical_Power'].values[0]
            logging.info(f"--- Verification ---")
            logging.info(f"Power at dt=1: {p1:.1f}%")
            logging.info(f"Power at dt=5: {p5:.1f}%")
            logging.info(f"Drop: {p1 - p5:.1f}% (Should be positive)")
    except Exception as e:
        logging.warning(f"Could not run sanity check: {e}")

if __name__ == "__main__":
    main()