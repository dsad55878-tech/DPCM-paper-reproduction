#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Execution Script for DPCM Simulation.
Reproduces the "Design-Dependent Detectability" experiments.

Usage:
    python main.py
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

# Import core logic
from dpcm_core import check_stability, generate_trajectory, apply_measurement_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_mediation(df):
    """
    Runs simple mediation analysis (Baron & Kenny approach for speed).
    Returns standardized indirect effect (beta) and significance.
    """
    try:
        # Path a: X -> M
        model_a = sm.OLS(df['M'], sm.add_constant(df['X'])).fit()
        a_coef = model_a.params.get('X', 0.0)
        
        # Path b: M -> Y (controlling for X)
        model_b = sm.OLS(df['Y'], sm.add_constant(df[['M', 'X']])).fit()
        b_coef = model_b.params.get('M', 0.0)
        
        # Indirect Effect
        ab = a_coef * b_coef
        
        # Sobel Test (Approximate significance)
        a_se = model_a.bse.get('X', 1.0)
        b_se = model_b.bse.get('M', 1.0)
        z_score = ab / np.sqrt(b_coef**2 * a_se**2 + a_coef**2 * b_se**2)
        is_sig = abs(z_score) > 1.96
        
        return ab, is_sig
    except Exception:
        return 0.0, False

def single_run(seed, config, N, dt, alpha, A_stable):
    rng = np.random.default_rng(seed)
    
    # 1. Generate underlying process
    B = np.array(config['model']['B_sensitivity'])
    Sigma = np.diag(config['model']['process_noise'])
    latent, input_vec = generate_trajectory(N, config['design']['total_weeks'], A_stable, B, Sigma, rng)
    
    # 2. Apply measurement design filter
    df = apply_measurement_model(latent, input_vec, N, dt, alpha, rng)
    
    # 3. Analyze
    beta, sig = run_mediation(df)
    return beta, sig

def main():
    # 1. Load Configuration
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
    
    # 4. Main Loop
    for N in N_list:
        for alpha in alpha_list:
            for dt in dt_list:
                counter += 1
                if counter % 3 == 0: # Reduce log verbosity
                    logging.info(f"Processing... [{counter}/{total_conds}] N={N}, alpha={alpha}, dt={dt}")
                
                # Parallel Execution
                seeds = np.random.SeedSequence(42 + counter).generate_state(n_iters)
                
                batch_res = Parallel(n_jobs=-1)(
                    delayed(single_run)(seed, config, N, dt, alpha, A_stable)
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
                    'Statistical_Power': np.mean(sigs)
                })
    
    # 5. Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results.csv', index=False)
    logging.info("Simulation completed. Results saved to 'simulation_results.csv'.")
    
    # 6. Sanity Check (Optional print)
    # Check if dt=5 has lower power than dt=1
    try:
        p_high = df_results[(df_results['N']==500) & (df_results['Delta_t']==1)]['Statistical_Power'].mean()
        p_low = df_results[(df_results['N']==500) & (df_results['Delta_t']==5)]['Statistical_Power'].mean()
        logging.info(f"Sanity Check (N=500): Power at dt=1 is {p_high:.2f}, at dt=5 is {p_low:.2f}")
    except:
        pass

if __name__ == "__main__":
    main()