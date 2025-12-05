Dynamic PsyCap Coupling Model (DPCM) Replication Repository
1. Overview
This repository contains the full implementation of the Dynamic PsyCap Coupling Model (DPCM), along with data generation scripts and analysis tools. It is designed to support the simulation study on the detectability of Psychological Capital (PsyCap) mediation in AI-mediated learning environments.
Researchers can use this toolkit to:
Replicate the core findings of the paper.
Verify how design parameters (Temporal Resolution 
Δ
t
Δt
, Sample Size 
N
N
, Reliability 
α
α
) affect statistical power.
Adapt the parameters to simulate their own specific research designs.
Repository URL: https://github.com/dsad55878-tech/DPCM-paper-reproduction.git
2. Repository Structure
code
Text
DPCM-paper-reproduction/
├── data/                   # Directory for generated simulation data (.csv)
├── results/                # Directory for analysis outputs (.csv)
├── figures/                # Directory for generated plots (.png/.pdf)
├── supplement/             # Directory for supplementary materials
├── config.json             # Main configuration file (Parameter settings)
├── requirements.txt        # Python dependencies
├── install_packages.R      # R dependency installation script
├── dpcm_core.py            # Core model logic (LTI system & observation filter)
├── generate_data.py        # Main script for generating data
├── power_analysis.R        # Main R script for mediation & power analysis
├── plot_results.R          # R script for visualization
├── verify_stability.py     # Utility to check matrix stability
└── README.md               # This documentation
3. Installation & Requirements
Python Environment
Requires Python 3.8+.
code
Bash
# Install required Python packages
pip install -r requirements.txt
(Content of requirements.txt: numpy, pandas, scipy)
R Environment
Requires R 4.1+.
code
R
# Run this in R console or terminal to install dependencies
Rscript install_packages.R
(Content of install_packages.R: install.packages(c("lavaan", "dplyr", "ggplot2", "tidyr", "boot")))
4. Usage Guide (Step-by-Step)
Step 1: Configuration
Modify config.json to set your desired simulation parameters (e.g., sample size, time intervals).
Note: The default configuration reproduces the paper's "Baseline" condition (
N
=
500
,
Δ
t
=
1
N=500,Δt=1
 week, 
α
=
0.80
α=0.80
).
Step 2: Generate Simulation Data
Run the Python script to generate latent trajectories and observe them based on your design settings.
code
Bash
python generate_data.py
Input: Reads config.json.
Output: Generates CSV files in the data/ folder (e.g., sim_data_N500_dt1_alpha0.8.csv).
Step 3: Run Statistical Analysis
Execute the R script to perform bootstrap mediation analysis and calculate statistical power.
code
Bash
Rscript power_analysis.R
Input: Reads CSV files from data/.
Output: Saves statistical results to results/power_results.csv and results/boot_effects.csv.
Step 4: Visualization
Generate the plots and supplementary tables used in the paper.
code
Bash
Rscript plot_results.R
Output: Saves figures (e.g., Power Heatmap, Attenuation Plot) to the figures/ directory.
5. Configuration File Details (config.json)
IMPORTANT: Standard JSON does not support comments (//). Please use the clean format below for your config.json file. Refer to the table below for parameter explanations.
Clean config.json Example (Copy this to your file)
code
JSON
{
  "model_parameters": {
    "coupling_matrix": {
      "diagonal": [0.80, 0.75, 0.75, 0.80],
      "off_diagonal": 0.15
    },
    "intervention_sensitivity": [0.35, 0.25, 0.20, 0.22],
    "process_noise": 0.10,
    "intervention_decay": 0.05
  },
  "design_parameters": {
    "sample_size": 500,
    "temporal_resolution": 1,
    "reliability": 0.80,
    "total_weeks": 10,
    "burn_in_steps": 20
  },
  "simulation_parameters": {
    "n_replications": 1000,
    "n_bootstrap": 1000,
    "random_seed": 42,
    "missing_data_type": "MCAR",
    "missing_rate": 0.0
  },
  "output_parameters": {
    "data_dir": "data/",
    "results_dir": "results/",
    "figures_dir": "figures/",
    "supplement_dir": "supplement/"
  }
}
Parameter Reference
Category	Parameter	Description
Model	diagonal	Autoregressive stability for Efficacy, Hope, Optimism, Resilience (0-1).
off_diagonal	Cross-lagged coupling strength between components.
intervention_sensitivity	Sensitivity of each component to the AI intervention vector.
process_noise	Variance of the process noise (Innovation term).
Design	sample_size	Number of participants (N).
temporal_resolution	Measurement interval 
Δ
t
Δt
 (weeks). 1=Weekly, 5=Pre-post.
reliability	Cronbach's 
α
α
 for adding measurement error.
Sim	n_replications	Monte Carlo iterations (Recommended: 1000).
n_bootstrap	Bootstrap resamples for mediation CIs (Recommended: 1000).
random_seed	Seed for reproducibility.
6. Output Description
Input Data Example (data/)
The generated file sim_data_*.csv contains the observed time-series:
ID	Time	Efficacy	Hope	Optimism	Resilience	Learning_Outcome
1	0	0.23	0.18	0.09	0.15	0.31
1	1	0.35	0.29	0.17	0.28	0.45
Analysis Result Example (results/)
The file power_results.csv summarizes the detectability:
Sample_Size	Temporal_Resolution	Reliability	Power	Mean_Beta	Attenuation
500	1	0.80	87.2	0.148	-1.3%
500	5	0.80	44.2	0.097	-35.3%
Power: Percentage of replications where the 95% CI excludes zero.
Attenuation: Loss of effect size compared to the high-frequency baseline.