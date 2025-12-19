# Dynamic PsyCap Coupling Model (DPCM) Simulation

This repository contains the replication code for the study:
**"Why Do Psychological Effects Flicker? Unveiling Design-Dependent Detectability in AI-Mediated Learning"**.

## Overview
The scripts simulate how sampling frequency ($\Delta t$) and measurement reliability ($\alpha$) interact to affect the empirical detectability of psychological mechanisms in high-velocity AI learning environments.

## Repository Structure
- `dpcm_core.py`: Core logic for the stochastic differential equation model and measurement filter.
- `main.py`: Main execution script to run the Monte Carlo experiments across the design grid.
- `config.json`: Configuration file defining model matrices and experimental parameters.
- `requirements.txt`: Python dependencies.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt