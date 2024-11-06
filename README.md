# Geometric Brownian Motion Simulator

This project provides a simulator for Geometric Brownian Motion (GBM) and Ornstein-Uhlenbeck (OU) processes.

## Features

- Simulate paths for GBM and OU processes.
- Plot and save the simulated paths.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

## Usage

Run the simulator from the command line:

### Geometric Brownian Motion (GBM)


## Arguments

- `--process`: Type of stochastic process (`gbm` or `ou`).
- `--y0`: Initial value \( Y(0) \).
- `--mu`: Drift coefficient (GBM) or long-term mean (OU).
- `--sigma`: Volatility coefficient.
- `--theta`: Mean reversion speed (OU only, if applicable).
- `--T`: Total time horizon.
- `--N`: Number of time steps.
- `--method`: Simulation method (`analytical`, `euler`, or `milstein`).
- `--output`: Output image file for the plot.

## Example usage
- pygbm simulate --y0 1.0 --mu 0.05 --sigma 0.2 --T 1.0 --N 100 --method analytical --output output_analytical.png
- pygbm simulate --y0 1.0 --mu 0.05 --sigma 0.2 --T 1.0 --N 100 --method euler --output output_euler.png
- pygbm simulate --y0 1.0 --mu 0.05 --sigma 0.2 --T 1.0 --N 100 --method milstein --output output_milstein.png
