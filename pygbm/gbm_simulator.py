#!/usr/bin/env python
"""A simulator for Geometric Brownian Motion (GBM) and its variants."""

from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class BaseSimulator(ABC):
    """Base class for stochastic process simulators."""
    
    def __init__(self, y0: float) -> None:
        """Initialize the base simulator.
        
        Args:
            y0: Initial value Y(0)
        """
        self.y0 = y0

    @abstractmethod
    def simulate_path(self, T: float, N: int, method: str = 'analytical') -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a single path of the stochastic process.
        
        Args:
            T: Total time horizon
            N: Number of time steps
            method: Method to use for simulation ('analytical', 'euler', 'milstein')
            
        Returns:
            Tuple containing:
                - Array of time points
                - Array of simulated values
        """
        pass

class GBMSimulator(BaseSimulator):
    """A simulator for Geometric Brownian Motion (GBM).
    
    The GBM follows the stochastic differential equation:
    dY(t) = μY(t)dt + σY(t)dB(t)
    
    where:
    - Y(t) is the value at time t
    - μ (mu) is the drift coefficient (expected return)
    - σ (sigma) is the volatility/diffusion coefficient
    - B(t) is a Brownian motion
    """
    
    def __init__(self, y0: float, mu: float, sigma: float) -> None:
        """Initialize the GBM simulator.
        
        Args:
            y0: Initial value Y(0)
            mu: Drift coefficient (expected return)
            sigma: Volatility/diffusion coefficient
        """
        super().__init__(y0)
        self.mu = mu
        self.sigma = sigma

    def simulate_path(self, T: float, N: int, method: str = 'analytical') -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a single path of the GBM.
        
        Args:
            T: Total time horizon
            N: Number of time steps
            method: Method to use for simulation ('analytical', 'euler', 'milstein')
            
        Returns:
            Tuple containing:
                - Array of time points
                - Array of simulated values
        """
        dt = T / N
        t_values = np.linspace(0, T, N+1)
        y_values = np.zeros(N+1)
        y_values[0] = self.y0

        if method == 'analytical':
            for i in range(1, N+1):
                dB = np.random.normal(0, np.sqrt(dt))
                y_values[i] = y_values[i-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dB)
        elif method == 'euler':
            for i in range(1, N+1):
                dB = np.random.normal(0, np.sqrt(dt))
                y_values[i] = y_values[i-1] + self.mu * y_values[i-1] * dt + self.sigma * y_values[i-1] * dB
        elif method == 'milstein':
            for i in range(1, N+1):
                dB = np.random.normal(0, np.sqrt(dt))
                y_values[i] = y_values[i-1] + self.mu * y_values[i-1] * dt + self.sigma * y_values[i-1] * dB + 0.5 * self.sigma**2 * y_values[i-1] * (dB**2 - dt)
        else:
            raise ValueError("Invalid method. Choose from 'analytical', 'euler', or 'milstein'.")

        return t_values, y_values

class OUSimulator(BaseSimulator):
    """A simulator for Ornstein-Uhlenbeck process.
    
    The OU process follows the stochastic differential equation:
    dY(t) = θ(μ - Y(t))dt + σdB(t)
    
    where:
    - Y(t) is the value at time t
    - θ (theta) is the mean reversion speed
    - μ (mu) is the long-term mean
    - σ (sigma) is the volatility
    - B(t) is a Brownian motion
    """
    
    def __init__(self, y0: float, theta: float, mu: float, sigma: float) -> None:
        """Initialize the OU simulator.
        
        Args:
            y0: Initial value Y(0)
            theta: Mean reversion speed
            mu: Long-term mean
            sigma: Volatility coefficient
        """
        super().__init__(y0)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def simulate_path(self, T: float, N: int, method: str = 'analytical') -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a single path of the OU process.
        
        Args:
            T: Total time horizon
            N: Number of time steps
            method: Method to use for simulation ('analytical', 'euler', 'milstein')
            
        Returns:
            Tuple containing:
                - Array of time points
                - Array of simulated values
        """
        dt = T / N
        t_values = np.linspace(0, T, N+1)
        y_values = np.zeros(N+1)
        y_values[0] = self.y0

        if method == 'analytical':
            for i in range(1, N+1):
                dB = np.random.normal(0, np.sqrt(dt))
                y_values[i] = y_values[i-1] * np.exp(-self.theta * dt) + self.mu * (1 - np.exp(-self.theta * dt)) + self.sigma * np.sqrt((1 - np.exp(-2 * self.theta * dt)) / (2 * self.theta)) * dB
        elif method == 'euler':
            for i in range(1, N+1):
                dB = np.random.normal(0, np.sqrt(dt))
                y_values[i] = y_values[i-1] + self.theta * (self.mu - y_values[i-1]) * dt + self.sigma * dB
        elif method == 'milstein':
            for i in range(1, N+1):
                dB = np.random.normal(0, np.sqrt(dt))
                y_values[i] = y_values[i-1] + self.theta * (self.mu - y_values[i-1]) * dt + self.sigma * dB + 0.5 * self.sigma**2 * (dB**2 - dt)
        else:
            raise ValueError("Invalid method. Choose from 'analytical', 'euler', or 'milstein'.")

        return t_values, y_values

def main() -> None:
    """Run the stochastic process simulation with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate stochastic processes.")
    parser.add_argument("simulate", type=str, help="Simulate command")
    parser.add_argument("--y0", type=float, required=True, help="Initial value Y(0)")
    parser.add_argument("--mu", type=float, required=True, help="Drift coefficient (GBM) or long-term mean (OU)")
    parser.add_argument("--sigma", type=float, required=True, help="Volatility coefficient")
    parser.add_argument("--theta", type=float, help="Mean reversion speed (OU only)")
    parser.add_argument("--T", type=float, required=True, help="Total time")
    parser.add_argument("--N", type=int, required=True, help="Number of time steps")
    parser.add_argument("--method", type=str, default='analytical', help="Method to use for simulation ('analytical', 'euler', 'milstein')")
    parser.add_argument("--output", type=str, help="Output image file for the plot")

    args = parser.parse_args()

    if args.simulate != "simulate":
        parser.error("The first argument must be 'simulate'")

    if args.theta is not None:
        simulator = OUSimulator(args.y0, args.theta, args.mu, args.sigma)
    else:
        simulator = GBMSimulator(args.y0, args.mu, args.sigma)

    t_values, y_values = simulator.simulate_path(args.T, args.N, args.method)

    plt.plot(t_values, y_values, label=f"{'OU' if args.theta else 'GBM'} Path ({args.method})")
    plt.xlabel("Time")
    plt.ylabel("Y(t)")
    plt.title(f"Simulated {'OU' if args.theta else 'GBM'} Path ({args.method})")
    plt.legend()

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == "__main__":
    main()

# Example of running the script from the command line for each method:
# Analytical method:
# pygbm simulate --y0 1.0 --mu 0.05 --sigma 0.2 --T 1.0 --N 100 --method analytical --output output_analytical.png
# Euler method:
# pygbm simulate --y0 1.0 --mu 0.05 --sigma 0.2 --T 1.0 --N 100 --method euler --output output_euler.png
# Milstein method:
# pygbm simulate --y0 1.0 --mu 0.05 --sigma 0.2 --T 1.0 --N 100 --method milstein --output output_milstein.png
