import numpy as np
import nolds
import matplotlib.pyplot as plt

# Function to generate the logistic map
def logistic_map(r, x):
    return r * x * (1 - x)

# Function to compute time series of the logistic map
def compute_series(r, x_init, n, burn_in=1000):
    x = x_init
    # Burn in period
    for _ in range(burn_in):
        x = logistic_map(r, x)
    # Compute series
    series = np.empty(n)
    for i in range(n):
        x = logistic_map(r, x)
        series[i] = x
    return series

# Function to compute Lyapunov exponent for a range of r values, using nolds
def compute_lyapunov(r_values, x_init, n, burn_in=1000):
    lyapunov_exponents = np.empty_like(r_values)
    for i, r in enumerate(r_values):
        series = compute_series(r, x_init, n, burn_in)
        lyapunov_exponents[i] = nolds.lyap_r(series)
    return lyapunov_exponents


# Parameters for the logistic map and Lyapunov computation
x_init = 0.5  # initial x value
n = 10000  # length of time series
burn_in = 1000  # burn in period
# r_values = np.linspace(3.5, 4.0, 1000)[:-1]  # range of r values
r_values = np.random.uniform(3.5, 4.0, 1000)

# Compute and plot Lyapunov exponent
lyapunov_exponents = compute_lyapunov(r_values, x_init, n, burn_in)
plt.scatter(r_values, lyapunov_exponents, s=2**2)
plt.xlabel('r')
plt.ylabel('Lyapunov exponent')
plt.grid(True)
plt.show()
