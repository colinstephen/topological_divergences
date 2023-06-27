import numpy as np
import matplotlib.pyplot as plt

def logistic_map_lyapunov(r, x0=0.5, n=10000, discard=1000):
    # Initialize x with the starting value
    x = x0 * np.ones(r.shape)

    # Discard the transient
    for _ in range(discard):
        x = r * x * (1 - x)

    # Then iterate n times and compute the sum for the Lyapunov exponent
    lyapunov_exp = np.zeros(r.shape)
    for _ in range(n):
        x = r * x * (1 - x)
        lyapunov_exp += np.log(abs(r - 2 * r * x))

    # Average over the number of iterations to get the Lyapunov exponent
    lyapunov_exp /= n

    return lyapunov_exp

# Example usage:
r = np.random.uniform(3.5, 4.0, 10000)
lyapunov_exponents = logistic_map_lyapunov(r, n=100000)

plt.scatter(r, lyapunov_exponents, s=1**2)
plt.xlabel('r')
plt.ylabel('Lyapunov exponent')
plt.grid(True)
plt.show()

