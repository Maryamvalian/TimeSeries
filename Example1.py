from scipy.fft import fft
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters
mu = 10  # Mean
sigma = 2  # Standard deviation

# Generate x values
x_values = np.linspace(4, 16, 100)
y_values = norm.pdf(x_values, mu, sigma)

# Plot the Gaussian distribution
plt.plot(x_values, y_values, label="Normal Distribution ")
plt.scatter([10, 12, 14], norm.pdf([10, 12, 14], mu, sigma), color='red', label="Computed Points")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.title("Gaussian (Normal) Distribution")
plt.grid()
plt.show()

