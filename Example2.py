import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
num_sensors = 3  # Number of MEG sensors
num_sources = 5  # Number of brain sources
num_samples = 100  # Number of time samples

# Step 2: Generate random empirical covariance of sensor data (C_B)
np.random.seed(42)
C_B_empirical = np.random.rand(num_sensors, num_sensors)
C_B_empirical = (C_B_empirical + C_B_empirical.T) / 2  # Symmetric matrix
np.fill_diagonal(C_B_empirical, np.abs(np.diag(C_B_empirical)) + 1)  # Ensure positive-definiteness

# Step 3: Generate leadfield matrix (L)
L = np.random.randn(num_sensors, num_sources)

# Step 4: Define noise covariance (Σ_ε)
Sigma_epsilon = np.eye(num_sensors) * 0.2  # Small noise variance

# Step 5: Initialize prior covariance C (identity matrix)
C_prior = np.eye(num_sources)

# Step 6: Compute initial modeled covariance Σ_B = Σ_ε + L C L^T
Sigma_B = Sigma_epsilon + L @ C_prior @ L.T

# Step 7: Iterative optimization of C using ML-II with regularization and smoothing
log_marginal_likelihoods = []
alpha = 0.9  # Smoothing factor
lambda_reg = 1e-4  # Regularization factor

for _ in range(20):  # 20 Iterations to optimize C
    # Compute regularized inverse of Σ_B
    Sigma_B_inv = np.linalg.inv(Sigma_B + lambda_reg * np.eye(num_sensors))

    # Compute log-marginal likelihood function
    log_marginal_likelihood = np.trace(C_B_empirical @ Sigma_B_inv) + np.log(np.linalg.det(Sigma_B))
    log_marginal_likelihoods.append(log_marginal_likelihood)

    # Compute updated C with smoothing
    C_new = np.linalg.pinv(L) @ (C_B_empirical - Sigma_epsilon) @ np.linalg.pinv(L.T)
    C_prior = alpha * C_prior + (1 - alpha) * C_new  # Smoothed update

    # Recompute Σ_B with updated C_prior
    Sigma_B = Sigma_epsilon + L @ C_prior @ L.T

# Step 8: Display optimization results
plt.plot(log_marginal_likelihoods, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Log Marginal Likelihood')
plt.title('ML-II Optimization of Source Covariance (C)')
plt.grid()
plt.show()

# Display final estimated C
print("Estimated Source Covariance (C):")
for i in range(num_sources):
    print(f"Source {i + 1}: {C_prior[i]}")
