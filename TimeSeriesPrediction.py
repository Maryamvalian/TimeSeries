import numpy as np
import matplotlib.pyplot as plt
import csv


# -------------------------------
# 1. Read Time Series Data from CSV
# -------------------------------
def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            data.append(float(row[1]))  # use the second column
    return np.array(data)


# Update filename to point to your SP500 CSV file.
filename = 'sp500_80_92.csv'
ts = read_data(filename)
t = np.arange(len(ts))


# -------------------------------
# 2. Remove Trend with a Leaky DC Filter
# -------------------------------
def leaky_dc_filter(x, alpha=0.99):
    """
    Removes trend using a leaky integrator filter.
    """
    filtered = np.zeros_like(x)
    dc = 0.0
    for i in range(len(x)):
        dc = alpha * dc + (1 - alpha) * x[i]
        filtered[i] = x[i] - dc
    return filtered


ts_detrended = leaky_dc_filter(ts, alpha=0.99)


# -------------------------------
# 3. Remove Seasonality by Folding
# -------------------------------
def remove_seasonality(ts, period):
    """
    Removes seasonality by computing the average seasonal pattern and subtracting it.
    """
    seasonal = np.zeros(period)
    counts = np.zeros(period)
    for i in range(len(ts)):
        seasonal[i % period] += ts[i]
        counts[i % period] += 1
    seasonal = seasonal / counts
    deseasonalized = np.array([ts[i] - seasonal[i % period] for i in range(len(ts))])
    return deseasonalized, seasonal


# For SP500 data, seasonality might be subtle.
# Here we assume a period (e.g., 5 for weekly patterns) – adjust as needed.
seasonal_period = 5
ts_deseasonalized, seasonal_pattern = remove_seasonality(ts_detrended, seasonal_period)

# -------------------------------
# 4. Plot Original, Trend-Removed, and Seasonality-Removed Series Together
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t, ts, label='Original', color='blue', alpha=0.7)
plt.plot(t, ts_detrended, label='Trend Removed', color='orange', alpha=0.7)
plt.plot(t, ts_deseasonalized, label='Seasonality Removed', color='green', alpha=0.7)
plt.title("SP500: Original, Trend Removed, and Seasonality Removed")
plt.xlabel("Time Index")
plt.legend()
plt.show()

# -------------------------------
# 5. Prepare Training/Test Split
# -------------------------------
# Leave the last 20 points for testing.
K = 20
train = ts_deseasonalized[:-K]
test = ts_deseasonalized[-K:]


# -------------------------------
# 6. Estimate Autocorrelation for the Training Data
# -------------------------------
def autocorrelation(x, p):
    """
    Computes autocorrelation for lags 0 to p.
    """
    n = len(x)
    r = np.zeros(p + 1)
    for lag in range(p + 1):
        r[lag] = np.sum(x[:n - lag] * x[lag:]) / n
    return r


p_order = 5  # AR model order
r = autocorrelation(train, p_order)

# -------------------------------
# 7. Solve the Yule–Walker Equations (Linear Predictive Coding)
# -------------------------------
# Build the Toeplitz autocorrelation matrix R.
R = np.empty((p_order, p_order))
for i in range(p_order):
    for j in range(p_order):
        R[i, j] = r[abs(i - j)]

# Right-hand side vector (lags 1 to p_order)
r_vector = r[1:p_order + 1]

# Solve for AR coefficients.
ar_coeffs = np.linalg.solve(R, r_vector)
print("Estimated AR coefficients:", ar_coeffs)


# -------------------------------
# 8. Forecasting Using the AR Model via Yule–Walker
# -------------------------------
def ar_forecast(train, ar_coeffs, steps):
    """
    Iterative one-step-ahead forecasting using the AR coefficients.
    The model is assumed to be: x[t] = a1*x[t-1] + a2*x[t-2] + ... + a_p*x[t-p]
    """
    p = len(ar_coeffs)
    # Initialize history with the last p values from training.
    history = list(train[-p:])
    predictions = []
    for _ in range(steps):
        # The most recent value is history[-1], then history[-2], etc.
        # Compute prediction: dot product with AR coefficients (order: a1 multiplies x[t-1], etc.)
        recent = history[-1:-p - 1:-1]  # reverse order so that recent[0] is x[t-1]
        pred = np.dot(ar_coeffs, recent)
        predictions.append(pred)
        history.append(pred)
    return np.array(predictions)


# Compute AR model predictions for the test period.
ar_predictions = ar_forecast(train, ar_coeffs, len(test))

# -------------------------------
# 9. Baseline: Prediction with Random Noise
# -------------------------------
# Generate a random forecast (white noise) with zero mean and same standard deviation as the training residual.
residual_std = np.std(train - np.mean(train))
noise_predictions = np.random.normal(0, residual_std, len(test))

# -------------------------------
# 10. Plot the Forecasts vs Actual Test Data
# -------------------------------
test_indices = np.arange(len(ts_deseasonalized) - K, len(ts_deseasonalized))

plt.figure(figsize=(12, 6))
plt.plot(test_indices, test, label='Actual Test Data', marker='o')
plt.plot(test_indices, ar_predictions, label='AR Model Predictions', marker='x')
plt.plot(test_indices, noise_predictions, label='Random Noise Baseline', marker='d')
plt.title("AR Model Forecast vs Actual Test Data and Random Noise Baseline")
plt.xlabel("Time Index")
plt.ylabel("Deseasonalized & Detrended Value")
plt.legend()
plt.show()
