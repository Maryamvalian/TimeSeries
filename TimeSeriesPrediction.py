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
            data.append(float(row[1]))  # second column of CSV
    return np.array(data)

# Update 'data.csv' to your actual CSV file path.
filename = 'sp500_80_92.csv'
ts = read_data(filename)
t = np.arange(len(ts))  # time axis for plotting

# Plot the original time series
plt.figure(figsize=(10, 4))
plt.plot(t, ts, label='Original Time Series')
plt.title("Original Time Series")
plt.legend()
plt.show()

# -------------------------------
# 2. Remove Trend with a Leaky DC Filter
# -------------------------------
def leaky_dc_filter(x, alpha=0.99):
    """
    Removes the trend from the signal using a leaky integrator filter.
    The parameter alpha (close to 1) controls how slowly the filter forgets past values.
    """
    filtered = np.zeros_like(x)
    dc = 0.0
    for i in range(len(x)):
        dc = alpha * dc + (1 - alpha) * x[i]
        filtered[i] = x[i] - dc
    return filtered

ts_detrended = leaky_dc_filter(ts, alpha=0.99)

# Plot the detrended series
plt.figure(figsize=(10, 4))
plt.plot(t, ts_detrended, label='Detrended Time Series')
plt.title("Time Series after Leaky DC Filter (Trend Removal)")
plt.legend()
plt.show()

# -------------------------------
# 3. Remove Seasonality by Folding
# -------------------------------
def remove_seasonality(ts, period):
    """
    Removes seasonality by computing the average seasonal pattern over the data.
    Returns the deseasonalized time series and the seasonal pattern.
    """
    seasonal = np.zeros(period)
    counts = np.zeros(period)
    for i in range(len(ts)):
        seasonal[i % period] += ts[i]
        counts[i % period] += 1
    seasonal = seasonal / counts
    deseasonalized = np.array([ts[i] - seasonal[i % period] for i in range(len(ts))])
    return deseasonalized, seasonal

# Set the seasonal period; adjust as needed for your data.
seasonal_period = 24
ts_deseasonalized, seasonal_pattern = remove_seasonality(ts_detrended, seasonal_period)

# Plot the deseasonalized series
plt.figure(figsize=(10, 4))
plt.plot(t, ts_deseasonalized, label='Deseasonalized Time Series')
plt.title("Time Series after Removing Seasonality")
plt.legend()
plt.show()

# -------------------------------
# 4. Set Up Training/Test Split
# -------------------------------
train_size = int(len(ts_deseasonalized) * 0.75)
train, test = ts_deseasonalized[:train_size], ts_deseasonalized[train_size:]

# -------------------------------
# 5. Forecasting with a Simple Linear Autoregressive Model
# -------------------------------
def create_lagged_features(data, lags):
    """
    Constructs lagged features from the time series data.
    Returns matrix X (each row contains 'lags' previous values) and the target y.
    """
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

lags = 5
X_train, y_train = create_lagged_features(train, lags)

def fit_linear_regression(X, y):
    """
    Fits a linear regression model to predict y from lagged features in X.
    Returns coefficients including an intercept (bias) term.
    """
    X_bias = np.column_stack([np.ones(X.shape[0]), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
    return coeffs

coeffs = fit_linear_regression(X_train, y_train)
print("Fitted coefficients (bias and lags):", coeffs)

def forecast(model_coeffs, history, lags, steps):
    """
    Performs iterative one-step-ahead forecasting.
    Starts with the last 'lags' values from history and forecasts 'steps' ahead.
    """
    predictions = []
    current_history = list(history[-lags:])
    for _ in range(steps):
        x_input = np.array(current_history[-lags:])
        x_bias = np.insert(x_input, 0, 1)
        pred = np.dot(model_coeffs, x_bias)
        predictions.append(pred)
        current_history.append(pred)
    return np.array(predictions)

steps = len(test)
predictions = forecast(coeffs, train, lags, steps)

# -------------------------------
# 6. Plot and Evaluate Forecasts
# -------------------------------
plt.figure(figsize=(10, 4))
plt.plot(np.arange(train_size, len(ts_deseasonalized)), test, label='Actual')
plt.plot(np.arange(train_size, len(ts_deseasonalized)), predictions, label='Forecast', linestyle='--')
plt.title("Forecast vs Actual")
plt.legend()
plt.show()

# Calculate Mean Squared Error (MSE) as an evaluation metric
mse = np.mean((test - predictions) ** 2)
print("Mean Squared Error on Test Set:", mse)
