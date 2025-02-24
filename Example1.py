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

filename = 'sp500_80_92.csv'  # Update with your CSV file path.
ts = read_data(filename)
t = np.arange(len(ts))  # time axis for plotting

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

# -------------------------------
# 4. Plot Original, Detrended, and Deseasonalized Series Together
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t, ts, label='Original', color='blue')
plt.plot(t, ts_detrended, label='Trend Removed', color='orange')
plt.plot(t, ts_deseasonalized, label='Seasonality Removed', color='green')
plt.title("Original, Trend Removed, and Seasonality Removed Time Series")
plt.xlabel("Time")
plt.legend()
plt.show()
