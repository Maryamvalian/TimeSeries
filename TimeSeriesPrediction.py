import numpy as np
import matplotlib.pyplot as plt
import csv

#Assignment 3:
#  first part: removing trend and seasonality
#  Second part: prediction with AR Model (test set: last 20 points)


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            data.append(float(row[1]))  # Use the second column
    return np.array(data)
#-----------------------------------------------
def compute_mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


# ----- Trend Removal----------------
def leaky_dc_filter(x, alpha=0.99):

    detrended = np.zeros_like(x)
    dc = 0.0
    for i in range(len(x)):
        dc = alpha * dc + (1 - alpha) * x[i]
        detrended[i] = x[i] - dc
    return detrended

# ----- Seasonality Removal -----
def remove_seasonality(data, period):

    pattern = np.zeros(period) #seasonality
    count = np.zeros(period)
    for i in range(len(data)):
        pattern[i % period] += data[i]
        count[i % period] += 1
    pattern /= count
    deseasonalized = np.array([data[i] - pattern[i % period] for i in range(len(data))])
    return deseasonalized, pattern


# -------------------------------------------------------
def plot_time_series(time, original, detrended, deseasonalized):
    plt.plot(time, original, label="Original", color="blue", alpha=0.7)
    plt.plot(time, detrended, label="Trend Removed", color="orange", alpha=0.7)
    plt.plot(time, deseasonalized, label="Seasonality Removed", color="green", alpha=0.7)
    plt.legend()
    plt.show()

def plot_forecast(test_time, actual, predicted, baseline):

    plt.plot(test_time, actual, label="Actual Data", marker="o", color="blue")
    plt.plot(test_time, predicted, label="AR Model Forecast", marker="x", color="red")
    plt.plot(test_time, baseline, label="Random Noise Forecast", marker="d", color="purple")
    plt.legend()
    plt.show()


# --------------------------------------------------
def compute_autocorrelation(data, max_lag):

    n = len(data)
    ac = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        ac[lag] = np.sum(data[:n - lag] * data[lag:]) / n
    return ac

#-------------------------------------------------------
def estimate_ar_coefficients(data, order):
    #Yule-Walker
    ac = compute_autocorrelation(data, order)
    R = np.empty((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = ac[abs(i - j)]
    r_vec = ac[1:order + 1]
    coefficients = np.linalg.solve(R, r_vec)
    return coefficients

#----------------------------------------------------
def forecast_ar_model(train, coeffs, steps):

    p = len(coeffs)
    history = list(train[-p:])  # Start with the last p data points(Test Set)
    predictions = []
    for _ in range(steps):
        recent = history[-p:][::-1]  # Reverse so that newest value comes first
        next_val = np.dot(coeffs, recent)
        predictions.append(next_val)
        history.append(next_val)
    return np.array(predictions)


#-------------------------------------
def run_analysis(filename):
     # First Part: remove trend and seasonality---------------------
    data = read_data(filename)
    time = np.arange(len(data))
    detrended = leaky_dc_filter(data, alpha=0.99)
    deseasonalized, _ = remove_seasonality(detrended, 5)
    plot_time_series(time, data, detrended, deseasonalized)

    # Second part: prediction---------------------------------------
    train = deseasonalized[:-20]  #all points except last 20
    test = deseasonalized[-20:]  #last 20 points
    #The order is extracted from Auto-correlation figure
    # the figure reaches to zero around 1200. to avoid overfitting Around 1000 will be good
    coeffs = estimate_ar_coefficients(train, 1000)
    print("Estimated AR coefficients:", coeffs)
    forecast_steps = len(test)
    ar_pred = forecast_ar_model(train, coeffs, forecast_steps)
    # random noise
    noise_std = np.std(train - np.mean(train))
    random_pred = np.random.normal(0, noise_std, forecast_steps)
    test_time = np.arange(len(deseasonalized) - 20, len(deseasonalized))
    plot_forecast(test_time, test, ar_pred, random_pred)

    print("MSE AR Model:", compute_mse(test, ar_pred))
    print("MSE Baseline:", compute_mse(test, random_pred))

# ----- Main -------------------
#run_analysis("earths rotation.csv") #generate error for high orders, small dataset
run_analysis("sp500_80_92.csv")

