import numpy as np
import matplotlib.pyplot as plt
import csv
from statsmodels.graphics.tsaplots import plot_acf

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            data.append(float(row[1]))  # Use the second column
    return np.array(data)
#---------------------------------------------
def rem_trend(data, size, num):
    out_d = np.zeros(num)
    for n in range(num):
        tmp = 0.0
        reduce = 0

        for i in range(-size, size):
            if (n + i < 0) or (n + i >= num): # if out of bond
                reduce += 1
            else:   # valid data point
                tmp += data[n + i]

        tmp /= (2 * size + 1 - reduce)
        out_d[n] = tmp
        detrendd = np.array([])
        for i in range(len(data)):
            detrendd = np.append(detrendd, data[i] -out_d[i] )

    return detrendd
#------------------------------------------------
data = read_data("sp500_80_92.csv")
detr=rem_trend(data, 30, len(data))
plot_acf(detr)
plt.show()


fft_result = np.fft.fft(detr)
frequencies = np.fft.fftfreq(len(detr))

# Plot the magnitude of the FFT
plt.plot(frequencies, np.abs(fft_result))
plt.title("FFT of Time Series")
plt.show()