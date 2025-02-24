from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt

# Define the input signal
x = [2, 3, 5, 7]

# Compute the FFT using scipy.fft
fft_result = fft(x)
print(fft_result,'\n')
# Print FFT results
print("Only magnitudes",np.abs(fft_result))
