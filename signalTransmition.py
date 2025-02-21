import numpy as np
import matplotlib.pyplot as plt

# Generate time values from 0 to 2*pi with small intervals
t = np.linspace(0, 50 * np.pi, 1000)

# Generate a sine wave signal based on time values
signal = np.sin(t)

# Create a plot
plt.plot(t, signal, label='Sine Wave')

# Label the axes and the plot
plt.title('Sine Wave Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()