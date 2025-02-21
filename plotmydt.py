import matplotlib.pyplot as plt

# Data for plotting
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 8, 7]

# Create the plot
plt.plot(x, y, label='Line 1', marker='o')

# Add title and labels
plt.title('Simple Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Add legend
plt.legend()

# Display the plot
plt.show()
