import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the image table
# 'Number of coins' will be on the x-axis
number_of_coins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 'Output Voltage (V)' will be on the y-axis
output_voltage = np.array([0, 0.8, 1.5, 2.0, 2.9, 3.3, 4.0, 4.9, 5.2, 6.0, 7.0])

# --- Plotting the Data ---

# Create the figure and axes for the plot
plt.figure(figsize=(10, 6)) # Set the size of the plot

# Plot the data points as a line ('-') with markers ('o')
plt.plot(number_of_coins, output_voltage, marker='o', linestyle='-', color='blue', label='Voltage vs. Coins')

# Add labels and title
plt.xlabel('Number of Coins', fontsize=14)
plt.ylabel('Output Voltage (V)', fontsize=14)
plt.title('Output Voltage vs. Number of Coins', fontsize=16)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Add legend
plt.legend()

# Highlight the data points with labels (optional but helpful)
for i, (x, y) in enumerate(zip(number_of_coins, output_voltage)):
    plt.annotate(f'{y}V', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

# Display the plot
plt.show()

# --- Optional: Print the Data for verification ---
print("\nData used for plotting:")
for coins, voltage in zip(number_of_coins, output_voltage):
    print(f"Coins: {coins}, Voltage: {voltage} V")