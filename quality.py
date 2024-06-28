import csv
import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/jureantolinc/Library/Mobile Documents/com~apple~CloudDocs/Revordings faks/untitled.csv"

# Open the CSV file and read the 'GyroscopeY' column into an array
gyroscope_y_values = []
with open(file_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        gyroscope_y_values.append(float(row['GyroscopeY']))

# Convert the array to numpy array
gyroscope_y_values = np.array(gyroscope_y_values)

# Create a sine wave from the gyroscope_y_values
time = np.arange(0, len(gyroscope_y_values))  # Assuming time starts from 0
frequency = 0.1  # Adjust this value as needed
sin_wave = np.sin(2 * np.pi * frequency * time)

# Plot the original data and the sine wave
plt.plot(gyroscope_y_values, label='Original Data')
plt.plot(sin_wave, label='Sine Wave')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original Data vs Sine Wave')
plt.legend()
plt.show()
