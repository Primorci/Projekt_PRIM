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

# Adjust time axis
time = np.arange(0, len(gyroscope_y_values)) / 20.0  # assuming 20 Hz frequency

# Plot the original data with positive and negative values
plt.plot(time, np.abs(gyroscope_y_values))
plt.xlabel('Time (s)')
plt.ylabel('|GyroscopeY| Value')
plt.title('Original GyroscopeY Data (Positive and Negative Values)')
plt.show()
