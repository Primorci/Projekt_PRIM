import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import find_peaks

file_path = "/Users/jureantolinc/Library/Mobile Documents/com~apple~CloudDocs/Revordings faks/4/Gyro_HitrostnaOvira1_snemanje4.csv"

gyroscope_y_values = []
with open(file_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        gyroscope_y_values.append(float(row['GyroscopeY']))

gyroscope_y_values = np.array(gyroscope_y_values)

time = np.arange(0, len(gyroscope_y_values)) / 20.0  # assuming 20 Hz frequency

y_limit = max(abs(gyroscope_y_values) + 0.2)
plt.ylim(-y_limit, y_limit)

num_samples = len(time)
num_seconds = num_samples // 20  # assuming 20 Hz frequency

# Set x-axis ticks at every 2 seconds
plt.xticks(np.arange(0, num_seconds+1, step=2))

# Plot the gyroscope data
plt.plot(time, gyroscope_y_values)
plt.xlabel('Time (seconds)')
plt.ylabel('GyroscopeY Value')
plt.title('Original GyroscopeY Data')

# Check for values exceeding thresholds
exceeds_yellow_threshold = np.any(np.abs(gyroscope_y_values) >= 0.25)
exceeds_red_threshold = np.any(np.abs(gyroscope_y_values) >= 2)

# Draw background spans based on peaks within 2-second intervals
for i in range(0, num_seconds, 2):
    start_idx = i * 20
    end_idx = (i + 2) * 20
    
    # Find peaks within the interval
    peaks, _ = find_peaks(np.abs(gyroscope_y_values[start_idx:end_idx]), height=3)
    num_peaks = len(peaks)
    
    # Classify the interval based on the number of peaks
    if any(np.abs(gyroscope_y_values[start_idx:end_idx]) > 2):
        # If there are any peaks > 2
        plt.axvspan(i, i+2, color='red', alpha=0.3)
    elif np.any(np.abs(gyroscope_y_values[start_idx:end_idx]) >= 0.25):
        # If there's at least one peak with value 0.25 or higher
        plt.axvspan(i, i+2, color='yellow', alpha=0.3)
    else:
        # Otherwise, it's green
        plt.axvspan(i, i+2, color='green', alpha=0.3)

# Conditionally plot threshold lines and labels
if exceeds_red_threshold:
    plt.axhline(2, color='red', linestyle='--', alpha=0.4)
    plt.axhline(-2, color='red', linestyle='--', alpha=0.4)
    plt.text(0, 2, 'Slabo', color='red')
    plt.text(0, -2, 'Slabo', color='red')

if exceeds_yellow_threshold:
    plt.axhline(0.25, color='yellow', linestyle='--', alpha=0.4)
    plt.axhline(-0.25, color='yellow', linestyle='--', alpha=0.4)
    plt.text(0, 0.3, 'Vredu', color='orange')
    plt.text(0, -0.3, 'Vredu', color='orange')

# Always plot the green lines and labels
plt.axhline(0, color='green', linestyle='--', alpha=0.4)
plt.text(0, 0, 'Dobro', color='green')

plt.show()
