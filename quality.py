import numpy as np
import matplotlib.pyplot as plt
import csv

file_path = "/Users/jureantolinc/Library/Mobile Documents/com~apple~CloudDocs/Revordings faks/3/Gyro_AvtoCesta1_snemanje3.csv"

gyroscope_y_values = []
with open(file_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        gyroscope_y_values.append(float(row['GyroscopeY']))

gyroscope_y_values = np.array(gyroscope_y_values)

time = np.arange(0, len(gyroscope_y_values)) / 20.0  # assuming 20 Hz frequency

y_limit = max(abs(gyroscope_y_values) + 0.2)
plt.ylim(-y_limit, y_limit)

# Set x-axis ticks at each integer multiple of one second
num_samples = len(time)
num_seconds = num_samples // 20  # assuming 20 Hz frequency
plt.xticks(np.arange(0, num_seconds + 1, step=1))

plt.plot(time, gyroscope_y_values)
plt.xlabel('Time (seconds)')
plt.ylabel('GyroscopeY Value')
plt.title('Original GyroscopeY Data')
plt.show()
