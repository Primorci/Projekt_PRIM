import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import csv
import os

file_path = "/Users/jureantolinc/Library/Mobile Documents/com~apple~CloudDocs/Revordings faks/3/Gyro_StaraCesta1_snemanje3.csv"

gyroscope_y_values = []
with open(file_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        gyroscope_y_values.append(float(row['GyroscopeY']))

gyroscope_y_values = np.array(gyroscope_y_values)

def transform(data):
    augmented_data = data.copy()

    num_transformations = np.random.randint(2, 6)

    transformations = np.random.choice([flip, smooth, add_noise, convolve_signal], num_transformations, replace=True)

    # Apply the selected transformations
    for transform_func in transformations:
        augmented_data = transform_func(augmented_data)

    return augmented_data

def flip(data):
    return np.flip(data)

def smooth(data):
    window = np.hamming(5)
    smoothed_data = np.convolve(data, window, mode='same') / sum(window)
    return smoothed_data

def add_noise(data):
    noise = np.random.normal(0, 0.1, data.shape)
    noisy_data = data + noise
    return noisy_data

def convolve_signal(data):
    kernel = np.array([1, -1])
    convolved_data = convolve(data, kernel, mode='same')
    return convolved_data

def save_plot(data, output_path, ylim):
    plt.figure(figsize=(8, 6))
    plt.plot(data)
    plt.title('Augmented Signal')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.ylim(ylim)
    plt.savefig(output_path)
    plt.close()

def main():
    num_samples = 100
    output_dir = "augmented_data"
    os.makedirs(output_dir, exist_ok=True)

    # Determine y-axis limits based on the original signal
    ylim = (np.min(gyroscope_y_values), np.max(gyroscope_y_values))

    for i in range(num_samples):
        augmented_signal = transform(gyroscope_y_values)
        
        # Save augmented data plot
        output_path = os.path.join(output_dir, f'data_sample_{i}.png')
        save_plot(augmented_signal, output_path, ylim)

if __name__ == "__main__":
    main()
