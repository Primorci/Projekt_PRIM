import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection

# Load data from the CSV file
file_path = "/Users/jureantolinc/Library/Mobile Documents/com~apple~CloudDocs/Revordings faks/4/Gyro_Makedam1_snemanje4.csv"
gyroscope_y_values = []
timestamps = []
with open(file_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        timestamps.append(float(row['Timestamp']))
        gyroscope_y_values.append(float(row['GyroscopeY']))
timestamps = np.array(timestamps)
gyroscope_y_values = np.array(gyroscope_y_values)

class RealTimePlot:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time GyroscopeY Plot")
        self.root.geometry("1200x800")  # Set the window size to 1200x800 pixels

        self.fig, self.ax = plt.subplots(facecolor='white')  # Set figure background to white
        self.ax.tick_params(colors='black', which='both')  # Change the color of the ticks
        self.ax.spines['top'].set_color('black')
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['left'].set_color('black')
        self.ax.spines['right'].set_color('black')
        self.ax.xaxis.label.set_color('black')
        self.ax.yaxis.label.set_color('black')
        self.ax.title.set_color('black')
        self.ax.yaxis.set_tick_params(labelcolor='black')
        self.ax.xaxis.set_tick_params(labelcolor='black')
        self.ax.set_xlabel("Time (seconds)", color='black')  # Add x-axis label
        self.ax.set_ylabel("Gyroscope Y", color='black') 

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Make the plot fill the window

        self.gyroscope_y_values = gyroscope_y_values
        self.timestamps = np.arange(len(gyroscope_y_values)) / 20.0  # Convert to seconds
        self.num_values = len(gyroscope_y_values)
        self.num_segments = self.num_values // 20
        self.colors = ['green'] * self.num_segments  # Start with all green colors
        self.lines = []
        self.window_size = 10  # Time window size in seconds
        self.ani = FuncAnimation(self.fig, self.update, frames=range(0, self.num_segments), interval=1000)

        self.debug_mode = tk.BooleanVar()  # Flag to indicate debug mode status

        # Add debug checkbox
        self.debug_checkbox = tk.Checkbutton(self.root, text="Debug", variable=self.debug_mode, command=self.update_debug)
        self.debug_checkbox.pack()

        # Initialize debug features as None
        self.red_lines = None
        self.yellow_lines = None

    def update_debug(self):
        self.update(None)

    def update(self, frame):
        if frame is not None:
            if frame == 0:
                start_idx = 0
            else:
                start_idx = frame * 20 - 1  # Start from the last element of the previous segment
            end_idx = (frame + 1) * 20
            segment_data = self.gyroscope_y_values[start_idx:end_idx]
            segment_time = self.timestamps[start_idx:end_idx]

            # Change color based on thresholds for this segment
            color = 'green'
            if np.any(np.abs(segment_data) >= 2):
                color = 'red'
            elif np.any(np.abs(segment_data) >= 0.25):
                color = 'yellow'

            # Create segments with their respective colors
            points = np.array([segment_time, segment_data]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=color, linewidths=3, linestyle='solid')  # Thinner lines
            self.ax.add_collection(lc)

            self.ax.set_ylim(-5, 5)

            # Update the x-axis limits to create a scrolling effect
            current_time = self.timestamps[end_idx - 1]
            self.ax.set_xlim(max(0, current_time - self.window_size), current_time)

            # Set the x-axis ticks to show time in seconds with 1-second intervals
            self.ax.set_xticks(np.arange(max(0, current_time - self.window_size), current_time + 1, 1))
            self.ax.set_xticklabels(np.arange(max(0, current_time - self.window_size), current_time + 1, 1).astype(int))

        if self.debug_mode.get():
            if self.red_lines is not None:
                for line in self.red_lines:
                    line.remove()
            if self.yellow_lines is not None:
                for line in self.yellow_lines:
                    line.remove()
                    
            # Draw dashed lines for red threshold
            self.red_lines = [self.ax.axhline(2, color='red', linestyle='--', alpha=0.8),
                              self.ax.axhline(-2, color='red', linestyle='--', alpha=0.8)]
            # Draw dashed lines for yellow threshold
            self.yellow_lines = [self.ax.axhline(0.25, color='yellow', linestyle='--', alpha=0.8),
                                 self.ax.axhline(-0.25, color='yellow', linestyle='--', alpha=0.8)]
        else:
            if self.red_lines is not None:
                for line in self.red_lines:
                    line.remove()
                self.red_lines = None
            if self.yellow_lines is not None:
                for line in self.yellow_lines:
                    line.remove()
                self.yellow_lines = None

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimePlot(root)
    root.mainloop()
