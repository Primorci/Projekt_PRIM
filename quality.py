import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load data from the CSV file
file_path = "/Users/jureantolinc/Library/Mobile Documents/com~apple~CloudDocs/Revordings faks/4/Gyro_HitrostnaOvira1_snemanje4.csv"
gyroscope_y_values = []
with open(file_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        gyroscope_y_values.append(float(row['GyroscopeY']))
gyroscope_y_values = np.array(gyroscope_y_values)

class RealTimePlot:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time GyroscopeY Plot")

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.y_data = np.zeros(240)  # Array to store the last 240 values
        self.line, = self.ax.plot(self.y_data)
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlim(0, 240)

        self.gyroscope_y_values = gyroscope_y_values
        self.data_index = 0
        self.current_length = 0

        self.ani = FuncAnimation(self.fig, self.update, interval=1000)

    def update(self, frame):
        if self.data_index < len(self.gyroscope_y_values):
            new_data = self.gyroscope_y_values[self.data_index:self.data_index + 20]
            self.data_index += 20

            if self.current_length < 240:
                self.y_data[self.current_length:self.current_length + 20] = new_data
                self.current_length += 20
            else:
                self.y_data = np.roll(self.y_data, -20)
                self.y_data[-20:] = new_data

            self.line.set_ydata(self.y_data)
            self.ax.set_xlim(0, max(self.current_length, 240))  # Adjust x-axis limit based on current length
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimePlot(root)
    root.mainloop()
