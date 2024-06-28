import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from scipy.signal import find_peaks
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
        
        self.y_data = np.zeros(100)
        self.line, = self.ax.plot(self.y_data)
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlim(0, 100)
        
        self.y_limit = 5
        self.exceeds_yellow_threshold = False
        self.exceeds_red_threshold = False

        self.gyroscope_y_values = gyroscope_y_values
        self.data_index = 0

        self.ani = FuncAnimation(self.fig, self.update, interval=50)
    
    def update(self, frame):
        if self.data_index < len(self.gyroscope_y_values):
            self.y_data = np.roll(self.y_data, -1)
            self.y_data[-1] = self.gyroscope_y_values[self.data_index]
            self.data_index += 1
            
            self.line.set_ydata(self.y_data)
            self.ax.collections.clear()

            for i in range(0, 100, 20):
                segment = self.y_data[i:i+20]
                peaks, _ = find_peaks(np.abs(segment), height=3)
                
                if any(np.abs(segment) > 2):
                    self.ax.axvspan(i, i+20, color='red', alpha=0.2)
                elif np.any(np.abs(segment) >= 0.25):
                    self.ax.axvspan(i, i+20, color='yellow', alpha=0.2)
                else:
                    self.ax.axvspan(i, i+20, color='green', alpha=0.2)

            if np.any(np.abs(self.y_data) >= 2):
                self.exceeds_red_threshold = True
                self.ax.axhline(2, color='red', linestyle='--', alpha=0.8)
                self.ax.axhline(-2, color='red', linestyle='--', alpha=0.8)
                self.ax.text(0, 2, 'Slabo', color='red')
                self.ax.text(0, -2, 'Slabo', color='red')

            if np.any(np.abs(self.y_data) >= 0.25):
                self.exceeds_yellow_threshold = True
                self.ax.axhline(0.25, color='yellow', linestyle='--', alpha=0.8)
                self.ax.axhline(-0.25, color='yellow', linestyle='--', alpha=0.8)
                self.ax.text(0, 0.3, 'Vredu', color='orange')
                self.ax.text(0, -0.3, 'Vredu', color='orange')

            self.ax.axhline(0, color='green', linestyle='--', alpha=0.8)
            self.ax.text(0, 0, 'Dobro', color='green')

            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimePlot(root)
    root.mainloop()
