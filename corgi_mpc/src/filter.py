import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation


class LowPassFilter:
    def __init__(self, cutoff, fs, order=5):
        self.fs = fs
        self.cutoff = cutoff
        self.order = order
        self.b, self.a = self.butter_lowpass(cutoff, fs, order)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)

    def butter_lowpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def apply_filter(self, sample):
        filtered, self.zi = lfilter(self.b, self.a, [sample], zi=self.zi)
        return filtered[0]

    def reset_filter(self):
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)


class RealTimePlot:
    def __init__(self):
        self.raw_x_data = deque(maxlen=200)
        self.filtered_x_data = deque(maxlen=200)
        self.raw_z_data = deque(maxlen=200)
        self.filtered_z_data = deque(maxlen=200)
        self.init_plot()

    def init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(4, 1)
        (self.raw_x_line,) = self.ax[0].plot([], [], label="Raw X", color="blue")
        (self.filtered_x_line,) = self.ax[1].plot([], [], label="Filtered X", color="green")
        (self.raw_z_line,) = self.ax[2].plot([], [], label="Raw Z", color="red")
        (self.filtered_z_line,) = self.ax[3].plot([], [], label="Filtered Z", color="orange")
        for axis in self.ax:
            axis.set_xlim(0, 200)
            axis.set_ylim(-3, 3)
            axis.legend()
            axis.grid()

        self.ax[0].set_title("Raw X Data")
        self.ax[1].set_title("Filtered X Data")
        self.ax[2].set_title("Raw Z Data")
        self.ax[3].set_title("Filtered Z Data")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

    def update_plot(self, raw_x, filtered_x, raw_z, filtered_z):
        self.raw_x_data.append(raw_x)
        self.filtered_x_data.append(filtered_x)
        self.raw_z_data.append(raw_z)
        self.filtered_z_data.append(filtered_z)
        self.raw_x_line.set_data(np.arange(len(self.raw_x_data)), list(self.raw_x_data))
        self.filtered_x_line.set_data(np.arange(len(self.filtered_x_data)), list(self.filtered_x_data))
        self.raw_z_line.set_data(np.arange(len(self.raw_z_data)), list(self.raw_z_data))
        self.filtered_z_line.set_data(np.arange(len(self.filtered_z_data)), list(self.filtered_z_data))
        self.ax[0].set_xlim(0, len(self.raw_x_data))
        self.ax[1].set_xlim(0, len(self.filtered_x_data))
        self.ax[2].set_xlim(0, len(self.raw_z_data))
        self.ax[3].set_xlim(0, len(self.filtered_z_data))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finish_plot(self):
        plt.ioff()
        plt.show()


# Example usage:
if __name__ == "__main__":
    fs = 1000.0  # Sample rate, Hz
    cutoff = 40.0  # Desired cutoff frequency of the filter, Hz
    processor = LowPassFilter(cutoff, fs, order=6)

    # Generate a sample signal (for simulation)
    T = 1.0  # duration in seconds
    n = int(T * fs)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)  # time vector

    # Simulate processing samples in real-time
    for i in range(n):
        sample = np.sin(1.2 * 2 * np.pi * t[i]) + 1.5 * np.cos(9 * 2 * np.pi * t[i]) + np.random.normal(0, 0.5)
        filtered_sample = processor.apply_filter(sample)
        print(f"Original: {sample:.5f}, Filtered: {filtered_sample:.5f}")
