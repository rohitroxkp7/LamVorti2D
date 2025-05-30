import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
#import pandas as pd

# ==== User-defined parameters ====
x_probe = 0.4      # x location to probe
y_probe = 0.0      # y location to probe
dt = 0.1           # time step size
imax = 181
jmax = 181
# ==================================

# Load grid
xg = np.loadtxt('xg.csv', delimiter=',')
yg = np.loadtxt('yg.csv', delimiter=',')

# Find closest grid point to (x_probe, y_probe)
dist = (xg - x_probe)**2 + (yg - y_probe)**2
k_min = np.argmin(dist)
i_probe, j_probe = np.unravel_index(k_min, xg.shape)

print(f'Closest point to ({x_probe:.2f}, {y_probe:.2f}) is at grid index '
      f'(i={i_probe+1}, j={j_probe+1}), x={xg[i_probe, j_probe]:.4f}, y={yg[i_probe, j_probe]:.4f}')

# Loop through each Re_*_rot folder
base_folders = sorted([f for f in os.listdir() if os.path.isdir(f) and f.startswith('Re_') and f.endswith('_rot')])

for folder_name in base_folders:
    omega_folder = os.path.join(folder_name, 'omega_data')
    omega_files = sorted(glob(os.path.join(omega_folder, 'omega_t*.csv')))

    omega_at_point = []

    for file in omega_files:
        omega_flat = np.loadtxt(file, delimiter=',')
        omega_field = omega_flat.reshape((imax, jmax))
        omega_at_point.append(omega_field[i_probe, j_probe])

    omega_at_point = np.array(omega_at_point)

    # Time vector
    t = np.arange(len(omega_files)) * dt

    # Plot time signal
    plt.figure()
    plt.plot(t, omega_at_point, 'b-', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel(f'ω(x={xg[i_probe, j_probe]:.2f}, y={yg[i_probe, j_probe]:.2f})')
    plt.title(f'Vorticity vs Time at Probe Point — {folder_name}', fontsize=12)
    plt.grid(True)
    plt.savefig(f'{folder_name}_omega_vs_time.png')
    plt.close()

    # FFT Analysis
    N = len(omega_at_point)
    omega_fft = np.fft.fft(omega_at_point)
    freq = np.fft.fftfreq(N, d=dt)
    magnitude = np.abs(omega_fft) / N

    # Keep only positive frequencies
    pos_mask = freq >= 0
    freq = freq[pos_mask]
    magnitude = magnitude[pos_mask]

    # Find dominant frequency (ignore zero frequency)
    dom_idx = np.argmax(magnitude[1:]) + 1
    dom_freq = freq[dom_idx]

    print(f'Folder: {folder_name} — Dominant frequency: {dom_freq:.4f} Hz')

    # Plot FFT
    plt.figure()
    plt.plot(freq, magnitude, 'r-', linewidth=2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title(f'FFT of Vorticity Signal — {folder_name}', fontsize=12)
    plt.grid(True)
    plt.xlim(0, freq.max())
    plt.savefig(f'{folder_name}_fft.png')
    plt.close()
