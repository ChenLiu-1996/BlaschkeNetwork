import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

import os
import sys
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from analytical.analytical_decomposition import complexify_signal


def generate_signal(length=256, smooth_sigma=5, noise_scale=2.0, random_seed=1):
    np.random.seed(random_seed)
    noise = np.random.randn(length) * noise_scale
    smooth_signal = gaussian_filter1d(noise, sigma=smooth_sigma)
    return smooth_signal


if __name__ == '__main__':
    num_signals = 3
    signals = np.array([generate_signal(random_seed=3*i+1)/2 for i in range(num_signals)])

    seq_len = len(signals[0])
    time = np.arange(seq_len) / seq_len
    signal_complex = complexify_signal(signals)

    blues = ['#0d47a1',         # deep cobalt blue
             '#1976d2',         # vivid steel blue
             '#42a5f5']         # cooler sky blue
    reds = ['#b71c1c',  # deep crimson red
            '#e53935',  # strong, bright red
            '#f06292']  # vibrant pinkish red

    fig = plt.figure(figsize=(24, 8))
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    for i in range(num_signals):
        ax.plot(time, signals[i], zs=i, zdir='y', color=blues[i], alpha=0.5, linewidth=4)
        ax.set_zlim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=35, azim=-60)
    origin = (-0.15, -0.3, min(s.min() for s in signals))
    ax.quiver(*origin, 1.4, 0, 0, color='black', linewidth=2, arrow_length_ratio=0.05)
    ax.quiver(*origin, 0, 3.8, 0, color='black', linewidth=2, arrow_length_ratio=0.05)
    ax.quiver(*origin, 0, 0, 1.4, color='black', linewidth=2, arrow_length_ratio=0.05)
    ax.text(origin[0] + 1.5, origin[1], origin[2], 'Time', fontsize=20)
    ax.text(origin[0], origin[1] + 3.9, origin[2], 'Channel', fontsize=20)
    ax.text(origin[0], origin[1], origin[2] + 1.5, 'Amplitude', fontsize=20)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    for i in range(num_signals):
        ax.plot(time, np.real(signal_complex[i]), zs=i, zdir='y', color=blues[i], alpha=0.5, linewidth=4)
        ax.plot(time, np.imag(signal_complex[i]), zs=i, zdir='y', color=reds[i], alpha=0.5, linewidth=4)
        ax.set_zlim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=35, azim=-60)
    origin = (-0.15, -0.3, min(np.real(s).min() for s in signal_complex))
    ax.quiver(*origin, 1.4, 0, 0, color='black', linewidth=2, arrow_length_ratio=0.05)
    ax.quiver(*origin, 0, 3.8, 0, color='black', linewidth=2, arrow_length_ratio=0.05)
    ax.quiver(*origin, 0, 0, 1.4, color='black', linewidth=2, arrow_length_ratio=0.05)
    ax.text(origin[0] + 1.5, origin[1], origin[2], 'Time', fontsize=20)
    ax.text(origin[0], origin[1] + 3.9, origin[2], 'Channel', fontsize=20)
    ax.text(origin[0], origin[1], origin[2] + 1.5, 'Amplitude', fontsize=20)

    ax.plot([], [], color=blues[i], alpha=0.5, linewidth=4, label='Real part')
    ax.plot([], [], color=reds[i], alpha=0.5, linewidth=4, label='Imaginary part')
    ax.legend(loc='lower left', fontsize=20)

    for i in range(num_signals):
        for j in range(4):
            ax = fig.add_subplot(6, 12, 24*i+j+9)
            ax.plot(time[64*j:64*(j+1)], np.real(signal_complex[i])[64*j:64*(j+1)], color=blues[i], alpha=0.5, linewidth=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            ax = fig.add_subplot(6, 12, 24*i+12+j+9)
            ax.plot(time[64*j:64*(j+1)], np.imag(signal_complex[i])[64*j:64*(j+1)], color=reds[i], alpha=0.5, linewidth=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(2)

    fig.tight_layout(pad=2)
    fig.savefig('signal.png', dpi=300)

