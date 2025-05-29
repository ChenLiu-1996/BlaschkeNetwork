import os
import numpy as np
from tqdm import tqdm


def generate_signal(num_freqs, length=5000):
    t = np.linspace(0, length, length)
    signal = np.zeros_like(t)
    for _ in range(num_freqs):
        freq = np.random.uniform(1/length, 10/length)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.5, 1.5)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    return signal.astype(np.float32)

def simulate_and_save(npz_path: str, num_samples=100):
    signals = []
    labels = []
    for _ in tqdm(range(num_samples)):
        num_freqs = np.random.randint(1, 6)         # Labels: 1 to 5
        signal = generate_signal(num_freqs)
        signals.append(signal)
        labels.append([num_freqs])

    signals = np.stack(signals).astype(np.float16)  # Shape: [N, L]
    labels = np.array(labels, dtype=np.uint8)       # Shape: [N, 1]
    np.savez(npz_path, signal=signals, label=labels)
    print(f'Saved {num_samples} samples to {npz_path}')


if __name__ == '__main__':
    save_path = '../../data/simulated/simulated_sines.npz'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    simulate_and_save(save_path, num_samples=10000)
