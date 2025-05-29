import numpy as np
import torch
from torch.utils.data import Dataset


class SimulatedSineDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.signals = data['signal']                                # shape: [N, L], float16
        self.labels = data['label']                                  # shape: [N, 1], uint8

    def num_classes(self):
        return len(np.unique(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]                                   # shape: [L]
        label_idx = self.labels[idx].item() - 1                      # shape: [1]

        signal = (signal - signal.mean()) / (signal.std() + 1e-6)    # z-score normalization
        signal = torch.from_numpy(signal).float().unsqueeze(0)       # shape: [1, 1, L]
        label = torch.zeros(self.num_classes(), dtype=torch.float32) # shape: [C]
        label[label_idx] = 1.0                                       # one-hot encoded
        return signal, label


if __name__ == '__main__':
    dataset = SimulatedSineDataset('../../data/simulated/simulated_sines.npz')
