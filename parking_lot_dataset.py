import numpy as np
import torch
import torch.utils.data as data

from env_vars import HUMANS_FOLDER, LABELS_FOLDER


class ParkingLotDataset(data.Dataset):
    def __init__(self, data_dir, label_fs, max_points, shift, scale):
        super().__init__()
        self.data_dir = data_dir
        self.label_fs = label_fs
        self.max_points = max_points
        self.shift = shift
        self.scale = scale

    def __len__(self):
        return len(self.label_fs)

    def __getitem__(self, idx):
        label_f = self.label_fs[idx]
        labels = np.load(f"{self.data_dir}/{LABELS_FOLDER}/{label_f}")
        # Transform values to lie in [-1, 1].
        labels[:-2] = labels[:-2] - self.shift
        labels[:-2] = self.scale * labels[:-2] - 1

        points = np.load(f"{self.data_dir}/{HUMANS_FOLDER}/{label_f}")
        n_points = len(points)
        take_points = min(n_points, self.max_points)
        idxs = np.random.choice(n_points, take_points, replace=False)
        xyzs = torch.zeros(self.max_points, 3)
        xyzs[:n_points] = torch.Tensor(points[idxs])
        feats = xyzs[:, :3]

        return (xyzs, feats, torch.Tensor(labels))


class ParkingLotTestDataset(data.Dataset):
    def __init__(self, data_dir, npy_fs, max_points):
        super().__init__()
        self.data_dir = data_dir
        self.npy_fs = npy_fs
        self.max_points = max_points

    def __len__(self):
        return len(self.npy_fs)

    def __getitem__(self, idx):
        npy_f = self.npy_fs[idx]
        points = np.load(f"{self.data_dir}/{HUMANS_FOLDER}/{npy_f}")
        n_points = len(points)
        take_points = min(n_points, self.max_points)
        idxs = np.random.choice(n_points, take_points, replace=False)
        xyzs = torch.zeros(self.max_points, 3)
        xyzs[:n_points] = torch.Tensor(points[idxs])
        feats = xyzs[:, :3]

        return (xyzs, feats)
