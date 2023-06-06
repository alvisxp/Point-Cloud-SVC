from typing import Any
from torch.utils.data import Dataset
import numpy as np
import os

class Dataloader(Dataset):
    def __init__(self, dataset, root, split='train'):
        self.dataset = dataset
        self.root = root
        self.split = split

        if split == 'train':
            self.points = np.load(os.path.join(root, 'train_points.npy'))
            self.labels = np.load(os.path.join(root, 'train_labels.npy'))
        else:
            self.points = np.load(os.path.join(root, 'test_points.npy'))
            self.labels = np.load(os.path.join(root, 'test_labels.npy'))

        print('The size of %s data is %d'%(split, len(self.points)))
    
    def _len__(self):
        return len(self.points)
    
    def __getitem__(self, index):

        pts = self.points[index][:, 0:3]
        cls = self.labels[index]

        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        radius = np.max(np.linalg.norm(pts, axis=1))
        pts = pts/radius
        