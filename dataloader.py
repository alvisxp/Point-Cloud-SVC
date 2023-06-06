from torch.utils.data import Dataset
import numpy as np
import os

class Dataloader(Dataset):
    def __init__(self, dataset, root, split='train'):
        self.dataset = dataset
        self.root = root
        self.split = split

        if split == 'train':
            self.points = np.load(os.path)