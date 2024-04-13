import torch
from torch.utils.data import Dataset
import numpy as np
from Training.Configurations import CFG
from Utilities.Preprocess_functions import apply_norm_with_clip, add_noise


class TrainingDataset(Dataset):

    # dataset for loading Data into model for training

    def __init__(self, x, y, training=True):
        super(Dataset, self).__init__()
        self.x = x  # [(C,H,W),...]
        self.y = y  # [(C,H,W),...]
        self.image_size = CFG.image_size
        self.in_chans = CFG.in_chans
        self.training = training
        if training:
            self.transform = CFG.train_aug
        else:
            self.transform = CFG.valid_aug

        self.block_boundaries = [x[0].shape[0]]
        for block in x[1:]:
            self.block_boundaries.append(self.block_boundaries[-1] + block.shape[0])

    def __len__(self):
        return sum([y.shape[0] for y in self.y])

    def __getitem__(self, idx):

        if idx < self.block_boundaries[0]:
            x = self.x[0][idx]
            y = self.y[0][idx]

        for jj in range(len(self.block_boundaries[1:])):

            if idx >= self.block_boundaries[jj] and idx < self.block_boundaries[jj+1]:
                x = self.x[jj+1][idx - self.block_boundaries[jj]]
                y = self.y[jj+1][idx - self.block_boundaries[jj]]

        x = x.unsqueeze(dim=0)

        data = self.transform(image=x.numpy().transpose(1, 2, 0), mask=y.numpy())
        x = data['image']
        y = data['mask'] >= 127

        x = apply_norm_with_clip(x.to(torch.float32))

        if self.training:
            x = add_noise(x)
            i = np.random.randint(4)  # Performing simple operations outside of the albumentations library reduces memory usage and improves speed
            x = x.rot90(i, dims=(1, 2))
            y = y.rot90(i, dims=(0, 1))
            for i in range(3):
                if np.random.randint(2):
                    x = x.flip(dims=(i,))
                    if i >= 1:
                        y = y.flip(dims=(i-1,))

        if CFG.backbone == 'mit_b2':
            x = x.squeeze(dim=0).repeat((3, 1, 1))  # mit_b2 model requires 3 channel input

        return x, y  # (uint8,uint8)
