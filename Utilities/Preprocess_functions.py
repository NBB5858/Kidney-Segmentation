import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from Training.Configurations import CFG


def apply_block_threshold(image_block):

    # Apply sharp cutoff to pixel values with magnitudes above 1-chopping_percentile

    # cutoff largest pixel values
    pixel_list = image_block.reshape(-1).numpy()
    index = -int(len(pixel_list) * CFG.chopping_percentile)
    pixel_list = np.partition(pixel_list, index)[index]
    image_block[image_block > pixel_list] = int(pixel_list)

    # cutoff smallest pixel values
    pixel_list = image_block.reshape(-1).numpy()
    index = -int(len(pixel_list) * CFG.chopping_percentile)
    pixel_list = np.partition(pixel_list, -index)[-index]
    image_block[image_block < pixel_list] = int(pixel_list)

    return image_block


def min_max_normalization(x):

    # Normalizes pixels to interval [0,1]. Used to convert to uint8 later on; improves albumentations speed

    shape = x.shape
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    min_ = x.min(dim=-1, keepdim=True)[0]
    max_ = x.max(dim=-1, keepdim=True)[0]
    if min_.mean() == 0 and max_.mean() == 1:
        return x.reshape(shape)

    x = (x-min_)/(max_-min_+1e-9)
    return x.reshape(shape)


def apply_norm_with_clip(image):

    # Gaussian normalization for pixels

    mean = image.mean()
    std = image.std()

    image = (image - mean)/(std + 1e-5)

    image[image > 5] = (image[image > 5]-5)*1e-3 + 5
    image[image < -3] = (image[image < -3]+3)*1e-3 - 3

    return image


def add_noise(image, max_rand_rate=0.5):

    rand_rate = max_rand_rate*np.random.rand()*torch.rand(1, dtype=torch.float32)

    denom = (1 + rand_rate**2)**0.5

    return (image + torch.randn(size=image.shape, dtype=torch.float32)*rand_rate)/denom


class AuxSet(Dataset):

    # Auxiliary dataset; helps reduce RAM while creating kidney image blocks

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).to(torch.uint8)

        return img


def load_data_block(paths):
    aux_set = AuxSet(paths)
    aux_loader = DataLoader(aux_set, batch_size=16)
    data = []
    for batch in aux_loader:

        data.append(batch)

    x = torch.cat(data, dim=0)

    del data

    x = apply_block_threshold(x)
    x = (min_max_normalization(x.to(torch.float16)[None])[0]*255).to(torch.uint8)

    return x
