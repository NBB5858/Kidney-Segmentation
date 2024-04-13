import torch
import numpy as np
from torch.utils.data import Dataset
from Training.Configurations import CFG
from Utilities.Preprocess_functions import apply_norm_with_clip


class EvaluateSet(Dataset):

    def __init__(self, image_block, perm, tile_size, overlap=0.5):
        self.image_block = image_block
        self.perm = perm
        self.tile_size = np.array(tile_size)
        self.overlap = overlap

        data_dicts = []

        for slice_number, image_slice in enumerate(image_block):
            image_shape = image_slice.shape
            height, width = image_shape

            min_overlap = float(self.overlap)
            max_stride = self.tile_size * (1.0 - min_overlap)

            num_patches = np.ceil(np.array([height, width]) / max_stride).astype(np.int64)

            starts = [np.int64(np.linspace(0, width - self.tile_size[1], num_patches[1])),
                      np.int64(np.linspace(0, height - self.tile_size[0], num_patches[0]))]
            stops = [starts[0] + self.tile_size[0], starts[1] + self.tile_size[1]]

            for y1, y2 in zip(starts[1], stops[1]):
                for x1, x2 in zip(starts[0], stops[0]):
                    data_dicts.append({'slice_number': slice_number, 'perm': perm,
                                       'y_range': (y1, y2), 'x_range': (x1, x2), 'orig_shape': image_shape})

        self.data_dicts = data_dicts

    def __getitem__(self, idx):

        data_dict = self.data_dicts[idx]

        image = self.image_block[data_dict['slice_number']]

        image = apply_norm_with_clip(image.to(torch.float32))

        x1, x2 = (data_dict['x_range'][0], data_dict['x_range'][1])
        y1, y2 = (data_dict['y_range'][0], data_dict['y_range'][1])

        image_tile = image[y1:y2, x1:x2]

        if CFG.backbone == 'mit_b2':
            image_tile = image_tile.squeeze(dim=0).repeat((3, 1, 1))  # only takes in 3 channels

        return image_tile, data_dict

    def __len__(self):
        return len(self.data_dicts)


def custom_collator(data):
    batch_size = len(data)
    tile_dim = data[0][0].shape  # (1, height, width)

    image_tile_batch = torch.zeros(batch_size, CFG.in_chans, tile_dim[-2], tile_dim[-1])

    data_dict_list = []
    for idx, (image_tile, data_dict) in enumerate(data):
        image_tile_batch[idx] = image_tile
        data_dict_list.append(data_dict)

    return [image_tile_batch, data_dict_list]
