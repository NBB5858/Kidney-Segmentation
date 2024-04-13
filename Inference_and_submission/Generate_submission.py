import numpy as np
from glob import glob
import os
import torch
from torch.nn.parallel import DataParallel

from Training.Configurations import CFG
from Models.Model import build_model
from Utilities.Preprocess_functions import load_data_block
from Evaluation_loop import evaluate


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    '''

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    if rle == '':
        rle = '1 0'
    return rle


model = build_model()
model = DataParallel(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('Models/Retrained/Final_ResNet_Model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

dataset_folder = 'Data'
kidneys = glob(os.path.join(dataset_folder, 'test', '*'))

ids = []
rles = []

for kidney in kidneys:

    perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    inverse_perms = [(0, 1, 2), (2, 0, 1), (1, 2, 0)]

    # perms = [(1,2,0)]
    # inverse_perms = [(2,0,1)]

    print(f'loading kidney {kidney}')
    kidney_image_names = sorted(glob(os.path.join(kidney, "images", "*.tif")))
    kidney_image_block = load_data_block(kidney_image_names)

    orig_shape = kidney_image_block.shape

    print(f'evaluating kidney {kidney}')
    output_blocks, used_perms = evaluate(model, kidney_image_block, perms, thresh=0.5, TTA=True)

    del kidney_image_block

    print('combining blocks')
    overall_block = np.zeros(orig_shape, dtype=np.uint8)
    for index, (block, perm) in enumerate(zip(output_blocks, used_perms)):
        perm_index = perms.index(perm)

        overall_block += torch.tensor(block).permute(inverse_perms[perm_index]).numpy().astype(np.uint8)

    submission_block = np.zeros_like(overall_block, dtype=np.uint8)
    submission_block[overall_block >= len(used_perms) / 2] = 1
    submission_block[overall_block < len(used_perms) / 2] = 0

    for slice_number, mask in enumerate(submission_block):
        identifier = kidney_image_names[slice_number].split("/")[-3:]
        identifier.pop(1)
        identifier = "_".join(identifier)

        ids.append(identifier[:-4])
        rles.append(rle_encode(mask))

    del output_blocks
    del overall_block
    del submission_block
