import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from glob import glob
from Training.Configurations import CFG
from Utilities.Preprocess_functions import load_data_block
from Training.Train_loop import train_model
from Utilities.Loss_functions import DiceLoss_batch_avg
from Models.Model import build_model
from Training_dataset import TrainingDataset

# collect image names and mask names
kidney1_image_names = sorted(glob('/kidney_1_dense/images/*'))
kidney1_mask_names = sorted(glob('/kidney_1_dense/labels/*'))

kidney3_mask_names = sorted(glob('/kidney_3_dense/labels/*'))
kidney3_image_names = [x.replace('labels', 'images').replace('dense', 'sparse') for x in kidney3_mask_names]


# load images into 3d block
kidney1_image_block = load_data_block(kidney1_image_names)
kidney1_mask_block = load_data_block(kidney1_mask_names)

kidney3_image_block = load_data_block(kidney3_image_names)
kidney3_mask_block = load_data_block(kidney3_mask_names)


# collect permutations of blocks
train_images = []
train_masks = []

train_images.append(kidney1_image_block)
train_images.append(kidney1_image_block.permute(1, 2, 0))
train_images.append(kidney1_image_block.permute(2, 0, 1))

train_masks.append(kidney1_mask_block)
train_masks.append(kidney1_mask_block.permute(1, 2, 0))
train_masks.append(kidney1_mask_block.permute(2, 0, 1))

val_images = []
val_masks = []

val_images.append(kidney3_image_block)
val_masks.append(kidney3_mask_block)


# build training sets and dataloaders
TrainSet = TrainingDataset(train_images, train_masks, training=True)
ValidSet = TrainingDataset(val_images, val_masks, training=False)

TrainLoader = DataLoader(TrainSet, batch_size=CFG.train_batch_size, num_workers=4, shuffle=True)
ValidLoader = DataLoader(ValidSet, batch_size=CFG.valid_batch_size, num_workers=4, shuffle=False)


# prepare for model training
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

model = build_model()
model = DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr,
                                                steps_per_epoch=len(TrainLoader), epochs=CFG.epochs+1,
                                                pct_start=0.1,)

loss_fc = DiceLoss_batch_avg()

history = train_model(model, optimizer, scheduler, loss_fc, CFG.epochs, TrainLoader, ValidLoader)
