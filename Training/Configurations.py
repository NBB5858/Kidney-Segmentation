import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class CFG:

    # ============== model CFG =============
    model_name = 'Unet'
    # backbone = 'se_resnext50_32x4d'
    backbone = 'mit_b2'

    target_size = 1
    in_chans = 3
    # ============== training CFG =============
    image_size = 512
    input_size = 512

    train_batch_size = 4
    valid_batch_size = train_batch_size * 2

    epochs = 25
    lr = 6e-5
    chopping_percentile = 1e-3

    # ============== augmentation =============
    train_aug_list = [
        # Random rotations and horizontal / vertical flips are done in Data loader with numpy
        A.RandomScale(scale_limit=(0.8, 1.25), interpolation=cv2.INTER_CUBIC, p=0.5),
        A.RandomCrop(input_size, input_size, p=1),
        A.RandomGamma(p=0.75),
        A.RandomBrightnessContrast(p=0.5,),
        A.GaussianBlur(p=0.5),
        A.MotionBlur(p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.pytorch.ToTensorV2(transpose_mask=True),
    ]
    train_aug = A.Compose(train_aug_list)
    valid_aug_list = [
        A.RandomCrop(input_size, input_size, p=1),
        A.pytorch.ToTensorV2(transpose_mask=True),

    ]
    valid_aug = A.Compose(valid_aug_list)
