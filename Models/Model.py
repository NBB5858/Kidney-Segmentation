import torch.nn as nn
import segmentation_models_pytorch as smp
from Training.Configurations import CFG


class UNetModel(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=CFG.backbone,
            encoder_weights=weight,
            in_channels=CFG.in_chans,
            classes=CFG.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.model(image)
        return output[:, 0]


def build_model(weight="imagenet"):

    model = UNetModel(weight)

    return model.cuda()
