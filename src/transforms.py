from typing import Tuple, Dict

import kornia
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


MEAN = (0.5, )
STD = (0.5,)


def augment_transforms(cfg) -> nn.Sequential:
    augs = nn.Sequential(
       # kornia.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3, p=0.5),
        kornia.augmentation.RandomGaussianNoise(std=0.3),
        kornia.augmentation.RandomErasing(scale=(0.02, cfg.data.augmentation.random_erase), value=1, p=0.3),
       # kornia.augmentation.RandomGrayscale(p=1),
        kornia.augmentation.RandomResizedCrop(
            size=cfg.data.input_shape,
            scale=(cfg.data.augmentation.resize_scale, 1.0),
            ratio=(0.25, 1.33),
            p=0.5
        ),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.Normalize(
            mean=torch.tensor(MEAN),
            std=torch.tensor(STD)
        )
    )
    augs = augs.to(cfg.device)
    return augs


def load_transforms(cfg) -> T.Compose:
    return T.Compose([
        T.Resize(size=cfg.data.input_shape),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]),
        T.ToTensor()
    ])


def test_transforms(cfg) -> T.Compose:
    return T.Compose([
        T.Resize(size=cfg.data.input_shape),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
