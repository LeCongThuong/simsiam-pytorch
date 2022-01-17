from typing import Tuple, Dict

import kornia
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def augment_transforms(
    input_shape, device
) -> nn.Sequential:
    augs = nn.Sequential(
        kornia.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=1),
        kornia.augmentation.RandomAffine(degrees=10, translate=0.1, p=0.3),
        kornia.augmentation.RandomHorizontalFlip(p=0.3),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(5, 5), sigma=(2.0, 2.0), p=0.3),
        kornia.augmentation.RandomResizedCrop(
            size=input_shape,
            scale=(0.2, 1.0),
            ratio=(0.75, 1.33),
            p=0.3
        ),
        kornia.augmentation.RandomErasing(p=0.3),
        kornia.augmentation.Normalize(
            mean=torch.tensor(MEAN),
            std=torch.tensor(STD)
        )
    )
    augs = augs.to(device)
    return augs


def load_transforms(input_shape: Tuple[int, int]) -> T.Compose:
    return T.Compose([
        T.Resize(size=input_shape, interpolation=Image.LANCZOS),
        T.ToTensor(),
    ])


def test_transforms(input_shape: Tuple[int, int]) -> T.Compose:
    return T.Compose([
        T.Resize(size=input_shape, interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
