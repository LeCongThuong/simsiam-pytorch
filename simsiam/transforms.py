from typing import Tuple, Dict

import kornia
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def augment_transforms(
    cfg, input_shape, train_flag, device
) -> nn.Sequential:
    # max_erase_scale = cfg.data.train.augmentation.random_erase if train_flag else cfg.data.eval.augmentation.random_erase
    # min_resize_crop_scale = cfg.data.train.augmentation.resize_scale if train_flag else cfg.data.eval.augmentation.resize_scale
    augs = nn.Sequential(
        kornia.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3, p=0.5),
        kornia.augmentation.RandomGaussianNoise(std=0.3),
        kornia.augmentation.RandomErasing(scale=(0.02, cfg.data.augmentation.random_erase), value=1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=1),
        kornia.augmentation.RandomResizedCrop(
            size=input_shape,
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
    augs = augs.to(device)
    return augs


def load_transforms(input_shape: Tuple[int, int], p_blur=0) -> T.Compose:
    transform_list = [T.Resize(size=input_shape, interpolation=1)]
    if p_blur != 0:
        transform_list.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=p_blur))
    transform_list.append(T.ToTensor())
    return T.Compose(transform_list)


def test_transforms(input_shape: Tuple[int, int]) -> T.Compose:
    return T.Compose([
        T.Resize(size=input_shape, interpolation=1),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
