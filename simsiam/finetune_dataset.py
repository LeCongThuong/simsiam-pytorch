import random
import os

from torch.utils.data import Dataset
import torch
from pathlib import Path
from simsiam.utils import sort_by_name
from simsiam.transforms import load_transforms
from PIL import Image


class FontDataset(Dataset):
    def __init__(self, cfg, data_dir, transform=None):
        self.cfg = cfg
        if transform is None:
            self.transform = load_transforms(cfg.data.input_shape, cfg.data.p_blur)
        else:
            self.transform = transform
        self.data_dir = data_dir
        self.img_path_list = list(Path(self.data_dir).rglob('*.png'))
        self.img_path_list.sort(key=sort_by_name)
        self.label_list = [int(img_path.parts[-2]) for img_path in self.img_path_list]

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img_label = self.label_list[idx]
        return img, img_label

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    data_dir = "/home/hmi/Documents/thuong_doc/mocban/Printed-Chinese-Character-OCR/train_test_retrieval_dataset/train"
    train_dataset = FontDataset(data_dir)
    print(len(train_dataset))
    print(train_dataset[0])
