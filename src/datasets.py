from torch.utils.data import Dataset
from pathlib import Path
from src.utils import sort_by_name
from src.transforms import load_transforms
from PIL import Image
import os


class DaiNamDataset(Dataset):
    def __init__(self, cfg, mode='train', transform=None):
        self.img_path_list = list(Path(os.path.join(cfg.data.path, mode) if mode != "" else cfg.data.path).glob("*.png")) + list(Path(os.path.join(cfg.data.path, mode) if mode != "" else cfg.data.path).rglob("*.jpg"))
        self.img_path_list.sort(key=sort_by_name)
        self.transform = transform
        if transform is None:
            self.transform = load_transforms(cfg)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path_list)


class FontDataset(Dataset):
    def __init__(self, cfg, mode='train', transform=None):
        self.cfg = cfg
        if transform is None:
            self.transform = load_transforms(cfg)
        else:
            self.transform = transform
        self.img_path_list = list(Path(os.path.join(cfg.data.path, mode) if mode != "" else cfg.data.path).rglob('*.png')) + list(Path(os.path.join(cfg.data.path, mode) if mode != "" else cfg.data.path).rglob("*.jpg"))
        self.img_path_list.sort(key=sort_by_name)
        print(len(self.img_path_list))
        self.label_list = [int(img_path.parts[-2]) for img_path in self.img_path_list]

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        img_label = self.label_list[idx]
        return img, img_label

    def __len__(self):
        return len(self.label_list)


# if __name__ == '__main__':
    # # test Font dataset
    # data_dir = "./train_test_retrieval_dataset/train"
    # train_dataset = FontDataset(data_dir)
    # print(len(train_dataset))
    # print(train_dataset[0])
    #
    # # test DaiNam dataset
    # data_dir = "./test_dataset/wiki"
    # dataset = DaiNamDataset(data_dir)
    # fig = plt.figure()
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     print(i, sample.shape)
    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     img = torchvision.transforms.ToPILImage()(sample)
    #     ax.imshow(img)
    #     ax.axis('off')
    #     if i == 3:
    #         plt.show()
    #         break




