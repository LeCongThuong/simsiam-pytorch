from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from simsiam.transforms import load_transforms
import matplotlib.pyplot as plt
import torchvision
from simsiam.utils import sort_by_name


class DaiNamDataset(Dataset):
    def __init__(self, data_dir, transform=None, tgt_size=224):
        self.data_dir = data_dir
        self.tgt_size = tgt_size
        self.img_path_list = list(Path(data_dir).glob("*.png"))
        self.img_path_list.sort(key=sort_by_name)
        self.transform = transform
        if transform is None:
            self.transform = load_transforms((tgt_size, tgt_size))

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path_list)


if __name__ == '__main__':
    data_dir = "/mnt/hdd/thuonglc/mocban/ocr_retrieval_dataset/origin/test_dataset/wiki"
    dataset = DaiNamDataset(data_dir)

    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample.shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        img = torchvision.transforms.ToPILImage()(sample)
        # print(img.shape)
        ax.imshow(img)
        ax.axis('off')

        if i == 3:
            plt.show()
            break