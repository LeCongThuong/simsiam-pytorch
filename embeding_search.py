import json
import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import faiss
import cv2

from simsiam.models import Encoder
from simsiam.transforms import test_transforms
from simsiam.utils import sort_by_name, preprocess_img
from get_embedding import get_embedding, get_dataloader, get_model
import matplotlib.pyplot as plt


class EmbeddingSearch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = get_model(cfg)
        print("Loading model is done")
        self.transform = test_transforms(cfg.data.input_shape)

        if cfg.create_embedding.embedding_path != "":
            self.embeddings = np.load(cfg.create_embedding.embedding_path)
        else:
            self.embeddings = self._create_embeddings()
        print("Loading embedding is done")
        self.img_src_dir = list(Path(cfg.data.path).glob("*.png"))
        self.img_src_dir.sort(key=sort_by_name)
        print("Sort img path based name")
        self.faiss_index = self._create_faiss_index()

    def _create_embeddings(self):
        dataloader = get_dataloader(self.cfg)
        return get_embedding(self.cfg, self.model, dataloader)

    def _create_faiss_index(self):
        index = faiss.IndexFlatL2(self.model.emb_dim)
        index.add(self.embeddings)
        return index

    def _create_single_embedding(self, img):
        img = self.transform(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(self.cfg.device)
        embedding = self.model(img)
        return torch.reshape(embedding, (1, -1)).detach().cpu().numpy()

    def get_sim_img(self, img, k_neigbor=5):
        img = self._get_img(img, False)
        embedding = self._create_single_embedding(img)
        D, I = self.faiss_index.search(embedding, k_neigbor)
        return [self.img_src_dir[I[0][i]] for i in range(I.shape[1])]

    def _get_img(self, img, preprocess=True):
        if isinstance(img, str):
            img = cv2.imread(img)
        if preprocess:
            img = preprocess_img(img, cfg.data.input_shape[0]).convert('RGB')

        return img

    def visualize_sim_img(self, img, k_neigbor=5):
        img = self._get_img(img)
        neighbor_id_list = self.get_sim_img(img, k_neigbor)
        print(neighbor_id_list)
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(2 * 2, k_neigbor * 2))
        axes[0, 0].imshow(img)
        axes[0, 0].set_axis_off()
        for i in range(k_neigbor):
            neigbor_img = Image.open(neighbor_id_list[i]).convert('RGB')
            axes[1, i].imshow(neigbor_img)
            axes[1, i].set_axis_off()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    embedding_search_model = EmbeddingSearch(cfg)
    embedding_search_model.visualize_sim_img(args.img_path)
