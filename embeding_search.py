import json
import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import faiss
import cv2
import time

from src.transforms import test_transforms
from src.utils import sort_by_name, preprocess_img
from get_metric_embedding import get_embedding, get_model
import matplotlib.pyplot as plt


class EmbeddingSearch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = get_model(cfg)
        self.model.eval()
        print("Loading model is done!")
        self.transform = test_transforms(cfg)

        if cfg.embedding_info.embedding_file_path != "":
            self.embeddings = np.load(cfg.embedding_info.embedding_file_path)
        else:
            self.embeddings = self._create_embeddings()
        print("Loading embedding is done")
        self.img_src_dir = list(Path(cfg.data.path).glob("*.png"))
        self.img_src_dir.sort(key=sort_by_name)
        self.faiss_index = self._create_faiss_index()

    def _create_embeddings(self):
        return get_embedding(self.cfg, self.model)

    def _create_faiss_index(self):
        index = faiss.IndexFlatL2(self.cfg.model.embedding_dim)
        index.add(self.embeddings)
        return index

    def _create_single_embedding(self, img):
        img = self.transform(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(self.cfg.device)
        embedding = self.model(img)
        return torch.reshape(embedding, (1, -1)).detach().cpu().numpy()

    def get_sim_img(self, img, k_neighbor=5):
        if isinstance(img, str):
            img = self._get_img(img)
        embedding = self._create_single_embedding(img)
        D, I = self.faiss_index.search(embedding, k_neighbor)
        return [self.img_src_dir[I[0][i]] for i in range(I.shape[1])]

    def _get_img(self, img_path):
        img = cv2.imread(img_path)
        img = preprocess_img(img, self.cfg.data.input_shape[0]).convert('RGB')
        return img

    def visualize_sim_img(self, img_path, k_neigbor=5):
        # start = time.time()
        img = self._get_img(img_path)
        neighbor_id_list = self.get_sim_img(img, k_neigbor)
        # print("Execution Time: ", time.time() - start)
        # print(neighbor_id_list)
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(2 * 2, k_neigbor * 2))
        axes[0, 0].imshow(img)
        axes[0, 0].set_axis_off()
        for i in range(k_neigbor):
            neighbor_img = Image.open(neighbor_id_list[i]).convert('RGB')
            axes[1, i].imshow(neighbor_img)
            axes[1, i].set_axis_off()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    parser.add_argument("--force_gen_embed", action='store_true')
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    if args.force_gen_embed:
        cfg.create_embedding.embedding_path = ""

    embedding_search_model = EmbeddingSearch(cfg)
    embedding_search_model.visualize_sim_img(args.img_path)
