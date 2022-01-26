import json
import argparse
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from src.models import MetricLearningModel
from src.transforms import test_transforms
from src.datasets import DaiNamDataset
from src.utils import parse_aug


def get_model(cfg):
    model = MetricLearningModel(backbone=cfg.model.backbone, embedding_dim=cfg.model.embedding_dim,
                                pretrained=cfg.model.pretrained, freeze=False)
    if cfg.model.weights_path != "":
        model.load_state_dict(torch.load(cfg.model.weights_path))
    model = model.to(cfg.device)
    return model


def get_data(cfg):
    dataset = DaiNamDataset(
        cfg, mode="",
        transform=test_transforms(cfg),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.embedding_info.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=torch.multiprocessing.cpu_count()
    )
    return dataset, dataloader


def get_embedding(cfg, model):
    dataset, dataloader = get_data(cfg)
    model.eval()
    embeddings = np.ones((len(dataset), cfg.model.embedding_dim), np.float32)
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        img_iter = 0
        for batch, img in pbar:
            batch_size = img.shape[0]
            img_iter += batch_size
            img = img.to(cfg.device)
            img_embedding = model(img).cpu().numpy()
            embeddings[img_iter - batch_size: img_iter] = img_embedding
        np.save(cfg.embedding_info.embedding_file_path, embeddings)
    return embeddings


if __name__ == "__main__":
    cfg = parse_aug()
    model = get_model(cfg)
    dataloader = get_data(cfg)
    embedding_stack = get_embedding(cfg, model)
