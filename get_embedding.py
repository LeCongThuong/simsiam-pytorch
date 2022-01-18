import json
import argparse
from types import SimpleNamespace

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from simsiam.models import Encoder
from simsiam.transforms import test_transforms
from simsiam.dataset import DaiNamDataset
import torch
import timm



def get_model(cfg):
    if cfg.use == 'finetuned':
        model = Encoder(
            backbone=cfg.model.backbone,
            pretrained=False
        )

        if cfg.model.weights_path:
            model.load_state_dict(torch.load(cfg.model.weights_path))
    else:
        model = timm.create_model(cfg.model.backbone, pretrained=True, num_classes=0)
    model = model.to(cfg.device)
    return model


def get_dataloader(cfg):
    dataset = DaiNamDataset(
        data_dir=cfg.data.path,
        transform=test_transforms(input_shape=cfg.data.input_shape),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.create_embedding.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=torch.multiprocessing.cpu_count()
    )
    return dataloader


def get_embedding(cfg, model, dataloader):
    model.eval()
    embeddings = []
    chunk_size = 500000
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        img_iter = 0
        for batch, img in pbar:
            batch_size = img.shape[0]
            img_iter += batch_size
            img = img.to(cfg.device)
            img_embedding = model(img).cpu().numpy()
            embeddings.append(img_embedding)
            if img_iter % chunk_size == 0:
                embeddings_stack = np.concatenate(embeddings, axis=0)
                np.save(f"{cfg.create_embedidng.output_dir}/embedding_{img_iter}.npy", embeddings_stack)
                embeddings = []
        embeddings_stack = np.concatenate(embeddings, axis=0)
        np.save(f"{cfg.create_embedding.output_dir}/embedding.npy", embeddings_stack)
    return embeddings_stack


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    model = get_finetuned_model(cfg)
    dataloader = get_dataloader(cfg)
    embedding_stack = get_embedding(cfg, model, dataloader)
