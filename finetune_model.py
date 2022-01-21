import os
import json
import argparse
from types import SimpleNamespace

import torch
import torchvision
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from simsiam.models import ResNet
from simsiam.transforms import load_transforms, augment_transforms, test_transforms
from simsiam.finetune_dataset import FontDataset
from pytorch_metric_learning import samplers
from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from simsiam.utils import eval_finetuned_model


def main(cfg: SimpleNamespace) -> None:
    model = ResNet(
        backbone=cfg.model.backbone,
        embedding_dim=cfg.model.embedding_dim,
        pretrained=cfg.model.pretrained,
        freeze=cfg.model.freeze
    )

    if cfg.model.weights_path != "":
        model.encoder.load_state_dict(torch.load(cfg.model.weights_path))

    model = model.to(cfg.device)

    opt = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.train.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.train.weight_decay
    )

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=cfg.train.loss_margin, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=cfg.train.loss_margin, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision",
                                                      'mean_average_precision_at_r', 'r_precision'), k=None)

    train_transform = load_transforms(cfg.data.input_shape, cfg.data.augmentation.p_blur)
    train_dataset = FontDataset(cfg, os.path.join(cfg.data.path, 'train'), transform=train_transform)

    query_transform = test_transforms(cfg.data.input_shape)
    query_dataset = FontDataset(cfg, os.path.join(cfg.data.path, 'query'), transform=query_transform)

    eval_transform = test_transforms(cfg.data.input_shape)
    eval_dataset = FontDataset(cfg, os.path.join(cfg.data.path, 'eval'), transform=eval_transform)

    train_sampler = samplers.MPerClassSampler(train_dataset.label_list, cfg.data.sample_per_cls, batch_size=None,
                                              length_before_new_iter=len(train_dataset.label_list))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.train.batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   pin_memory=True,
                                                   num_workers=torch.multiprocessing.cpu_count())

    transforms = augment_transforms(
        cfg=cfg,
        input_shape=cfg.data.input_shape,
        train_flag=True,
        device=cfg.device
    )

    writer = SummaryWriter()

    n_iter = 0
    for epoch in range(cfg.train.epochs):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, (x, y) in pbar:
            opt.zero_grad()

            x, y = x.to(cfg.device), y.to(cfg.device)
            x = transforms(x)
            embedding = model(x)
            indices_tuple = mining_func(embedding, y)
            loss = loss_func(embedding, y, indices_tuple)
            loss.backward()
            opt.step()
            pbar.set_description("Epoch {}, Loss: {:.4f}".format(epoch, float(loss)))

            if n_iter % cfg.train.log_interval == 0:
                writer.add_scalar(tag="loss", scalar_value=float(loss), global_step=n_iter)
                print(
                    "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                        epoch, n_iter, loss, mining_func.num_triplets
                    )
                )

            if n_iter % cfg.train.eval_inter == 0:
                accuracies = eval_finetuned_model(query_dataset, eval_dataset, model, accuracy_calculator)
                writer.add_scalar("eval/acc", scalar_value=float(accuracies["precision_at_1"]), global_step=n_iter)
                writer.add_scalar("eval/mAP", scalar_value=float(accuracies['mean_average_precision']),
                                  global_step=n_iter)
                writer.add_scalar("eval/r_precision", scalar_value=float(accuracies['r_precision']), global_step=n_iter)
                writer.add_scalar("eval/mean_average_precision_at_r",
                                  scalar_value=float(accuracies['mean_average_precision_at_r']), global_step=n_iter)
                model.train()
            n_iter += 1
        # save checkpoint
        if (epoch + 1) % cfg.train.checkpoint_inter == 0:
            torch.save(model.state_dict(), os.path.join(writer.log_dir, cfg.model.name + f"_{epoch}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    main(cfg)
