import os
import json
import argparse
from types import SimpleNamespace

import torch
import torchvision
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from simsiam.models import SimSiam
from simsiam.losses import negative_cosine_similarity
from simsiam.transforms import augment_transforms, load_transforms, test_transforms
from simsiam.dataset import DaiNamDataset
from simsiam.utils import eval, calculate_std_l2_norm, AverageMeter


def main(cfg: SimpleNamespace) -> None:

    model = SimSiam(
        backbone=cfg.model.backbone,
        latent_dim=cfg.model.latent_dim,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        pred_hidden_dim=cfg.model.pred_hidden_dim
    )
    model = model.to(cfg.device)
    model.train()

    # opt = torch.optim.SGD(
    #     params=model.parameters(),
    #     lr=cfg.train.lr,
    #     momentum=cfg.train.momentum,
    #     weight_decay=cfg.train.weight_decay
    # )
    opt = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.train.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.train.weight_decay
    )

    train_dataset = DaiNamDataset(data_dir=cfg.data.path + "/train", transform=load_transforms(cfg.data.input_shape,
                                                                                               cfg.data.p_blur))
    eval_dataset = DaiNamDataset(data_dir=cfg.data.path + "/eval", transform=test_transforms(cfg.data.input_shape))
    # print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=cfg.train.batch_size,
                    shuffle=True,
                    drop_last=True,
                    pin_memory=True,
                    num_workers=torch.multiprocessing.cpu_count()
    )

    eval_dataloader = torch.utils.data.DataLoader(
                    dataset=eval_dataset,
                    batch_size=cfg.train.batch_size,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=torch.multiprocessing.cpu_count()
    )

    transforms = augment_transforms(
        input_shape=cfg.data.input_shape,
        device=cfg.device
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.8)
    writer = SummaryWriter()

    n_iter = 0
    std_tracker = AverageMeter('std_stacker')
    for epoch in range(cfg.train.epochs):
        std_tracker.reset()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0, leave=False)
        for batch, x in pbar:
            # print("Shape of x:", x.shape)
            opt.zero_grad()

            x = x.to(cfg.device)

            # augment
            # print("type of transform", transforms)
            x1, x2 = transforms(x), transforms(x)
            # print("After transform, shape of input: ", x1.shape, x2.shape)
            # encode
            e1, e2 = model.encode(x1), model.encode(x2)

            # project
            z1, z2 = model.project(e1), model.project(e2)

            # predict
            p1, p2 = model.predict(z1), model.predict(z2)

            # compute loss
            loss1 = negative_cosine_similarity(p1, z1)
            loss2 = negative_cosine_similarity(p2, z2)
            loss = (loss1 + loss2)/2
            loss.backward()
            opt.step()
            with torch.no_grad():
                z1_std = calculate_std_l2_norm(z1)
                z2_std = calculate_std_l2_norm(z2)
                # print("Std", z2_std, z1_std)
                std_tracker.update(z1_std + z2_std)

            pbar.set_description("Epoch {}, Loss: {:.4f}, Std: {:.6f}".format(epoch, float(loss), std_tracker.avg))

            if n_iter % cfg.train.log_interval == 0:
                writer.add_scalar(tag="loss/train", scalar_value=float(loss), global_step=n_iter)
                writer.add_scalar(tag='loss/std', scalar_value=std_tracker.avg, global_step=n_iter)

            if n_iter % cfg.train.eval_inter == 0:
                eval_loss = eval(eval_dataloader, model, transforms, cfg.device, writer, n_iter)

            n_iter += 1

        # save checkpoint
        if (epoch + 1) % cfg.train.checkpoint_inter == 0:
            torch.save(model.encoder.state_dict(), os.path.join(writer.log_dir, cfg.model.name + f"_{epoch + 1}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    main(cfg)
