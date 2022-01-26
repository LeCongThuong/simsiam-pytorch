import torch
from src.losses import negative_cosine_similarity
from pytorch_metric_learning import testers
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
from PIL import Image
import json
import argparse
from types import SimpleNamespace


def eval_self_sup_model(eval_dataloader, model, transforms, device, writer, n_iter):
    losses = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(eval_dataloader):
            x = x.to(device)
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
            loss = loss1 / 2 + loss2 / 2
            losses.append(float(loss.cpu()))
    writer.add_scalar(tag="loss/eval", scalar_value=np.mean(losses), global_step=n_iter)
    model.train()


def calculate_std_l2_norm(z):
    """
    Calculate standard of l2 normalization
    :param z:
    :return:
    """
   # with torch.no_grad():
    z_norm = F.normalize(z.detach(), dim=1)
    return float(torch.std(z_norm, dim=1).mean().cpu())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def sort_by_name(file_name):
    return file_name.name


def resize_and_pad(img, tgt_size, padding_value=255):
    old_size = img.size
    ratio = float(tgt_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # print(new_size)
    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_img = Image.new("L", (tgt_size, tgt_size), color=padding_value)
    new_img.paste(img, ((tgt_size-new_size[0])//2,
                        (tgt_size-new_size[1])//2))
    return new_img


def preprocess_img(cv_img, tgt_size, padding_value=255):
    resized_img = resize_and_pad(Image.fromarray(cv_img).convert('L'), tgt_size, padding_value)
    return resized_img


# convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


# compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def eval_metric_model(query_set, eval_set, model, accuracy_calculator, writer, n_iter):
    query_embeddings, query_labels = get_all_embeddings(query_set, model)
    eval_embeddings, eval_labels = get_all_embeddings(eval_set, model)
    query_labels = query_labels.squeeze(1)
    eval_labels = eval_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        query_embeddings, eval_embeddings, query_labels, eval_labels, False
    )
    writer.add_scalar("eval/acc", scalar_value=float(accuracies["precision_at_1"]), global_step=n_iter)
    writer.add_scalar("eval/mAP", scalar_value=float(accuracies['mean_average_precision']),
                      global_step=n_iter)
    writer.add_scalar("eval/r_precision", scalar_value=float(accuracies['r_precision']), global_step=n_iter)
    writer.add_scalar("eval/mean_average_precision_at_r",
                      scalar_value=float(accuracies['mean_average_precision_at_r']), global_step=n_iter)
    model.train()

    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    print(f"Test MAP: {accuracies['mean_average_precision']}")
    print(f"Test r_Precision: {accuracies['r_precision']}")
    print(f"Test mean_average_precision_at_r: {accuracies['mean_average_precision_at_r']}")
    return accuracies


def parse_aug():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
    return cfg
