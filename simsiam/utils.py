import torch
from simsiam.losses import negative_cosine_similarity
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2


def eval(eval_dataloader, model, transforms, device, writer, n_iter):
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
    return np.mean(losses)


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
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(tgt_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # print(new_size)
    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_img = Image.new("L", (tgt_size, tgt_size), color=padding_value)
    new_img.paste(img, ((tgt_size-new_size[0])//2,
                        (tgt_size-new_size[1])//2))
    return new_img


def preprocess_img(cv_img, tgt_size, padding_value=255, blur_winsize=3):
    blur_test_img = cv2.GaussianBlur(cv_img, (blur_winsize, blur_winsize), 0)
    resized_img = resize_and_pad(Image.fromarray(blur_test_img).convert('L'), tgt_size, padding_value)
    return resized_img
