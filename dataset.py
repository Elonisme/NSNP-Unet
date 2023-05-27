import random

import torch.utils.data as data
import PIL.Image as Image
import os


def make_dataset(root):
    imgs = []
    liver_path = os.path.join(root, "patient")
    tumor_path = os.path.join(root, "tumor")
    liver_img_path = os.listdir(liver_path)
    tumor_img_path = os.listdir(tumor_path)
    n = len(os.listdir(liver_path))

    for i in range(n):
        img = os.path.join(liver_path, liver_img_path[i])
        mask = os.path.join(tumor_path, tumor_img_path[i])
        imgs.append((img, mask))

    random.shuffle(imgs)
    return imgs


class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)