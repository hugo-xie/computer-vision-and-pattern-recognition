"""Oxford-IIIT Pet 前景分割数据集

目录结构（download_data.py 下载解压后即是这样）：
    data/oxford-iiit-pet/
        images/            Abyssinian_1.jpg ...          (7390 张 RGB 照片)
        annotations/
            trainval.txt   3680 行，官方训练+验证划分
            test.txt       3669 行，官方测试划分
            trimaps/       Abyssinian_1.png ...          像素值 1=宠物 2=背景 3=边缘过渡带

本任务做二分类：宠物(前景)=1，背景=0。trimap 里的 3（毛发边缘）算作前景。
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_split(root, split, val_ratio=0.1, seed=0):
    """split ∈ {train, val, test}。官方 trainval 按固定种子再切出 10% 作验证集。"""
    list_file = "test.txt" if split == "test" else "trainval.txt"
    with open(os.path.join(root, "annotations", list_file)) as f:
        names = [line.split()[0] for line in f if line.strip() and not line.startswith("#")]
    if split == "test":
        return names
    rng = random.Random(seed)
    names = sorted(names)
    rng.shuffle(names)
    n_val = int(len(names) * val_ratio)
    return names[n_val:] if split == "train" else names[:n_val]


def load_pair(root, name):
    img = Image.open(os.path.join(root, "images", name + ".jpg")).convert("RGB")
    tri = np.array(Image.open(os.path.join(root, "annotations", "trimaps", name + ".png")))
    mask = Image.fromarray(((tri == 1) | (tri == 3)).astype(np.uint8) * 255)
    return img, mask


def to_tensor(img):
    """PIL RGB → 标准化后的 float 张量 [3,H,W]"""
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class PetSegDataset(Dataset):
    def __init__(self, root, split="train", img_size=160, augment=None):
        self.root, self.split, self.size = root, split, img_size
        self.augment = (split == "train") if augment is None else augment
        self.names = read_split(root, split)

    def __len__(self):
        return len(self.names)

    def _augment(self, img, mask):
        # 1) 随机缩放裁剪：保留原图 60%~100% 面积
        w, h = img.size
        s = random.uniform(0.6, 1.0) ** 0.5
        cw, ch = int(w * s), int(h * s)
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        img, mask = img.crop((x0, y0, x0 + cw, y0 + ch)), mask.crop((x0, y0, x0 + cw, y0 + ch))
        # 2) 随机水平翻转
        if random.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

    def __getitem__(self, i):
        img, mask = load_pair(self.root, self.names[i])
        if self.augment:
            img, mask = self._augment(img, mask)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)     # 标签必须用最近邻插值
        x = to_tensor(img)
        if self.augment:   # 3) 亮度/对比度轻微扰动（只作用于图像，不动标签）
            x = x * random.uniform(0.8, 1.2) + random.uniform(-0.2, 0.2)
        y = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None]   # [1,H,W]
        return x, y
