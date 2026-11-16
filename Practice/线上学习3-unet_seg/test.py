"""测试 U-Net

用法：
    python test.py                                 # 在官方测试集(3669张)上评估 checkpoints/best.pt
    python test.py --ckpt checkpoints/best.pt --n-vis 12
    python test.py --image my_cat.jpg              # 对任意一张图片做预测，结果存到 results/

输出（默认 results/ 目录）：
    test_metrics.txt   平均 IoU / Dice / 像素准确率
    test_vis.png       若干测试样本：原图 | 真值 | 预测 | 叠加
    per_image.csv      每张测试图的 IoU（可用来找"最差的样本"做误差分析）
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import PetSegDataset, to_tensor
from model import UNet
from utils import confusion, per_image_iou_dice, pick_device


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/oxford-iiit-pet")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--out", default="results")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-vis", type=int, default=8, help="可视化多少张测试样本")
    p.add_argument("--image", default=None, help="对单张图片预测（不需要数据集）")
    return p.parse_args()


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]
    model = UNet(in_channels=3, num_classes=1, base=cfg["base"]).to(device).eval()
    model.load_state_dict(ck["model"])
    print("已加载 %s（第 %d 轮，验证 IoU %.4f）" % (ckpt_path, ck["epoch"] + 1, ck.get("val_iou", ck.get("best_iou", 0))))
    return model, cfg["img_size"]


def denorm(x):
    """标准化张量 → uint8 图像 [H,W,3]"""
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    img = (x.cpu().numpy() * std + mean).clip(0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def overlay(img, mask, color=(255, 40, 40), alpha=0.45):
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + np.array(color) * alpha
    return out.astype(np.uint8)


def gray3(m):
    return np.repeat((m.astype(np.uint8) * 255)[..., None], 3, axis=2)


@torch.no_grad()
def predict_single(model, img_size, path, out_dir, device):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    x = to_tensor(img.resize((img_size, img_size), Image.BILINEAR))[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    # 概率图放大回原始分辨率后再取阈值，边界更平滑
    prob = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
    mask = prob > 0.5
    stem = os.path.splitext(os.path.basename(path))[0]
    Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(out_dir, stem + "_mask.png"))
    Image.fromarray(overlay(np.array(img), mask)).save(os.path.join(out_dir, stem + "_overlay.png"))
    print("前景占比 %.1f%%，结果已保存到 %s/" % (mask.mean() * 100, out_dir))


@torch.no_grad()
def evaluate(model, img_size, args, device):
    test_set = PetSegDataset(args.data_root, "test", img_size, augment=False)
    loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers)
    print("测试集 %d 张" % len(test_set))
    ious, dices = [], []
    tp = fp = fn = tn = 0
    vis_rows = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        iou, dice = per_image_iou_dice(logits, y)
        ious.append(iou.cpu())
        dices.append(dice.cpu())
        a, b, c, d = confusion(logits, y)
        tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d
        if len(vis_rows) < args.n_vis:
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0]
            for k in range(min(len(x), args.n_vis - len(vis_rows))):
                img, gt = denorm(x[k]), y[k, 0].cpu().numpy() > 0.5
                vis_rows.append(np.concatenate([img, gray3(gt), gray3(pred[k]), overlay(img, pred[k])], axis=1))
    ious, dices = torch.cat(ious).numpy(), torch.cat(dices).numpy()

    lines = [
        "测试图像数        : %d" % len(ious),
        "平均 IoU (逐图)   : %.4f" % ious.mean(),
        "平均 Dice (逐图)  : %.4f" % dices.mean(),
        "前景 IoU (全体像素): %.4f" % (tp / (tp + fp + fn)),
        "背景 IoU (全体像素): %.4f" % (tn / (tn + fp + fn)),
        "像素准确率        : %.4f" % ((tp + tn) / (tp + fp + fn + tn)),
        "IoU<0.5 的困难样本: %d 张" % int((ious < 0.5).sum()),
    ]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out, "test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(args.out, "per_image.csv"), "w") as f:
        f.write("name,iou,dice\n")
        for n, i, d in zip(test_set.names, ious, dices):
            f.write("%s,%.4f,%.4f\n" % (n, i, d))
    if vis_rows:
        Image.fromarray(np.concatenate(vis_rows, axis=0)).save(os.path.join(args.out, "test_vis.png"))
        print("可视化已保存: %s（每行：原图 | 真值 | 预测 | 叠加）" % os.path.join(args.out, "test_vis.png"))


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    model, img_size = load_model(args.ckpt, device)
    if args.image:
        predict_single(model, img_size, args.image, args.out, device)
    else:
        evaluate(model, img_size, args, device)


if __name__ == "__main__":
    main()
