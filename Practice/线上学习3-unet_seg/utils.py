"""损失函数与评价指标"""
import torch
import torch.nn.functional as F


def dice_loss(logits, target, eps=1.0):
    """软 Dice 损失：1 - 2|P∩G| / (|P|+|G|)，P 用 sigmoid 概率（可导）"""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def bce_dice_loss(logits, target, w_dice=1.0):
    """二值交叉熵（逐像素） + Dice（整体重叠），分割中最常用的组合"""
    return F.binary_cross_entropy_with_logits(logits, target) + w_dice * dice_loss(logits, target)


@torch.no_grad()
def confusion(logits, target, thr=0.5):
    """返回 TP, FP, FN, TN 的像素计数（在整个 batch 上求和）"""
    pred = (torch.sigmoid(logits) > thr).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    return tp, fp, fn, tn


@torch.no_grad()
def per_image_iou_dice(logits, target, thr=0.5, eps=1e-7):
    """每张图分别算 IoU 和 Dice，返回两个长度为 B 的张量"""
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    s = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    iou = (inter + eps) / (s - inter + eps)
    dice = (2 * inter + eps) / (s + eps)
    return iou, dice


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")   # Apple 芯片
    return torch.device("cpu")
