"""训练 U-Net：Oxford-IIIT Pet 前景分割

用法示例：
    python train.py                                   # 默认参数（有 GPU 约 10~15 分钟）
    python train.py --epochs 10 --img-size 128 --base 16   # 只有 CPU 时的轻量设置
    python train.py --resume                          # 从 checkpoints/last.pt 继续训练

输出：
    checkpoints/best.pt   验证集 IoU 最高的模型（test.py 默认加载它）
    checkpoints/last.pt   最近一轮（含优化器状态，可断点续训）
    checkpoints/log.csv   每轮的损失与指标
"""
import argparse
import contextlib
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PetSegDataset
from model import UNet, count_params
from utils import bce_dice_loss, per_image_iou_dice, pick_device


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/oxford-iiit-pet")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=160, help="训练/测试时把图像缩放到的边长，需为16的倍数")
    p.add_argument("--base", type=int, default=32, help="U-Net 第一层通道数，越大模型越大")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="从 out/last.pt 继续")
    p.add_argument("--time-limit", type=float, default=0, help="运行多少分钟后保存并退出（0=不限制）")
    p.add_argument("--subset", type=int, default=0, help="只用前 N 张训练图（0=全部），CPU 上快速试跑用")
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def run_epoch(model, loader, device, optimizer=None, scaler=None, deadline=None):
    """optimizer=None 时为评估模式。返回 (平均损失, 平均IoU, 平均Dice, 是否完整跑完)"""
    train = optimizer is not None
    model.train(train)
    tot_loss, ious, dices, n = 0.0, [], [], 0
    use_amp = scaler is not None
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with (torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()):
                logits = model(x)
                loss = bce_dice_loss(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        iou, dice = per_image_iou_dice(logits.float(), y)
        tot_loss += loss.item() * x.size(0)
        ious.append(iou.cpu())
        dices.append(dice.cpu())
        n += x.size(0)
        if train and step % 50 == 0:
            print("    step %4d/%d  loss %.4f" % (step, len(loader), loss.item()), flush=True)
        if deadline and time.time() > deadline:
            return tot_loss / n, torch.cat(ious).mean().item(), torch.cat(dices).mean().item(), False
    return tot_loss / n, torch.cat(ious).mean().item(), torch.cat(dices).mean().item(), True


def main():
    args = get_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print("设备:", device)

    train_set = PetSegDataset(args.data_root, "train", args.img_size)
    val_set = PetSegDataset(args.data_root, "val", args.img_size)
    if args.subset:
        train_set.names = train_set.names[:args.subset]
    print("训练集 %d 张，验证集 %d 张" % (len(train_set), len(val_set)))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = UNet(in_channels=3, num_classes=1, base=args.base).to(device)
    print("U-Net 参数量: %.2fM" % (count_params(model) / 1e6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 学习率：余弦退火，从 lr 逐渐降到接近 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = None   # 混合精度（仅 CUDA），省显存、提速
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()

    start_epoch, best_iou = 0, 0.0
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_iou = ck["epoch"] + 1, ck["best_iou"]
        print("从第 %d 轮继续，当前最佳验证 IoU = %.4f" % (start_epoch, best_iou))

    log_path = os.path.join(args.out, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,lr,train_loss,train_iou,val_loss,val_iou,val_dice,seconds\n")

    deadline = time.time() + args.time_limit * 60 if args.time_limit > 0 else None
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d/%d  lr=%.2e" % (epoch + 1, args.epochs, lr))
        tr_loss, tr_iou, _, done = run_epoch(model, train_loader, device, optimizer, scaler, deadline)
        if not done:
            print("到达时间上限，本轮未完成，已放弃本轮（下次 --resume 从第 %d 轮重新开始）" % (epoch + 1))
            break
        va_loss, va_iou, va_dice, _ = run_epoch(model, val_loader, device)
        scheduler.step()
        sec = time.time() - t0
        print("  train loss %.4f IoU %.4f | val loss %.4f IoU %.4f Dice %.4f | %.0fs"
              % (tr_loss, tr_iou, va_loss, va_iou, va_dice, sec))
        with open(log_path, "a") as f:
            f.write("%d,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f\n"
                    % (epoch + 1, lr, tr_loss, tr_iou, va_loss, va_iou, va_dice, sec))

        cfg = {"base": args.base, "img_size": args.img_size}
        if va_iou > best_iou:
            best_iou = va_iou
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_iou": va_iou}, best_path)
            print("  ★ 新的最佳模型已保存: %s" % best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_iou": best_iou,
                    "cfg": cfg}, last_path)
        if deadline and time.time() + sec > deadline:   # 剩余时间不够再跑一整轮
            print("到达时间上限，已保存。用 --resume 继续。")
            break
    print("训练结束，最佳验证 IoU = %.4f" % best_iou)


if __name__ == "__main__":
    main()
