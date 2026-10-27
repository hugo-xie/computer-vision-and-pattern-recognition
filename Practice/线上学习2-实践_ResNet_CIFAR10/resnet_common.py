"""CIFAR-10 数据、设备选择、训练管理、模型评估与彩色图像绘制。"""
import copy
import math
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed=2026):
    """固定随机状态；不同设备与版本之间仍可能存在数值差异。"""
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed 必须在 0 到 2**32-1 之间。")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested="auto"):
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device 只能为 cpu / cuda / mps / auto。")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。检查驱动与 PyTorch 构建，或改为 --device cpu。")
    if requested == "mps" and not has_mps:
        raise RuntimeError("MPS 不可用。检查硬件与 PyTorch 构建，或改为 --device cpu。")
    return torch.device(requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def now(device):
    synchronize(device)
    return time.perf_counter()


def image_transform(augment=False):
    steps = []
    if augment:
        steps.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    steps.extend([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return transforms.Compose(steps)


def _cifar(root, train, download, augment=False):
    # torchvision 检查原始批次文件的 MD5，并使用数据集官方 HTTPS 地址下载。
    return datasets.CIFAR10(root=str(root), train=train, download=download,
                            transform=image_transform(augment))


def make_loaders(root="data", train_size=45000, val_size=5000, batch_size=128,
                 seed=2026, download=True, device=torch.device("cpu")):
    """从官方 50,000 张训练图像划分互不重叠的训练/验证子集。"""
    if min(train_size, val_size, batch_size) < 1 or train_size + val_size > 50000:
        raise ValueError("样本数与 batch_size 必须为正，训练+验证不能超过 50,000。")
    full = _cifar(root, train=True, download=download, augment=True)
    if train_size + val_size > len(full):
        raise ValueError("可用训练数据少于指定的训练+验证样本数。")
    # 共享只读原始图像，分别使用训练增强和验证预处理，避免验证随机裁剪。
    validation_view = copy.copy(full)
    validation_view.transform = image_transform(augment=False)
    indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size + val_size]
    options = dict(batch_size=batch_size, num_workers=0, pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(full, train_ids), shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **options)
    val_loader = DataLoader(Subset(validation_view, val_ids), shuffle=False, **options)
    return train_loader, val_loader


def make_test_loader(root="data", batch_size=128, download=False,
                     device=torch.device("cpu")):
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0。")
    return DataLoader(_cifar(root, train=False, download=download), batch_size=batch_size,
                      shuffle=False, num_workers=0, pin_memory=device.type == "cuda")


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False):
    """切换到评估模式，冻结梯度，并按样本数加权计算指标。"""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    predictions, labels = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        predicted = logits.argmax(1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
        if return_predictions:
            predictions.append(predicted.cpu())
            labels.append(targets.cpu())
    if total == 0:
        raise ValueError("评估数据为空。")
    result = {"loss": total_loss / total, "acc": total_correct / total}
    if return_predictions:
        result.update(pred=torch.cat(predictions), true=torch.cat(labels))
    return result


def fit(model, train_loader, val_loader, train_one_epoch, device, epochs=20,
        lr=0.1, checkpoint="outputs/resnet18_best.pt", seed=2026):
    """SGD + 余弦退火训练，每轮验证，保存并恢复最佳验证参数。"""
    if epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs 必须为正整数，lr 必须为正的有限数值。")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    best_acc, best_state = -1.0, None
    path = Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        start = now(device)
        print(f"epoch {epoch}/{epochs} | learning rate {current_lr:.6f}", flush=True)
        train = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "lr": current_lr,
               "train_loss": train["loss"], "train_acc": train["acc"],
               "val_loss": val["loss"], "val_acc": val["acc"],
               "seconds": now(device) - start}
        history.append(row)
        print(f"epoch {epoch:03d} | train loss {row['train_loss']:.4f} "
              f"acc {row['train_acc']:.2%} | val loss {row['val_loss']:.4f} "
              f"acc {row['val_acc']:.2%} | {row['seconds']:.2f}s", flush=True)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            # 包含 BatchNorm 的 running_mean、running_var 与计数器，CPU 模式也需 clone。
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "epoch": epoch, "val_acc": best_acc,
                        "val_loss": val["loss"], "lr": lr, "epoch_lr": current_lr,
                        "seed": seed, "train_size": len(train_loader.dataset),
                        "val_size": len(val_loader.dataset), "batch_size": train_loader.batch_size,
                        "model_name": type(model).__name__, "class_names": CLASS_NAMES,
                        "normalization": {"mean": list(CIFAR_MEAN), "std": list(CIFAR_STD)},
                        "optimizer": "SGD", "momentum": 0.9, "weight_decay": 5e-4,
                        "scheduler": "CosineAnnealingLR", "epochs": epochs,
                        "torch_version": str(torch.__version__), "device": str(device)}, path)
        scheduler.step()
    if best_state is None:
        raise RuntimeError("未生成有效的验证结果，请检查损失与训练配置。")
    model.load_state_dict(best_state)
    return history


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def denormalize(image):
    mean = torch.tensor(CIFAR_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD, dtype=image.dtype).view(3, 1, 1)
    return (image.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def show_batch(loader, count=12):
    if count < 1:
        raise ValueError("count 必须大于 0。")
    images, labels = next(iter(loader))
    count = min(count, images.size(0))
    fig, axes = plt.subplots(2, (count + 1) // 2, figsize=(12, 4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(count):
        axes.flat[i].imshow(denormalize(images[i]))
        axes.flat[i].set_title(CLASS_NAMES[labels[i].item()], fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig


def _finish_plot(fig, save_path=None, show=True):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_history(history, save_path=None, show=True):
    x = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for key in ("train_loss", "val_loss"):
        axes[0].plot(x, [row[key] for row in history], marker="o", label=key)
    for key in ("train_acc", "val_acc"):
        axes[1].plot(x, [row[key] for row in history], marker="o", label=key)
    axes[2].plot(x, [row["lr"] for row in history], marker="o", label="learning_rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross entropy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[2].set_ylabel("Learning rate")
    plt.tight_layout()
    _finish_plot(fig, save_path, show)
    return fig


def plot_confusion(true, pred, save_path=None, show=True):
    true, pred = torch.as_tensor(true).cpu(), torch.as_tensor(pred).cpu()
    matrix = torch.bincount(true * 10 + pred, minlength=100).reshape(10, 10).numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    view = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=60, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(view, ax=ax)
    plt.tight_layout()
    _finish_plot(fig, save_path, show)
    return matrix


@torch.no_grad()
def show_mistakes(model, loader, device, count=8, save_path=None, show=True):
    if count < 1:
        raise ValueError("count 必须大于 0。")
    model.eval()
    items = []
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        for i in (predictions != labels).nonzero(as_tuple=True)[0].tolist():
            items.append((images[i], labels[i].item(), predictions[i].item()))
            if len(items) == count:
                break
        if len(items) == count:
            break
    if not items:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassified examples", ha="center", va="center")
        ax.axis("off")
    else:
        fig, axes = plt.subplots(1, len(items), figsize=(2 * len(items), 2.8), squeeze=False)
        for ax, (image, target, predicted) in zip(axes.flat, items):
            ax.imshow(denormalize(image))
            ax.set_title(f"T: {CLASS_NAMES[target]}\nP: {CLASS_NAMES[predicted]}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    _finish_plot(fig, save_path, show)
    return fig
