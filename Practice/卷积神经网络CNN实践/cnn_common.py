"""CNN 图像分类公共工具：数据读取、设备选择、训练管理、评估与绘图。"""
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed=2026):
    """固定随机种子；不同硬件/版本仍可能存在数值差别。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested="auto"):
    """显式指定不可用设备时报错，避免把 CPU 误记成 GPU。"""
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device 只能为 auto / cpu / cuda / mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。检查 NVIDIA 驱动及 PyTorch 构建，或先改用 cpu。")
    if requested == "mps" and not has_mps:
        raise RuntimeError("MPS 不可用。检查 Mac 硬件、系统与 PyTorch 版本，或改用 cpu。")
    return torch.device(requested)


def synchronize(device):
    """GPU 运算是异步的，计时前后需要同步。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def now(device):
    synchronize(device)
    return time.perf_counter()


def _fashion(root, train, download):
    # torchvision 某些版本默认使用旧 HTTP 源；改用官方 S3 HTTPS 与作者仓库。
    class CourseFashionMNIST(datasets.FashionMNIST):
        @property
        def raw_folder(self):
            # 保持 torchvision 标准缓存目录，兼容已有的离线数据。
            return str(Path(self.root) / "FashionMNIST" / "raw")

        @property
        def processed_folder(self):
            return str(Path(self.root) / "FashionMNIST" / "processed")

    CourseFashionMNIST.mirrors = [
        "https://fashion-mnist.s3.eu-central-1.amazonaws.com/",
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    ]
    return CourseFashionMNIST(
        root=str(root), train=train, download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    )


def make_loaders(root="data", train_size=10000, val_size=2000, batch_size=64,
                 seed=2026, download=True, device=torch.device("cpu")):
    """仅从官方 60,000 张训练图像中产生互不重叠的训练/验证子集。"""
    full = _fashion(root, train=True, download=download)
    if min(train_size, val_size, batch_size) < 1 or train_size + val_size > len(full):
        raise ValueError("样本数和 batch_size 必须为正，且训练+验证不超过 60,000。")
    ids = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_ids, val_ids = ids[:train_size], ids[train_size:train_size + val_size]
    train_set, val_set = Subset(full, train_ids), Subset(full, val_ids)
    options = dict(batch_size=batch_size, num_workers=0, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **options)
    val_loader = DataLoader(val_set, shuffle=False, **options)
    return train_loader, val_loader


def make_test_loader(root="data", batch_size=64, download=False,
                     device=torch.device("cpu")):
    """仅在最终配置确定后调用；测试数据不参加训练与选参。"""
    return DataLoader(_fashion(root, train=False, download=download),
                      batch_size=batch_size, shuffle=False, num_workers=0,
                      pin_memory=device.type == "cuda")


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    predictions, labels = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        predicted = logits.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
        if return_predictions:
            predictions.append(predicted.cpu())
            labels.append(targets.cpu())
    result = {"loss": total_loss / total, "acc": total_correct / total}
    if return_predictions:
        result.update(pred=torch.cat(predictions), true=torch.cat(labels))
    return result


def fit(model, train_loader, val_loader, train_one_epoch, device, epochs=3,
        lr=1e-3, checkpoint="outputs/baseline_best.pt", seed=2026):
    """逐轮训练与验证，保存并恢复验证准确率最高轮次的模型参数。"""
    if epochs < 1 or lr <= 0:
        raise ValueError("epochs 必须为正整数，lr 必须大于 0。")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_acc, best_state = -1.0, None
    path = Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        start = now(device)
        train = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device)
        seconds = now(device) - start
        row = {"epoch": epoch, "train_loss": train["loss"], "train_acc": train["acc"],
               "val_loss": val["loss"], "val_acc": val["acc"], "seconds": seconds}
        history.append(row)
        print(f"epoch {epoch:02d} | train loss {row['train_loss']:.4f} "
              f"acc {row['train_acc']:.2%} | val loss {row['val_loss']:.4f} "
              f"acc {row['val_acc']:.2%} | {seconds:.2f}s")
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            # clone 避免 CPU 模式下 state_dict 随后续训练一起变化。
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "epoch": epoch,
                        "val_acc": best_acc, "lr": lr, "seed": seed,
                        "train_size": len(train_loader.dataset),
                        "val_size": len(val_loader.dataset),
                        "batch_size": train_loader.batch_size,
                        "model_name": type(model).__name__,
                        "class_names": CLASS_NAMES,
                        "normalization": {"mean": [0.5], "std": [0.5]},
                        "torch_version": str(torch.__version__),
                        "device": str(device)}, path)
    if best_state is None:
        raise RuntimeError("未生成有效的验证结果，请检查输入、损失与训练配置。")
    model.load_state_dict(best_state)
    return history


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_batch(loader, count=12):
    images, labels = next(iter(loader))
    count = min(count, images.size(0))
    fig, axes = plt.subplots(2, (count + 1) // 2, figsize=(12, 4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(count):
        axes.flat[i].imshow(images[i, 0] * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
        axes.flat[i].set_title(CLASS_NAMES[labels[i].item()], fontsize=9)
    plt.tight_layout()
    plt.show()


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
    x = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key in ["train_loss", "val_loss"]:
        axes[0].plot(x, [r[key] for r in history], marker="o", label=key)
    for key in ["train_acc", "val_acc"]:
        axes[1].plot(x, [r[key] for r in history], marker="o", label=key)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross entropy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    _finish_plot(fig, save_path, show)
    return fig


def plot_confusion(true, pred, save_path=None, show=True):
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
            items.append((images[i, 0], labels[i].item(), predictions[i].item()))
            if len(items) == count:
                break
        if len(items) == count:
            break
    if not items:
        print("本次数据中未找到误分类样例。")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassified examples", ha="center", va="center")
        ax.axis("off")
        _finish_plot(fig, save_path, show)
        return
    fig, axes = plt.subplots(1, len(items), figsize=(2 * len(items), 2.8), squeeze=False)
    for ax, (image, target, predicted) in zip(axes.flat, items):
        ax.imshow(image * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"T: {CLASS_NAMES[target]}\nP: {CLASS_NAMES[predicted]}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    _finish_plot(fig, save_path, show)
