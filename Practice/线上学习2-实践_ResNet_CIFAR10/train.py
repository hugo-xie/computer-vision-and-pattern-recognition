"""从随机初始化开始训练 CIFAR-10 ResNet-18，并保存最佳验证参数。"""
import argparse
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("需要大于 0 的整数")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("需要大于 0 的有限数值")
    return number


def run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("仅可使用英文字母、数字、下划线和连字符")
    return value


def seed_value(value):
    number = int(value)
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError("seed 范围为 0 到 2**32-1")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--train-size", type=positive_int, default=45000)
    parser.add_argument("--val-size", type=positive_int, default=5000)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--lr", type=positive_float, default=0.1)
    parser.add_argument("--seed", type=seed_value, default=2026)
    parser.add_argument("--run-name", type=run_name, default="resnet18")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-download", action="store_true", help="只使用现有完整数据缓存")
    parser.add_argument("--num-threads", type=positive_int, default=4)
    args = parser.parse_args()
    if args.train_size + args.val_size > 50000:
        parser.error("--train-size 与 --val-size 之和不能超过 50000")
    return args


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import torch
    import torchvision
    from resnet_common import (CIFAR_MEAN, CIFAR_STD, set_seed, select_device,
                               make_loaders, fit, count_parameters, plot_history)
    from resnet_model import ResNet18, train_one_epoch

    torch.set_num_threads(args.num_threads)
    device = select_device(args.device)
    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Python:", sys.version.split()[0], "| interpreter:", sys.executable, flush=True)
    print("torch:", torch.__version__, "| torchvision:", torchvision.__version__, flush=True)
    print("actual device:", device, "| data:", data_root, flush=True)
    train_loader, val_loader = make_loaders(data_root, args.train_size, args.val_size,
                                          args.batch_size, args.seed, not args.no_download, device)
    if not set(train_loader.dataset.indices).isdisjoint(val_loader.dataset.indices):
        raise RuntimeError("训练与验证样本发生重叠。")
    model = ResNet18().to(device)
    print("samples train / val:", len(train_loader.dataset), len(val_loader.dataset), flush=True)
    print("trainable parameters:", count_parameters(model), flush=True)
    checkpoint_path = output_dir / f"{args.run_name}_best.pt"
    history_path = output_dir / f"{args.run_name}_history.json"
    config_path = output_dir / f"{args.run_name}_config.json"
    curves_path = output_dir / f"{args.run_name}_curves.png"
    config = {
        "run_name": args.run_name, "model": "ResNet18", "dataset": "CIFAR-10",
        "seed": args.seed, "train_size": args.train_size, "val_size": args.val_size,
        "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
        "optimizer": "SGD", "momentum": 0.9, "weight_decay": 5e-4,
        "scheduler": "CosineAnnealingLR", "pretrained": False,
        "normalization": {"mean": list(CIFAR_MEAN), "std": list(CIFAR_STD)},
        "augmentation": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip"],
        "device_requested": args.device, "device": str(device),
        "data_root": str(data_root), "output_dir": str(output_dir),
        "num_threads": args.num_threads, "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "parameters": count_parameters(model),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    history = fit(model, train_loader, val_loader, train_one_epoch, device,
                  epochs=args.epochs, lr=args.lr, checkpoint=checkpoint_path, seed=args.seed)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_history(history, save_path=curves_path, show=False)
    best = max(history, key=lambda row: row["val_acc"])
    print(f"best validation accuracy: {best['val_acc']:.2%} | epoch: {best['epoch']}")
    print("checkpoint:", checkpoint_path)
    print("history:", history_path)
    print("configuration:", config_path)
    print("training curves:", curves_path)
    print("训练完成，加载最佳参数进行独立测试：")
    print(f'python test.py --device {device} --checkpoint "{checkpoint_path}" '
          f'--data-root "{data_root}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
