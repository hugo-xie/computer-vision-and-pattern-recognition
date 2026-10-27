"""训练 Fashion-MNIST CNN，依据验证集保存最佳模型，不计算测试集结果。"""
import argparse
import json
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
    if not (0 < number < float("inf")):
        raise argparse.ArgumentTypeError("需要大于 0 的有限数值")
    return number


def run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("名称只能使用英文字母、数字、下划线或连字符")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--train-size", type=positive_int, default=10000)
    parser.add_argument("--val-size", type=positive_int, default=2000)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-name", type=run_name, default="baseline")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-download", action="store_true", help="仅使用现有的完整数据缓存")
    parser.add_argument("--num-threads", type=positive_int, default=4, help="CPU 运算线程数")
    args = parser.parse_args()
    if args.train_size + args.val_size > 60000:
        parser.error("--train-size 与 --val-size 之和不能超过 60000")
    return args


def main():
    args = parse_args()
    # 命令行保存图片；Notebook 仍可使用交互式绘图。
    import matplotlib
    matplotlib.use("Agg")
    import torch
    import torchvision
    from cnn_common import (set_seed, select_device, make_loaders, fit,
                            count_parameters, plot_history)
    from cnn_model import SimpleCNN, train_one_epoch

    torch.set_num_threads(args.num_threads)
    device = select_device(args.device)
    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Python:", sys.version.split()[0], "| interpreter:", sys.executable)
    print("torch:", torch.__version__, "| torchvision:", torchvision.__version__)
    print("actual device:", device, "| data:", data_root)
    train_loader, val_loader = make_loaders(
        root=data_root, train_size=args.train_size, val_size=args.val_size,
        batch_size=args.batch_size, seed=args.seed, download=not args.no_download,
        device=device,
    )
    train_ids = set(train_loader.dataset.indices)
    val_ids = set(val_loader.dataset.indices)
    if not train_ids.isdisjoint(val_ids):
        raise RuntimeError("训练与验证样本发生重叠。")
    model = SimpleCNN().to(device)
    print("samples train / val:", len(train_loader.dataset), len(val_loader.dataset))
    print("trainable parameters:", count_parameters(model))
    checkpoint_path = output_dir / f"{args.run_name}_best.pt"
    history_path = output_dir / f"{args.run_name}_history.json"
    config_path = output_dir / f"{args.run_name}_config.json"
    curves_path = output_dir / f"{args.run_name}_curves.png"
    config = {
        "run_name": args.run_name, "model": "SimpleCNN", "seed": args.seed,
        "train_size": args.train_size, "val_size": args.val_size,
        "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
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
    print("训练完成。加载最佳参数进行独立测试：")
    print(f'python test.py --device {device} --checkpoint "{checkpoint_path}" '
          f'--data-root "{data_root}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
