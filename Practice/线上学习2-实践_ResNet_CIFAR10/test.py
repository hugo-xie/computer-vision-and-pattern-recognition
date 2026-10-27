"""加载已训练的 ResNet-18，在 CIFAR-10 官方测试集上评估。"""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("需要大于 0 的整数")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/resnet18_best.pt")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--download", action="store_true", help="允许下载缺少的测试数据")
    parser.add_argument("--num-threads", type=positive_int, default=4)
    args = parser.parse_args()
    if not args.checkpoint.expanduser().is_file():
        parser.error(f"模型文件不存在：{args.checkpoint}。先运行 train.py 完成训练。")
    return args


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import torch
    from torch import nn
    from resnet_common import (CLASS_NAMES, select_device, make_test_loader, evaluate,
                               count_parameters, plot_confusion, show_mistakes)
    from resnet_model import ResNet18

    torch.set_num_threads(args.num_threads)
    device = select_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("模型文件应由 train.py 保存，并包含 model_state。")
    model = ResNet18().to(device)
    model.load_state_dict(checkpoint["model_state"])
    test_loader = make_test_loader(data_root, args.batch_size, args.download, device)
    result = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, return_predictions=True)
    tag = checkpoint_path.stem
    if tag.endswith("_best"):
        tag = tag[:-5]
    metrics_path = output_dir / f"{tag}_test_metrics.json"
    confusion_path = output_dir / f"{tag}_confusion.png"
    mistakes_path = output_dir / f"{tag}_mistakes.png"
    metrics = {
        "checkpoint": str(checkpoint_path), "device": str(device),
        "test_samples": len(test_loader.dataset), "test_loss": result["loss"],
        "test_accuracy": result["acc"], "best_epoch": checkpoint.get("epoch"),
        "best_val_accuracy": checkpoint.get("val_acc"),
        "train_size": checkpoint.get("train_size"), "val_size": checkpoint.get("val_size"),
        "parameters": count_parameters(model), "class_names": CLASS_NAMES,
        "torch_version": str(torch.__version__),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_confusion(result["true"], result["pred"], save_path=confusion_path, show=False)
    show_mistakes(model, test_loader, device, save_path=mistakes_path, show=False)
    print("checkpoint:", checkpoint_path)
    print("actual device:", device, "| test samples:", len(test_loader.dataset))
    print(f"test loss: {result['loss']:.4f} | test accuracy: {result['acc']:.2%}")
    print("test metrics:", metrics_path)
    print("confusion matrix:", confusion_path)
    print("misclassified examples:", mistakes_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
