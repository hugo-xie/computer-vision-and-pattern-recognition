"""下载并检查 CIFAR-10 完整缓存，包含训练与测试批次。"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    from resnet_common import _cifar
    root = args.root.expanduser().resolve()
    for train in (True, False):
        dataset = _cifar(root, train=train, download=True)
        print("training" if train else "test", len(dataset), "images", flush=True)
    print("完整缓存:", root / "cifar-10-batches-py")
    print("保留整个 cifar-10-batches-py 目录即可使用 --no-download 离线训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
