"""下载并检查完整 Fashion-MNIST 缓存，确保训练与测试文件均已准备。"""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    from cnn_common import _fashion
    for train in [True, False]:
        dataset = _fashion(args.root, train=train, download=True)
        print("training" if train else "test", len(dataset), "images")
    print("完整缓存已准备好:", (args.root / "FashionMNIST" / "raw").resolve())
    print("raw 目录应包含四个无 .gz 后缀的原始文件，可直接用于离线读取。")


if __name__ == "__main__":
    main()
