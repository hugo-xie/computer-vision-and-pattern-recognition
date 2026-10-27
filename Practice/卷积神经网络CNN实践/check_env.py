"""运行 python check_env.py --device cpu / cuda / mps / auto 验收训练环境。"""
import argparse
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    print("Python:", sys.version.split()[0])
    print("解释器:", sys.executable)
    print("系统:", platform.platform(), "架构:", platform.machine())
    try:
        import torch
        import torchvision
    except Exception as exc:
        print(f"PyTorch / torchvision 导入失败: {type(exc).__name__}: {exc}")
        print("请按 README 安装匹配的版本，并确认当前 Python 属于 cv-cnn 环境。")
        return 1
    print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
    print("PyTorch 构建的 CUDA runtime:", torch.version.cuda)
    print("CUDA 可用:", torch.cuda.is_available())
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print("MPS 可用:", has_mps)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("指定了 cuda，但当前不可用。先执行 nvidia-smi，再检查 README 的 CUDA 安装分支。")
        return 2
    if device_name == "mps" and not has_mps:
        print("指定了 mps，但当前不可用。可先用 --device cpu 完成本课。")
        return 2
    device = torch.device(device_name)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("实际验收设备:", device)
    try:
        torch.manual_seed(2026)
        # 合成数据仅用于设备验收，不代表 Fashion-MNIST 准确率。
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 10),
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images = torch.randn(8, 1, 28, 28, device=device)
        labels = torch.randint(0, 10, (8,), device=device)
        before = model[-1].weight.detach().clone()
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(model(images), labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        if not torch.isfinite(loss).item() or torch.equal(before, model[-1].weight.detach()):
            raise RuntimeError("损失非有限值或参数没有更新。")
        print(f"PASS: 前向、损失、反向传播和参数更新成功，loss={loss.item():.4f}")
        print("数据和模型均位于", device, "；环境验收通过。")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        print("参见 README 故障排查；可先用 --device cpu 复测。")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
