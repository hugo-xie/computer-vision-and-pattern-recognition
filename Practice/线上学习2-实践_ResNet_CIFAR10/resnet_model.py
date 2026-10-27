"""适配 CIFAR-10 的完整 ResNet-18 和单轮训练代码。"""
import torch
from torch import nn


class BasicBlock(nn.Module):
    """两层 3×3 卷积与残差连接；改变尺寸时使用 1×1 投影。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """输入 [N,3,32,32]，输出 [N,10]；参数从随机初始化开始训练。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        # CIFAR-10 保持 32×32 输入，首层使用 3×3、stride=1，并省去初始最大池化。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, images):
        features = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(images)))))
        # CrossEntropyLoss 直接接收 logits，网络末尾无需 Softmax。
        return self.fc(torch.flatten(self.avgpool(features), 1))


def train_one_epoch(model, loader, criterion, optimizer, device):
    """执行一轮训练，按实际样本数计算平均损失与准确率。"""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        if step % 50 == 0 or step == len(loader):
            print(f"  batch {step}/{len(loader)} | loss {total_loss / total:.4f} "
                  f"| accuracy {total_correct / total:.2%}", flush=True)
    if total == 0:
        raise ValueError("训练数据为空。")
    return {"loss": total_loss / total, "acc": total_correct / total}
