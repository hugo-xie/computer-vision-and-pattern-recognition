"""完整的 CNN 模型和单轮训练函数，可由 Notebook 或命令行直接调用。"""
import torch
from torch import nn


class SimpleCNN(nn.Module):
    """将 [N,1,28,28] 的灰度图像映射为 [N,10] 的分类 logits。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        # CrossEntropyLoss 直接接收 logits，模型末尾无需 Softmax。
        return self.classifier(self.features(images))


def train_one_epoch(model, loader, criterion, optimizer, device):
    """完成一轮参数更新，返回按实际样本数加权的平均损失和准确率。"""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise ValueError("训练数据为空。")
    return {"loss": total_loss / total, "acc": total_correct / total}
