"""U-Net 模型（Ronneberger et al., MICCAI 2015）

与原论文的两处常见改动（目前几乎所有开源实现都这样做）：
  1. 卷积用 padding=1，输出与输入同尺寸，不需要原论文的"裁剪后拼接"；
  2. 每个卷积后加 BatchNorm，训练更稳定。

结构（base=32 时的通道数）：
  编码器  3 → 32 → 64 → 128 → 256 → 512(瓶颈)       每层：两次 3×3 卷积，然后 2×2 最大池化
  解码器  512 → 256 → 128 → 64 → 32 → 1(输出)       每层：2×2 转置卷积上采样，与编码器同层特征拼接(跳跃连接)，再两次 3×3 卷积
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(3×3 卷积 → BN → ReLU) × 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """编码器一层：2×2 最大池化（尺寸减半）→ DoubleConv（通道加倍）"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """解码器一层：转置卷积上采样（尺寸加倍）→ 与跳跃特征拼接 → DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 输入尺寸不是 16 的倍数时，上采样结果可能比 skip 小 1 像素，这里补齐
        dh, dw = skip.shape[2] - x.shape[2], skip.shape[3] - x.shape[3]
        if dh or dw:
            x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))   # 跳跃连接：沿通道维拼接


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = DoubleConv(in_channels, c[0])
        self.down1 = Down(c[0], c[1])
        self.down2 = Down(c[1], c[2])
        self.down3 = Down(c[2], c[3])
        self.down4 = Down(c[3], c[4])          # 瓶颈：分辨率 1/16
        self.up1 = Up(c[4], c[3])
        self.up2 = Up(c[3], c[2])
        self.up3 = Up(c[2], c[1])
        self.up4 = Up(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)   # 1×1 卷积：每个像素输出类别分数

    def forward(self, x):
        x1 = self.inc(x)      # 1
        x2 = self.down1(x1)   # 1/2
        x3 = self.down2(x2)   # 1/4
        x4 = self.down3(x3)   # 1/8
        x5 = self.down4(x4)   # 1/16
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)   # 未经 sigmoid 的 logits，形状 [B, num_classes, H, W]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 快速自检：python model.py
    m = UNet(base=32)
    y = m(torch.randn(2, 3, 160, 160))
    print("输出形状:", tuple(y.shape), " 参数量: %.2fM" % (count_params(m) / 1e6))
