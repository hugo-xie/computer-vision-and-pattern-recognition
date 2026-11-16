# U-Net 图像分割：训练与测试（Oxford-IIIT Pet 宠物前景分割）

一套完整、可直接运行的 U-Net 二值分割代码：把照片中的**猫/狗（前景）**和**背景**逐像素分开。
只依赖 PyTorch + NumPy + Pillow，CPU / NVIDIA GPU / Apple 芯片(MPS) 都能跑。

```
unet_seg/
├── download_data.py   下载并解压数据集（约 800 MB，支持断点续传）
├── dataset.py         数据读取、train/val/test 划分、数据增强
├── model.py           U-Net 模型（编码器-解码器 + 跳跃连接）
├── utils.py           BCE + Dice 损失、IoU / Dice 指标、设备选择
├── train.py           训练（余弦学习率、混合精度、保存最佳模型、断点续训）
├── test.py            测试集评估 + 可视化 + 单张图片预测
└── requirements.txt
```

---

## 1. 数据集：Oxford-IIIT Pet

| 项目 | 说明 |
|---|---|
| 内容 | 37 个品种的猫和狗，共 7349 张照片，每张都有逐像素标注 |
| 划分 | 官方 trainval 3680 张（本代码再按固定种子切出 10% = 368 张做验证）、test 3669 张 |
| 标注 | `trimaps/*.png`，像素值 1 = 宠物，2 = 背景，3 = 边缘过渡带（本代码把 1 和 3 都算前景） |
| 主页 | https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| 许可 | CC BY-SA 4.0 |

为什么选它：只有两个文件、下载一次即可，**不需要注册**；任务是二分类（前景/背景），目标大而明显，几轮训练就能看到效果，适合入门。

### 下载方式（任选其一）

**方式 A：一键脚本（推荐）**

```bash
python download_data.py            # 下载并解压到 data/oxford-iiit-pet/
```

**方式 B：手动下载**

```bash
mkdir -p data/oxford-iiit-pet && cd data/oxford-iiit-pet
wget https://thor.robots.ox.ac.uk/pets/images.tar.gz        # 约 755 MB
wget https://thor.robots.ox.ac.uk/pets/annotations.tar.gz   # 约 18 MB
tar -xzf images.tar.gz && tar -xzf annotations.tar.gz
```

Windows 没有 wget 时，把上面两个链接粘贴到浏览器下载，再用 7-Zip 解压到 `data/oxford-iiit-pet/`。

**方式 C：用 torchvision 下载（已装 torchvision 时）**

```python
from torchvision.datasets import OxfordIIITPet
OxfordIIITPet(root="data", download=True)   # 会生成 data/oxford-iiit-pet/，目录结构与上面相同
```

解压后的目录结构应为：

```
data/oxford-iiit-pet/
├── images/                 Abyssinian_1.jpg …（7390 张 jpg）
└── annotations/
    ├── trainval.txt
    ├── test.txt
    └── trimaps/            Abyssinian_1.png …
```

---

## 2. 环境

```bash
pip install -r requirements.txt
# 有 NVIDIA 显卡时，建议按 https://pytorch.org 的命令安装对应 CUDA 版本的 torch
```

Python ≥ 3.8，PyTorch ≥ 1.10。

---

## 3. 训练

```bash
python train.py                     # 默认：160×160，base=32（7.8M 参数），40 轮，batch 16
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--epochs` | 40 | 训练轮数 |
| `--img-size` | 160 | 输入缩放到的边长（需为 16 的倍数） |
| `--base` | 32 | 第一层通道数；16 → 1.9M 参数，32 → 7.8M，64 → 31M |
| `--batch-size` | 16 | 显存不够就调小 |
| `--lr` | 1e-3 | AdamW 初始学习率（余弦退火到 0） |
| `--resume` | — | 从 `checkpoints/last.pt` 继续训练 |
| `--time-limit` | 0 | 跑满 N 分钟后保存并退出（配合 `--resume` 分段训练） |
| `--subset` | 0 | 只用前 N 张训练图，CPU 上快速试跑 |

只有 CPU 时，可以先用轻量设置看到效果：

```bash
python train.py --img-size 128 --base 16 --subset 1000 --epochs 8 --workers 2
```

训练输出：`checkpoints/best.pt`（验证 IoU 最高的模型）、`checkpoints/last.pt`、`checkpoints/log.csv`。

---

## 4. 测试

```bash
python test.py                                 # 在官方测试集 3669 张上评估 best.pt
python test.py --image path/to/your_cat.jpg    # 对任意一张图片做预测
```

输出到 `results/`：
- `test_metrics.txt`：平均 IoU、Dice、前景/背景 IoU、像素准确率、困难样本数
- `test_vis.png`：每行“原图 | 真值 | 预测 | 叠加”
- `per_image.csv`：每张测试图的 IoU（按 IoU 排序就能找到最差的样本做误差分析）
- 单张预测时：`xxx_mask.png`（二值掩码）与 `xxx_overlay.png`（叠加图）

---

## 5. 已验证的运行结果（仅 CPU 的轻量设置）

在 4 核 CPU 上用上面的轻量命令（128×128，base=16，1.9M 参数，只用 1000 张训练图，8 轮，每轮约 64 秒）实测：

| 轮次 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 验证 IoU | 0.599 | 0.640 | 0.667 | 0.666 | 0.734 | 0.729 | 0.749 | 0.752 |

测试集（3669 张）：**平均 IoU 0.760，平均 Dice 0.853，像素准确率 0.881**。
这只是快速试跑的结果。用全部训练图和默认设置（160×160、base=32、40 轮）训练，指标还会更高。

---

## 6. 代码要点（对照课件）

- **跳跃连接**（`model.py` 的 `Up.forward`）：解码器上采样后，与编码器同分辨率的特征沿通道维拼接（`torch.cat`），把编码器的细节信息交给解码器，边界因此更准。
- **转置卷积上采样**：`ConvTranspose2d(kernel=2, stride=2)`，分辨率加倍。
- **损失 = 交叉熵 + Dice**（`utils.py`）：交叉熵逐像素地约束，Dice 直接优化前景的重叠度，前景占比小时也不容易全判成背景。
- **标签缩放用最近邻**（`dataset.py`）：标签是类别编号，双线性插值会产生不存在的中间值。
- **评价指标**：IoU = |P∩G| / |P∪G|，Dice = 2|P∩G| / (|P|+|G|)。

### 可以做的小实验

1. 去掉跳跃连接（`Up.forward` 里不拼接 `skip`，同时把 `DoubleConv` 的输入通道改为 `in_ch // 2`），比较 IoU 和边界效果。
2. 损失只用 BCE、只用 Dice、两者相加，比较结果。
3. 改变 `--base`（16 / 32 / 64）或 `--img-size`（96 / 160 / 224），看参数量、速度和精度怎么变。
4. 打开 `per_image.csv`，看 IoU 最低的 10 张图，总结 U-Net 在什么情况下会失败。
