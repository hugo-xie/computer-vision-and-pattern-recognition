# ResNet-18 图像分类实践：运行、训练与测试

使用完整代码训练 ResNet-18，完成 CIFAR-10 的 10 类彩色图像分类。依次完成数据读取、模型训练、最佳参数保存和独立测试，查看训练曲线与分类结果。

模型、训练循环、验证与测试代码均已提供，可运行 [Notebook](ResNet_CIFAR10实践课.ipynb)，也可使用独立脚本。环境沿用 `cv-cnn`，操作与参数说明见 [README](README.md)。

## 一、数据与模型

| 项目 | 内容 |
|---|---|
| 图像 | RGB 彩色图像，`32×32` |
| 类别 | airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck |
| 默认训练集 | 45,000 张，用于更新参数 |
| 默认验证集 | 5,000 张，用于选择最佳模型 |
| 测试集 | 官方独立的 10,000 张图像 |
| 模型 | 适配 CIFAR-10 的 ResNet-18，从随机参数开始训练 |
| 损失 | `CrossEntropyLoss` |
| 优化 | SGD，动量 `0.9`，权重衰减 `5e-4`，余弦学习率调度 |

训练与验证图像由官方 50,000 张训练数据按固定种子划分，索引互不重叠。训练图像使用随机裁剪和水平翻转，验证与测试图像使用确定性预处理。测试集不参与训练和模型选择。

ResNet-18 的首层为 `3×3`、步长为 1 的卷积，最前面不使用最大池化。四个残差阶段各含 2 个基本残差块，经过全局平均池化和全连接层输出 10 类 logits。输入 batch 的形状为 `N×3×32×32`，输出形状为 `N×10`。

## 二、准备数据

在项目目录中运行：

```bash
conda activate cv-cnn
python prepare_data.py
```

确认训练数据和测试数据均已准备完成。数据压缩下载量约 163 MiB，缓存目录为 `data/cifar-10-batches-py/`。已有完整数据时直接使用缓存；离线复制方式见 README。

## 三、运行完整训练

### 方式 A：Notebook

打开 `ResNet_CIFAR10实践课.ipynb`，选择 `Python (cv-cnn)` 内核，按单元顺序运行。

CPU 小规模运行可使用：

```python
DEVICE_CHOICE = "cpu"
TRAIN_SIZE = 2000
VAL_SIZE = 500
EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.1
RUN_NAME = "cpu_run"
```

使用兼容的 GPU 进行完整训练时，将设备改为 `"cuda"` 或 `"mps"`，训练样本数设为 45,000、验证样本数设为 5,000，训练 20 轮或更多。配置中的 `RUN_NAME` 决定输出文件前缀。

依次运行数据读取、图像预览、模型构建、训练、最佳模型加载与最终测试单元。无需补充网络层或训练代码。

### 方式 B：命令行

CPU 小规模训练：

```bash
python train.py --device cpu --train-size 2000 --val-size 500 --epochs 3 --run-name cpu_run
```

CUDA 完整训练：

```bash
python train.py --device cuda
```

默认配置为 45,000 张训练图像、5,000 张验证图像、20 轮、batch size 128、初始学习率 0.1。需要增加训练轮数时，可运行：

```bash
python train.py --device cuda --epochs 100 --run-name resnet18_100
```

每次执行训练入口都会从随机初始化开始，同名输出会被覆盖。增加轮数时使用新的运行名称，可保留已有结果。小规模配置用于运行和检查完整流程，分类效果受图像数量与训练轮数限制。

只检查流程时，可使用 128 张训练图像、64 张验证图像和 1 轮训练：

```bash
python train.py --device cpu --train-size 128 --val-size 64 --epochs 1 --run-name smoke
```

## 四、加载最佳模型并测试

CPU 小规模训练完成后执行：

```bash
python test.py --device cpu --checkpoint outputs/cpu_run_best.pt
```

默认完整训练完成后执行：

```bash
python test.py --device cuda --checkpoint outputs/resnet18_best.pt
```

使用其他运行名称时，对应修改参数文件路径，例如 `outputs/resnet18_100_best.pt`。模型参数可映射到另一种设备，GPU 训练后的模型也可使用 `--device cpu` 测试。

测试代码重新创建网络并加载最佳验证轮次的参数，评估官方 10,000 张测试图像。测试不更新参数，并保存损失、准确率、混淆矩阵和误分类图像。选择训练轮数、学习率等配置时依据验证集表现，配置确定后运行独立测试。

## 五、查看结果

默认运行名称为 `resnet18`，输出位于 `outputs/`：

| 文件 | 内容 |
|---|---|
| `resnet18_best.pt` | 最佳模型参数 |
| `resnet18_config.json` | 运行配置、设备与版本 |
| `resnet18_history.json` | 每轮训练与验证指标、学习率、耗时 |
| `resnet18_curves.png` | 损失、准确率与学习率曲线 |
| `resnet18_test_metrics.json` | 测试指标 |
| `resnet18_confusion.png` | 测试集混淆矩阵 |
| `resnet18_mistakes.png` | 误分类图像及类别 |

训练入口生成前四项，独立测试入口生成后三项。使用 `cpu_run`、`smoke` 等其他运行名称时，文件前缀随之改变。

记录实际运行结果：

| 项目 | 运行结果 |
|---|---|
| Python / torch / torchvision 版本 | |
| 实际运行设备 | |
| 训练 / 验证 / 测试样本数 | |
| 训练轮数、batch size、初始学习率 | |
| 最佳验证准确率及对应轮次 | |
| 测试损失与测试准确率 | |
| 最佳模型文件路径 | |

观察训练损失是否总体下降、验证准确率是否改善，以及混淆矩阵中哪些类别容易混淆。训练准确率持续提高而验证表现下降时，需要结合数据规模和训练轮数判断是否过拟合。不同配置的结果以实际指标和曲线为准。

## 六、完成检查

- 训练与验证样本数、实际设备和运行配置一致。
- 完整训练能够运行，并显示各轮训练与验证指标。
- 最佳模型参数、配置、历史指标和曲线已保存。
- 保存的模型能在独立测试入口重新加载。
- 官方测试集的损失、准确率、混淆矩阵和误分类图像可查看。
- 实际配置和结果已记录，Notebook 保留按顺序运行后的输出。
