# ResNet-18 与 CIFAR-10：训练和测试指南

使用完整的 PyTorch 代码训练 ResNet-18，完成 CIFAR-10 彩色图像分类。模型定义、数据处理、训练、验证、参数保存和独立测试代码均已提供，按顺序运行即可。

复用已有的 `cv-cnn` 环境。在解压后的项目目录中执行 `conda activate cv-cnn`，无需重新创建环境或安装另一套 PyTorch。已有环境中需要包含 torch、torchvision、NumPy、Matplotlib、JupyterLab 和 ipykernel；环境安装方法沿用卷积神经网络实践资料。

操作步骤见 [实践任务](任务书_ResNet_CIFAR10实践课.md)，交互式运行入口为 [Notebook](ResNet_CIFAR10实践课.ipynb)。

## 1. 文件结构

```text
实践课_ResNet_CIFAR10/
├── README.md                           # 操作、参数与常见问题
├── 任务书_ResNet_CIFAR10实践课.md       # 运行任务与结果检查
├── ResNet_CIFAR10实践课.ipynb           # 完整交互式训练与测试
├── resnet_model.py                     # 完整 ResNet-18 与单轮训练
├── resnet_common.py                    # 数据、设备、验证、保存与绘图
├── train.py                            # 训练入口
├── test.py                             # 独立测试入口
├── prepare_data.py                     # 下载并检查 CIFAR-10 数据
├── requirements.txt                    # 通用依赖清单
└── 参考资料/
    ├── 参考说明.md                     # 网络与代码结构说明
    ├── 验证记录.md                     # 验证范围与运行结果
    └── 生成Notebook.py                 # 从源码重新生成 Notebook
```

`data/` 和 `outputs/` 在运行时创建。初始资料包不包含 CIFAR-10 数据、预训练权重或已训练模型；网络从随机初始化开始训练。

## 2. 数据与模型

CIFAR-10 的每张图像为 `32×32` 的 RGB 彩色图，包含 10 个类别：

| 标签 | 英文名称 | 中文名称 |
|---:|---|---|
| 0 | airplane | 飞机 |
| 1 | automobile | 汽车 |
| 2 | bird | 鸟 |
| 3 | cat | 猫 |
| 4 | deer | 鹿 |
| 5 | dog | 狗 |
| 6 | frog | 青蛙 |
| 7 | horse | 马 |
| 8 | ship | 船 |
| 9 | truck | 卡车 |

官方提供 50,000 张训练图像与 10,000 张测试图像。本项目默认将官方训练数据划分为 45,000 张训练图像和 5,000 张验证图像，索引互不重叠。测试数据只在最终测试时读取，不参与训练和模型选择。

模型采用适配 CIFAR-10 的 ResNet-18：首层为 `3×3`、步长为 1 的卷积，省去 ImageNet 版本最前面的最大池化；四个残差阶段分别含 2 个基本残差块，通道数为 64、128、256、512。全局平均池化后通过全连接层输出 10 类 logits。

本项目直接使用 `32×32` 图像，不放大到 `224×224`，也不下载 ImageNet 预训练参数。网络末尾不加 Softmax，训练损失为 `CrossEntropyLoss`。

训练图像使用随机裁剪和随机水平翻转。验证、测试图像不使用随机增强。RGB 各通道分别使用下面的均值与标准差，训练、验证和测试沿用同一组数值：

```python
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
```

## 3. 准备数据

在项目目录中执行：

```bash
conda activate cv-cnn
python prepare_data.py
```

程序通过 torchvision 从 CIFAR-10 官方地址下载数据，并检查文件完整性。压缩包约 163 MiB，下载与解压请预留至少 1 GB 可用空间。已有完整缓存时会直接使用缓存。

下载后的主要目录为：

```text
data/
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

离线运行时，先在可联网的电脑上运行数据准备命令，再复制完整的 `data/cifar-10-batches-py/` 目录。训练命令附加 `--no-download` 后，仅使用已有缓存：

```bash
python train.py --device cpu --train-size 2000 --val-size 500 --epochs 3 --run-name cpu_run --no-download
```

压缩文件不能替代已解压的数据目录；数据准备失败时，不要通过修改文件名或跳过完整性检查继续运行。

## 4. 运行 Notebook

在项目目录、已激活的 `cv-cnn` 环境中执行：

```bash
python -m jupyterlab
```

打开 `ResNet_CIFAR10实践课.ipynb`，选择 `Python (cv-cnn)` 内核，从第一个单元开始顺序运行。Notebook 已包含完整模型和训练代码，同时调用同目录的公共工具，因此需要保留整个目录结构。

默认配置与训练脚本一致：

```python
DEVICE_CHOICE = "cpu"
TRAIN_SIZE = 45000
VAL_SIZE = 5000
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.1
RUN_NAME = "resnet18"
```

CPU 初次运行时，可将 `TRAIN_SIZE`、`VAL_SIZE`、`EPOCHS`、`RUN_NAME` 分别改为 `2000`、`500`、`3`、`"cpu_run"`。这用于运行完整流程，图像数量和轮数较少时，分类效果也会受到限制。

使用兼容的 NVIDIA GPU 时，将 `DEVICE_CHOICE` 改为 `"cuda"`；兼容 Mac 可使用 `"mps"`。`"auto"` 按 CUDA、MPS、CPU 的顺序选择可用设备。每次运行以程序打印的实际设备为准；显式指定的 GPU 不可用时会报错。

依次运行数据读取、模型检查、训练、模型保存与独立测试单元。完整运行后，Notebook 中可查看损失和准确率曲线、混淆矩阵以及误分类图像。

## 5. 使用脚本训练

### 5.1 检查运行流程

```bash
python train.py --device cpu --train-size 128 --val-size 64 --epochs 1 --run-name smoke
python test.py --device cpu --checkpoint outputs/smoke_best.pt
```

该配置只检查数据读取、前向计算、反向传播、参数保存与重新加载。它不用于判断模型的分类能力，测试仍会读取官方完整的 10,000 张测试图像。

### 5.2 CPU 小规模训练

```bash
python train.py --device cpu --train-size 2000 --val-size 500 --epochs 3 --run-name cpu_run
python test.py --device cpu --checkpoint outputs/cpu_run_best.pt
```

ResNet-18 的计算量高于前一个小型 CNN。CPU 可以运行训练与测试，但完整数据、多轮训练的耗时明显增加。需要在完整数据上训练时，可直接使用下一节的默认规模，并根据实际算力选择轮数。

### 5.3 GPU 完整训练

在 CUDA 已可用的环境中执行：

```bash
python train.py --device cuda
python test.py --device cuda --checkpoint outputs/resnet18_best.pt
```

默认使用 45,000 张训练图像、5,000 张验证图像，训练 20 轮。增加训练轮数的示例：

```bash
python train.py --device cuda --epochs 100 --run-name resnet18_100
python test.py --device cuda --checkpoint outputs/resnet18_100_best.pt
```

MPS 的调用方式相同，将 `--device cuda` 改为 `--device mps`。保存的参数可以在另一种设备上加载，例如 GPU 训练后使用 `--device cpu` 测试。

每次执行训练命令都会重新初始化模型和优化器；增加 `--epochs` 会启动一次新的训练，不会从已有模型继续训练。同名输出会被覆盖，使用不同的 `--run-name` 可以保留不同运行的结果。

## 6. 训练参数与日志

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--device` | `cpu` | 运行设备，可选 `cpu`、`cuda`、`mps`、`auto` |
| `--train-size` | 45000 | 从官方训练数据中取出的训练样本数 |
| `--val-size` | 5000 | 从官方训练数据中取出的验证样本数 |
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 128 | 每批样本数 |
| `--lr` | 0.1 | 初始学习率 |
| `--seed` | 2026 | 随机种子 |
| `--run-name` | `resnet18` | 输出文件名前缀 |
| `--num-threads` | 4 | CPU 运算线程数 |
| `--data-root` | 项目中的 `data/` | 数据缓存路径 |
| `--output-dir` | 项目中的 `outputs/` | 模型与结果保存路径 |
| `--no-download` | 未开启 | 禁止训练入口下载数据 |

训练与验证样本数均须大于 0，二者之和不能超过 50,000。运行名称使用英文字母、数字、下划线或连字符。

训练使用 SGD，动量为 `0.9`、权重衰减为 `5e-4`。余弦学习率调度器在设定的训练轮数内逐步降低学习率。改变总轮数也会改变学习率随轮次变化的过程。

每轮日志包括训练损失、训练准确率、验证损失、验证准确率、学习率与耗时。训练模式使用 `model.train()`；验证和测试使用 `model.eval()` 与 `torch.no_grad()`，BatchNorm 按评估模式工作。

程序自动保存验证准确率最高轮次的模型。准确率相同时保留较早轮次。训练结束后恢复这组最佳参数，独立测试入口也加载同一文件。

完整参数列表可运行：

```bash
python train.py --help
python test.py --help
python prepare_data.py --help
```

## 7. 查看输出

默认运行名称为 `resnet18`：

| 文件 | 内容 |
|---|---|
| `outputs/resnet18_best.pt` | 最佳验证轮次的模型参数与必要配置 |
| `outputs/resnet18_config.json` | 本次运行配置与版本 |
| `outputs/resnet18_history.json` | 每轮训练、验证指标、学习率和耗时 |
| `outputs/resnet18_curves.png` | 损失、准确率与学习率曲线 |
| `outputs/resnet18_test_metrics.json` | 独立测试损失、准确率与模型信息 |
| `outputs/resnet18_confusion.png` | 测试集混淆矩阵 |
| `outputs/resnet18_mistakes.png` | 误分类图像与真实、预测类别 |

更改运行名称后，上表中的文件前缀随之改变。前四项由训练入口生成，后三项由测试入口生成。图像显示前会进行反标准化，还原 RGB 色彩。

检查训练损失是否总体下降、验证准确率是否提高，以及最佳参数是否能重新加载并完成测试。训练准确率高而验证准确率低，可能表明过拟合。训练增强会增加训练样本的难度，训练准确率与验证准确率的高低关系也受增强和 BatchNorm 模式影响。

准确率受数据规模、训练轮数、学习率、设备和随机性影响。小规模运行的结果不能代表完整训练效果；保留实际配置和指标，不以某个固定准确率作为运行成功的唯一依据。

## 8. 常见问题

**数据下载失败或校验失败。** 检查网络连接与可用磁盘空间后重新运行 `python prepare_data.py`。也可按第 3 节复制另一台电脑上准备好的完整缓存。数据来源为 [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)。

**提示缺少 `resnet_common` 或模型文件。** 保留解压后的目录结构，从包含 `train.py` 和 `resnet_common.py` 的目录启动 Notebook。不要只复制一个 `.ipynb` 文件。

**CUDA 或 MPS 不可用。** 检查当前终端与 Notebook 是否使用已有的 `cv-cnn` 环境，按卷积神经网络实践资料检查相应设备。当前运行可将设备设为 `cpu`。

**显存不足。** 减小 `--batch-size`，例如改为 64 或 32。Notebook 对应修改 `BATCH_SIZE`，随后从配置单元开始重新运行。

**训练较慢。** 先使用小规模配置确认流程，再选择 GPU 或增加数据规模。CPU 线程过多也可能降低性能，可通过 `--num-threads` 调整。

**找不到模型文件。** 训练完成后确认实际保存路径；测试命令的 `--checkpoint` 必须与训练时的 `--run-name` 对应。相对路径以执行命令时所在目录为基准。

**只想重新测试已有模型。** 直接运行 `test.py`。默认要求本地测试数据完整；确需下载缺少的数据时，为测试命令添加 `--download`。无需重新运行训练。

网络结构与保存行为见 [参考说明](参考资料/参考说明.md)，代码验证范围见 [验证记录](参考资料/验证记录.md)。
