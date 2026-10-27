# 卷积神经网络实践课：环境搭建与使用指南

使用 PyTorch 运行卷积神经网络（CNN），完成 Fashion-MNIST 服饰图像分类。模型、训练循环和测试代码均已提供，可通过 [Notebook](卷积神经网络实践课.ipynb) 顺序运行，也可使用独立的训练与测试脚本。具体操作见 [实践任务](任务书_卷积神经网络实践课.md)。

**CPU 可以运行完整的训练与测试流程。** 兼容的 NVIDIA 显卡可使用 CUDA；Apple Silicon Mac 可使用 MPS。

首次安装需要联网，并预留数 GB 磁盘空间；CUDA 版 PyTorch 下载量较大。

## 1. 文件结构

```text
实践课_卷积神经网络/
├── README.md                         # 安装、启动、设备切换与排错
├── 任务书_卷积神经网络实践课.md       # 实验操作与结果检查
├── 卷积神经网络实践课.ipynb           # 可直接运行的完整实验
├── cnn_model.py                      # 完整 CNN 与每轮训练代码
├── cnn_common.py                     # 数据、验证、绘图和模型保存
├── train.py                          # 训练入口
├── test.py                           # 测试入口
├── requirements.txt                  # 通用 Python 依赖，不包含 torch
├── check_env.py                      # 检查指定设备能否完成训练步骤
└── prepare_data.py                   # 下载并检查完整数据缓存
```

数据在首次运行时下载到 `data/`，实验模型保存在 `outputs/`。这两个目录由程序创建，初始资料包不包含整个数据集和训练结果。

运行流程为：安装依赖 → 检查环境 → 准备数据 → 训练模型 → 加载最佳参数进行测试。训练日志、曲线和模型参数保存在 `outputs/`，可用于确认训练是否完成。

## 2. 基本概念

| 名称 | 在本实验中的作用 |
|---|---|
| Anaconda | 包含 Python、Conda 等工具的发行版，用于准备科学计算环境 |
| Conda 环境 | 一套独立的 Python 和依赖包；本实验环境命名为 `cv-cnn` |
| pip | Python 包安装工具；使用 `python -m pip` 能明确安装到当前 Python |
| PyTorch / `torch` | 张量计算、神经网络、自动求导与 CPU/GPU 训练框架 |
| `torchvision` | 图像数据集、图像预处理等视觉工具；版本应与 torch 配对 |
| JupyterLab / Notebook | 在浏览器中分单元运行代码、显示图像和撰写分析的工具 |
| Jupyter 内核 | 实际执行 Notebook 代码的 Python 进程，必须选择本实验环境 |
| CUDA | NVIDIA GPU 的计算平台；AMD、Intel、Apple 显卡不使用本指南的 CUDA 安装路线 |
| MPS | PyTorch 在兼容 Mac 上使用 GPU 的后端，与 CUDA 不同 |

创建独立的 **Python 3.11** 环境，避免新旧依赖互相影响。Anaconda 自带的 Python 版本不必为 3.11，创建环境时会指定。

## 3. 安装 Anaconda

终端执行 `conda --version` 能显示版本号时，直接进入第 4 节。

### 3.1 下载与选择安装器

从 [Anaconda 官方下载页](https://www.anaconda.com/download) 下载，完整流程见 [官方安装说明](https://www.anaconda.com/docs/getting-started/anaconda/install)。安装器必须同时匹配操作系统和处理器架构。

| 电脑 | 常见安装器选择 |
|---|---|
| Windows 普通 Intel/AMD 电脑 | Windows 64-bit / x86_64 图形安装器 |
| Apple M 系列芯片 Mac | macOS Apple Silicon / arm64 安装器 |
| Intel 芯片 Mac | macOS Intel / x86_64 历史安装器，见下面的说明 |
| Linux 普通 Intel/AMD 电脑 | Linux x86_64 安装脚本 |

Mac 可以点击“苹果菜单 → 关于本机”查看芯片。Linux/macOS 也可运行 `uname -m` 查看架构。Windows ARM、Linux ARM 等其他架构需要选择支持该架构的安装器和 PyTorch 包，不能直接使用 x86_64 二进制包。

**Intel Mac：**Anaconda 官方从 2025 年 8 月起停止提供新的 Intel Mac 安装器和新包构建。请从 [官方历史下载目录](https://repo.anaconda.com/archive/) 选择适用的 `MacOSX-x86_64` 安装器，并使用本文单列的 Intel Mac PyTorch 方案；不要安装 Apple Silicon 的 arm64 安装器。

### 3.2 Windows 安装步骤

1. 双击下载的 `.exe` 安装器，阅读许可并按提示继续。
2. 一般选择 **Just Me**，无需为了本实验使用管理员模式。
3. 安装位置尽量使用简单英文路径，例如 `C:\Users\yourname\anaconda3`，避免空格和特殊字符。
4. 对是否加入系统 PATH、是否注册为默认 Python 等选项，保留安装器默认设置即可。本实验使用 Anaconda Prompt，不需要手动修改全系统 PATH。
5. 安装完成后，从开始菜单打开 **Anaconda Prompt**，输入：

   ```bash
   conda --version
   ```

能显示版本号即表示入口可用。之后本指南的安装、启动命令都在这个窗口中输入。不要把命令输入 Python 的 `>>>` 提示符，也不要把终端命令直接粘贴到 Notebook 的普通 Python 单元中。

官方图示：[Windows 安装说明](https://www.anaconda.com/docs/getting-started/anaconda/install/windows-gui-install)。

### 3.3 macOS 安装步骤

使用与芯片匹配的图形安装器，按提示完成安装。然后重新打开“终端”，执行：

```bash
conda --version
```

如果终端仍找不到 `conda`，先确认 Anaconda 的实际安装目录，再按安装器提示完成 shell 初始化。例如安装在 `/opt/anaconda3` 时，可执行：

```bash
/opt/anaconda3/bin/conda init zsh
```

随后关闭并重新打开终端。若实际安装位置不同，替换上述路径；不要照抄一个不存在的目录。

官方图示：[macOS 安装说明](https://www.anaconda.com/docs/getting-started/anaconda/install/mac-gui-install)。

### 3.4 Linux 安装步骤

下载匹配架构的官方 `.sh` 安装脚本后，在下载目录执行下面形式的命令，把文件名换成实际下载的名称：

```bash
bash Anaconda3-实际版本-Linux-x86_64.sh
```

阅读许可并按提示选择个人用户目录。根据提示允许 Conda 初始化终端，然后重新打开终端，运行 `conda --version`。本实验不需要用 `sudo pip` 安装 Python 包。

## 4. 创建本实验的独立环境

打开 Anaconda Prompt（Windows）或终端（macOS/Linux），依次执行：

```bash
conda create -n cv-cnn python=3.11 pip -y
conda activate cv-cnn
python --version
python -m pip --version
```

最后两条命令应显示 Python 3.11，以及属于 `cv-cnn` 环境的 pip 路径。提示符前通常会出现 `(cv-cnn)`。如果首次使用 Conda 时要求阅读或接受软件源服务条款，请按官方提示阅读并作出选择；未完成这一步可能导致创建环境失败。

以后每次新开终端，需要重新执行：

```bash
conda activate cv-cnn
```

常用查看命令：

```bash
conda env list
python -c "import sys; print(sys.executable)"
python -m pip list
```

本实验采用“**Conda 管环境，pip 装 PyTorch**”的方式。PyTorch 2.6 起不再发布官方 Conda 包，因此不要混用网上旧教程的 `conda install pytorch ...` 命令。

## 5. 安装 PyTorch：根据电脑选择一个分支

先确认已执行 `conda activate cv-cnn`，然后更新 pip：

```bash
python -m pip install --upgrade pip
```

以下使用固定的版本配对，便于复现安装环境。CPU 和 CUDA 分支选择其中一个安装。若需要换路线，建议另建环境，或在当前环境中卸载旧的 `torch`、`torchvision` 后按目标路线重装，不要把几条命令依次全部执行。

### 5.1 Windows / Linux：使用 CPU

使用 CPU 训练时执行：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

CPU 版无需 NVIDIA 驱动、CUDA 或 cuDNN。安装后 `torch.cuda.is_available()` 为 `False`、`torch.version.cuda` 为 `None` 都是正常现象。

### 5.2 Windows / Linux：使用 NVIDIA GPU

先在终端执行：

```bash
nvidia-smi
```

能看到 GPU 型号和驱动信息，说明系统已识别 NVIDIA 驱动。若命令不存在或报驱动错误，从 [NVIDIA 官方驱动入口](https://www.nvidia.com/Download/index.aspx) 安装或更新适合本机的驱动。驱动未配置时，可先使用 CPU。

对于兼容的 NVIDIA GPU 与驱动，本实验固定的 CUDA 12.4 构建安装命令为：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

几个需要分清的关系：

- **NVIDIA 驱动**由系统提供，PyTorch 要通过它使用 GPU。
- **PyTorch 的 CUDA 运行时依赖**随官方预编译包安装。运行本实验普通训练代码，通常不需要先另装完整 CUDA Toolkit，也无需手动复制 cuDNN。
- **CUDA Toolkit**主要用于编译自定义 CUDA 程序、扩展或自行编译 PyTorch，本实验不涉及。
- `nvidia-smi` 中的 `CUDA Version` 表示驱动支持的 CUDA 能力；`torch.version.cuda` 表示 PyTorch 构建所用的 CUDA 运行时版本。二者不必显示完全相同的数字。
- 系统装了 Toolkit，不会自动把 CPU 版 torch 变成 CUDA 版；是否可用要由实际运行验证。

**较新或较旧的 GPU：**固定的 2.6/cu124 组合并非适配所有显卡。例如较新的 RTX 50 系列/Blackwell 需要选择支持相应架构的更新构建。请根据 [PyTorch 当前官方安装选择器](https://pytorch.org/get-started/locally/) 选择系统、Pip、Python 和适用 CUDA 构建，并安装选择器给出的 `torch` / `torchvision` 版本配对。不要仅凭“CUDA 版本数字更大”判断兼容性。

### 5.3 Apple Silicon Mac：CPU 或 MPS

M 系列芯片 Mac 使用原生 arm64 Anaconda/Python，在已激活的环境中执行：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0
```

这套 macOS 包可使用 CPU，并在硬件、macOS 和 PyTorch 构建支持时使用 MPS。MPS 2.6 文档要求 macOS 12.3 或更新版本，实际以 `torch.backends.mps.is_available()` 和后面的运算验收为准。

Mac 的 GPU 不使用 CUDA。即使 MPS 可用，`torch.cuda.is_available()` 仍为 `False`，这并不表示安装失败。

### 5.4 Intel Mac：旧版 CPU 兼容路线

PyTorch 2.6 没有 Intel Mac x86_64 的官方 wheel。本实验为 Intel Mac 单列以下版本，仍使用 Python 3.11：

```bash
python -m pip install "numpy<2"
python -m pip install torch==2.2.2 torchvision==0.17.2
```

该版本配对用于 Intel Mac x86_64 环境，在 Notebook 中选择 CPU，不与其他平台的安装命令混用。后续通用依赖也保留 `numpy<2`，避免旧 PyTorch 与 NumPy 2 的兼容问题。

这些固定版本与配对可在 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/) 查到。硬件或操作系统不满足对应 wheel 条件时，需要改用支持该平台的版本或兼容环境。

## 6. 安装其他依赖

把资料包解压到一个便于找到的目录，保留里面的文件结构。进入含有 `requirements.txt` 的项目目录。下面的路径仅为示例，需要换成你自己电脑上的位置。

Windows Anaconda Prompt 示例：

```bat
cd /d "D:\cv-labs\实践课_卷积神经网络"
```

macOS / Linux 终端示例：

```bash
cd "$HOME/Downloads/实践课_卷积神经网络"
```

也可把路径换成完整路径，例如 `/Users/你的用户名/Downloads/实践课_卷积神经网络`。路径中有空格时，保留两侧引号。

在含有 `requirements.txt`、`cnn_common.py` 和 `卷积神经网络实践课.ipynb` 的目录中执行：

```bash
python -m pip install -r requirements.txt
```

本实验只需要 NumPy、Matplotlib、JupyterLab 和 ipykernel 等通用依赖，不必额外安装 OpenCV、TensorFlow 或 torchaudio。`requirements.txt` 故意不重复写 torch，便于各平台使用第 5 节选定的构建。

如果普通依赖下载很慢，可对这一次安装临时使用合适的 PyPI 镜像，例如：

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

第 5 节 PyTorch CPU/CUDA 命令应保留对应的官方 `--index-url`；普通 PyPI 镜像不能替代 CUDA wheel 软件源。本指南不要求修改全局 pip 或 Conda 镜像配置，也不要关闭 HTTPS 证书验证。

## 7. 验收安装：真正运行一次训练步骤

在项目目录、`cv-cnn` 环境中执行：

```bash
python check_env.py --device cpu
```

脚本会打印 Python 解释器路径、系统架构、torch/torchvision 版本、CUDA/MPS 状态，并用合成数据执行一次 CNN 前向计算、交叉熵、反向传播和参数更新。最后出现 `PASS` 表示本次设备上的基础训练步骤通过。此处的随机损失不代表 Fashion-MNIST 的实验结果。

使用 NVIDIA GPU 时执行：

```bash
python check_env.py --device cuda
```

使用 Mac MPS 时执行：

```bash
python check_env.py --device mps
```

也可让脚本按 `CUDA → MPS → CPU` 的顺序自动选择：

```bash
python check_env.py --device auto
```

使用自动选择时一定要阅读“实际验收设备”，不能因为脚本通过就把结果记作 GPU 实验。显式指定的 GPU 不可用时，脚本会报错，便于发现配置问题。

## 8. 注册内核并启动 JupyterLab

在已激活的 `cv-cnn` 环境中执行：

```bash
python -m ipykernel install --user --name cv-cnn --display-name "Python (cv-cnn)"
python -m jupyterlab
```

浏览器打开 JupyterLab 后：

1. 打开 `卷积神经网络实践课.ipynb`。
2. 点击右上角内核名称，或选择 **Kernel → Change Kernel**。
3. 选择 **Python (cv-cnn)**。
4. 运行第一个代码单元，检查打印出的 Python 路径与版本。

在 Notebook 中可以用下列代码再次确认：

```python
import sys
import torch
import torchvision

print(sys.executable)
print(torch.__version__, torchvision.__version__)
print("CUDA:", torch.cuda.is_available(), "runtime:", torch.version.cuda)
print("MPS:", torch.backends.mps.is_available())
```

`sys.executable` 应指向 `cv-cnn` 环境。终端激活环境后，已经打开的 Notebook 不会自动切换内核；切换或重装包后，要选择正确内核并重启它。

若浏览器没有自动打开，复制终端显示的本机 Jupyter 地址到浏览器。训练过程中保留启动 Jupyter 的终端；结束时在终端按 `Ctrl+C` 并按提示确认关闭服务。

## 9. 下载与读取 Fashion-MNIST

Fashion-MNIST 共 10 类，每张图为单通道 `28×28` 图像，官方包含 60,000 张训练图和 10,000 张测试图。数据来源为 [数据集作者官方仓库](https://github.com/zalandoresearch/fashion-mnist)。

Notebook 的默认配置为：

```python
DEVICE_CHOICE = "cpu"
TRAIN_SIZE = 10000
VAL_SIZE = 2000
EPOCHS = 3
BATCH_SIZE = 64
DOWNLOAD = True
```

保持首次运行的 `DOWNLOAD = True`。公共工具通过 torchvision 读取数据，使用数据集官方 S3 的 HTTPS 源，并以作者 GitHub 仓库为备用源，保存到项目目录下的 `data/FashionMNIST/raw/`。完整原始压缩下载量约 30 MB，下载后会解压；基础实验虽只用训练集的一个子集，首次仍需下载官方原始文件。

在已激活的环境和项目目录中执行以下命令，下载并检查训练与测试所需的四个原始文件：

```bash
python prepare_data.py
```

该命令仅下载与检查数据，不训练模型，也不计算测试准确率。Notebook 在同一缓存上使用 `DOWNLOAD = True` 时不会重复下载完整文件。

训练子集和验证子集来自官方训练集，按固定种子划分且不重叠；剩余官方训练图在本实验不用。官方测试集在所有配置确定后才做最终评估，不能用测试集选择学习率或最佳轮次。

预处理为 `ToTensor()` 和 `Normalize((0.5,), (0.5,))`：像素先由 0–255 变为 0–1，再映射到约 −1–1。一个 batch 的图像形状为 `N×1×28×28`，标签为形状 `N` 的 `torch.long` 整数。

### 下载不稳定或机房离线时

已有完整数据缓存时，将 `data/FashionMNIST/raw/` 目录复制到当前项目中。目录结构为：

```text
data/
└── FashionMNIST/
    └── raw/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

目录中可以同时有对应的 `.gz` 压缩文件。**要在完全离线时将 `DOWNLOAD = False`，应确认上述 4 个无 `.gz` 后缀的解压文件都已经存在。** 只有 `.gz` 时，先按标准 gzip 格式解压为相同名称；联网缓存过程中 torchvision 会自动完成这一步。不要把文件放成多一层 `data/data/FashionMNIST/raw/`。

数据准备前可检查当前工作目录：

```python
from pathlib import Path
print(Path.cwd())
print((Path("data") / "FashionMNIST" / "raw").resolve())
```

离线安装 Python 包需要与目标系统、Python 版本和处理器架构匹配的环境或 wheel 包。数据文件可以跨平台复制，Windows 的二进制 Python 包不能直接用于 Mac。

## 10. 如何切换 CPU、CUDA 和 MPS 训练

在 Notebook 配置单元修改 `DEVICE_CHOICE`，然后从配置单元起按顺序重跑。训练单元会重新初始化模型与优化器。

| 取值 | 使用条件 |
|---|---|
| `"cpu"` | 所有平台的基础路线，也是默认值 |
| `"cuda"` | 兼容的 NVIDIA GPU、驱动、CUDA 版 torch，且设备验收通过 |
| `"mps"` | 兼容的 Mac 与 PyTorch 构建，且 MPS 验收通过 |
| `"auto"` | 自动按 CUDA、MPS、CPU 的顺序选择，需记录实际设备 |

**模型、输入图像和标签需要位于同一设备。** 完整训练代码已处理设备切换。下面展示设备使用方式：

```python
import torch
from cnn_common import select_device
from cnn_model import SimpleCNN

device = select_device("cpu")  # 有可用 NVIDIA GPU 时可改为 "cuda"
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

logits = model(images)          # [N, 10]，不先做 Softmax
loss = criterion(logits, labels)
print("model:", next(model.parameters()).device)
print("images:", images.device, "labels:", labels.device)
```

使用交叉熵时，最后一层直接输出 logits。展示预测概率时才另外调用 `torch.softmax(logits, dim=1)`；预测类别可直接用 `logits.argmax(dim=1)`。

GPU 并不一定让这个小模型更快：数据较少、batch 较小时，数据搬运与启动开销可能占比较大。比较 CPU 与 GPU 耗时时，保持相同数据、种子、模型、batch 和轮数，并采用相同计时范围。CUDA 或 MPS 在计时边界要同步，公共 `fit` 工具已经处理相应同步。

Windows 与 Notebook 中的 DataLoader 默认使用 `num_workers=0`，优先保证跨平台可运行。不要一开始就增加进程数排查复杂问题。

### 模型保存后如何在其他设备加载

公共工具保存的是模型参数及基本实验信息。先建立相同结构，再映射到目标设备，例如：

```python
device = select_device("cpu")
model = SimpleCNN().to(device)
checkpoint = torch.load("outputs/baseline_best.pt",
                        map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

这样可以在 CPU 上读取原来 GPU 训练得到的参数。`eval()` 切换层的行为，但不会自动关闭梯度；评估时还需要 `torch.no_grad()` 或等效的推理上下文。

## 11. 运行训练与测试

完整代码包含模型、训练、验证、保存和测试过程，不需要填写代码。Notebook 和命令行脚本使用相同的网络结构与数据处理方式，选择一种运行方式即可。

### 11.1 在 Notebook 中运行

1. 打开 `卷积神经网络实践课.ipynb`，选择 `Python (cv-cnn)` 内核。
2. 在配置单元设置设备、样本数和训练轮数；默认 `DEVICE_CHOICE = "cpu"`、训练 10,000 张、验证 2,000 张、3 轮、batch 64。
3. 按顺序运行全部单元。代码会检查数据、构建模型、完成训练、绘制曲线并保存最佳模型。
4. 训练结束后，运行最终测试单元，加载最佳模型参数并输出测试准确率、混淆矩阵和误分类图像。

### 11.2 使用命令行脚本

在项目目录中依次执行：

```bash
conda activate cv-cnn
python check_env.py --device cpu
python prepare_data.py
python train.py --device cpu --epochs 3
python test.py --device cpu --checkpoint outputs/baseline_best.pt
```

使用 NVIDIA GPU 时，将 `--device cpu` 改为 `--device cuda`；兼容 Mac 使用 `--device mps`。训练和测试可以使用不同设备，加载参数时会进行设备映射。

先检查流程时，可使用小规模配置：

```bash
python train.py --device cpu --train-size 256 --val-size 128 --epochs 1 --run-name smoke
```

该运行会生成 `outputs/smoke_best.pt`，与默认的 `baseline` 结果分开保存。

CPU 计算较慢时，可减少样本数：

```bash
python train.py --device cpu --train-size 5000 --val-size 1000 --epochs 3
```

增加训练轮数时，例如：

```bash
python train.py --device cpu --epochs 10
```

每次运行 `train.py` 都从新初始化的模型开始训练，同名输出会被覆盖；需要保留多次结果时，通过 `--run-name` 设置不同名称。上述命令中默认学习率为 `1e-3`，batch size 为 64；可以使用 `--lr` 和 `--batch-size` 调整。完整参数可通过 `python train.py --help` 和 `python test.py --help` 查看。

### 11.3 查看输出

默认运行生成：

| 文件 | 内容 |
|---|---|
| `outputs/baseline_best.pt` | 验证准确率最高时的模型参数 |
| `outputs/baseline_history.json` | 每轮训练/验证损失、准确率和耗时 |
| `outputs/baseline_config.json` | 设备、样本规模、轮数、学习率等运行设置 |
| `outputs/baseline_curves.png` | 训练/验证损失和准确率曲线 |
| `outputs/baseline_test_metrics.json` | 加载最佳模型后的测试损失和准确率 |
| `outputs/baseline_confusion.png` | 测试集混淆矩阵 |
| `outputs/baseline_mistakes.png` | 测试集误分类图像 |

训练使用训练集更新参数，使用验证集选择最佳轮次；`test.py` 仅加载模型进行测试，不更新参数。增加轮数或调整学习率时，根据验证曲线判断效果，配置确定后再进行最终测试。

损失总体下降、验证准确率提高、最佳模型成功保存并能独立加载测试，表明训练与测试流程已正常运行。训练时间与准确率随硬件、软件版本、数据规模和随机种子变化。

## 12. 常见问题

| 现象 | 优先检查与处理 |
|---|---|
| `conda` 不是内部命令 / command not found | Windows 使用 Anaconda Prompt；macOS/Linux 确认实际安装位置并完成对应 shell 初始化 |
| `ModuleNotFoundError: torch` 或其他包 | 在终端激活 `cv-cnn`，检查 `python -m pip --version`；Notebook 选择 `Python (cv-cnn)` 并重启内核 |
| `No matching distribution found` | 检查 Python 是否 3.11、系统和处理器架构是否受支持、版本是否配对；Intel Mac 使用 2.2.2/0.17.2 路线 |
| `torchvision` 导入报错或算子不存在 | 常见于 torch/torchvision 配对或构建混用；按第 5 节同一分支重新安装匹配版本并重启内核 |
| `Torch not compiled with CUDA enabled` | 当前通常是 CPU 版 torch；需要兼容 NVIDIA 硬件、驱动和 CUDA 版包，或在配置中使用 CPU |
| `torch.cuda.is_available()` 是 False | CPU 版、非 NVIDIA 硬件、驱动问题、GPU 不可见或内核选错均可能导致；依次查解释器、torch版本/runtime、`nvidia-smi`、实际验收 |
| `no kernel image is available` 等 GPU 架构错误 | 当前 torch/CUDA 构建可能不支持显卡架构；按官方选择器选兼容新旧版本，先改 CPU 完成任务 |
| `Expected all tensors to be on the same device` | 检查模型、图像、标签是否都 `.to(device)`；加载模型使用 `map_location` |
| CUDA out of memory | 先将 `BATCH_SIZE` 降为 32 或 16，关闭其他占用显存的任务；必要时重启内核后从头运行或使用 CPU |
| `mat1 and mat2 shapes cannot be multiplied` | 检查卷积/池化尺寸；本实验两次池化后应为 `32×7×7`，展平是 1568 维 |
| `expected scalar type Long` 等标签类型错误 | 分类标签应为整数类别编号，类型是 `torch.long`，不要把标签转为浮点图像 |
| 验证集准确率约为随机猜测或损失异常 | 检查标签、归一化、训练步骤、梯度更新及最后一层；交叉熵前不重复 Softmax，学习率过大可导致不稳定 |
| 旧版 PyTorch 出现 NumPy ABI / `_ARRAY_API` 错误 | 在当前环境执行 `python -m pip install "numpy<2"`，然后重启 Notebook 内核 |
| 数据下载失败 / 找不到数据集 | 检查网络、当前目录和 `data/FashionMNIST/raw`；使用第 9 节的完整数据缓存，离线时确认解压文件齐全 |
| `No module named cnn_common` | 从项目目录启动 Jupyter，保留 `cnn_common.py`、`cnn_model.py` 与 Notebook 的相对位置 |
| 修改 `.py` 文件后 Notebook 仍使用旧代码 | 重启内核并按顺序运行；Python 导入的模块可能仍缓存旧内容 |

定位问题时，记录完整错误信息、`sys.executable`、Python/torch/torchvision 版本、操作系统和设备信息。

## 13. 运行结果检查

- 环境检查输出 `PASS`，实际设备与配置一致。
- 数据能够读取，训练集与验证集不重叠。
- 训练日志包含每轮损失与准确率，训练曲线和最佳模型参数文件已保存。
- 最佳模型可以通过 `test.py` 或 Notebook 测试单元重新加载。
- 测试输出包含损失、准确率、混淆矩阵和误分类图像。
- 环境版本、设备、样本规模、学习率、batch size 和轮数已记录。

## 14. 官方参考资料

- [Anaconda 安装入口](https://www.anaconda.com/docs/getting-started/anaconda/install)
- [Anaconda Windows 图形安装](https://www.anaconda.com/docs/getting-started/anaconda/install/windows-gui-install)
- [Anaconda macOS 图形安装](https://www.anaconda.com/docs/getting-started/anaconda/install/mac-gui-install)
- [Anaconda 历史安装器](https://repo.anaconda.com/archive/)
- [PyTorch 当前安装选择器](https://pytorch.org/get-started/locally/)
- [PyTorch 历史版本与配对命令](https://pytorch.org/get-started/previous-versions/)
- [PyTorch 2.6 发布说明](https://pytorch.org/blog/pytorch2-6/)
- [PyTorch 2.6 MPS 文档](https://docs.pytorch.org/docs/2.6/notes/mps.html)
- [PyTorch 基础训练教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Fashion-MNIST 官方仓库](https://github.com/zalandoresearch/fashion-mnist)

软件版本与硬件支持会变化，更新环境时重新核对官方安装选择器与依赖配对。
