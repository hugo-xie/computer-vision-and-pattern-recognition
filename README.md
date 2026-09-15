# Computer Vision and Pattern Recognition

计算机视觉课程资料与实践代码。

## 内容

- `Practice/basic-python-cv/01_Python图像处理实践课.ipynb`：主版本，涵盖 OpenCV、Pillow、SciPy、NumPy 和 Matplotlib 的基础图像处理实践。
- `Practice/basic-python-cv/01_Python3.7图像处理实践课.ipynb`：Python 3.7 兼容版本。
- `Practice/basic-python-cv/images/`：Notebook 使用的示例图像。

## 运行环境

建议使用 Python 3.11 或兼容的 Conda 环境，并安装以下依赖：

```bash
python -m pip install numpy pillow scipy matplotlib opencv-python jupyter
```

在 Jupyter 中打开任意 Notebook 后，选择 `Kernel -> Restart & Run All` 运行全部单元。

## 说明

Notebook 使用相对路径读取同目录下的 `images/` 素材；素材缺失时会生成备用示例图像。上传版本已清除运行输出，避免保存本机路径、错误堆栈和嵌入式大图片。后续课程内容可按主题继续添加到 `Practice/` 下，并通过 Git 提交和推送。
