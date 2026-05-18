# VisionCNN

基于卷积神经网络的猫狗图像二分类实验工程，包含基础 CNN、数据增强 CNN 和 ResNet50 迁移学习的快速验证脚本。

## 数据集

本仓库不提交训练集、测试集、模型权重和训练输出文件。请从 Kaggle 下载原始数据：

https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

下载并解压后，推荐放置为：

```text
doc/dogs-vs-cats-redux-kernels-edition/train/
doc/dogs-vs-cats-redux-kernels-edition/test/
```

训练脚本使用带标签的 `train/` 目录；Kaggle 的 `test/` 目录没有公开标签，不参与本地训练评估。

## 环境安装

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果本机 Python 证书导致 pip SSL 校验失败，可临时使用：

```bash
.venv/bin/python -m pip install \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host download.pytorch.org \
  -r requirements.txt
```

## 快速验证

```bash
.venv/bin/python experiments/cats_dogs_cnn.py \
  --data_dir doc/dogs-vs-cats-redux-kernels-edition/train \
  --output_dir runs/quick_verify \
  --mode main \
  --limit_per_class 100 \
  --epochs 2 \
  --batch_size 16 \
  --num_workers 0 \
  --norm imagenet \
  --no-gradcam
```

快速验证只用于确认数据读取、划分、训练、评估和结果保存流程可运行，不代表正式模型效果。

## 正式训练

完整实验可去掉 `--limit_per_class`，并恢复论文配置：

```bash
.venv/bin/python experiments/cats_dogs_cnn.py \
  --data_dir doc/dogs-vs-cats-redux-kernels-edition/train \
  --output_dir runs/full \
  --mode main \
  --epochs 30 \
  --batch_size 32 \
  --num_workers 0 \
  --norm dataset
```

训练输出默认写入 `runs/`，该目录已被 Git 忽略。
