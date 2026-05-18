\# -\*- coding: utf-8 -\*-

"""

基于卷积神经网络的图像分类应用研究

——从基础 CNN 到迁移学习的递进实践

说明：

1\. 本代码按论文主线编写：

猫狗二分类数据集 -> 8:1:1 划分 -> 数据清洗/尺寸统一/标准化/数据增强

\-> 基础 CNN -> 数据增强 CNN -> ResNet50 迁移学习 CNN

\-> Accuracy / Precision / Recall / F1 / 混淆矩阵 / 训练曲线 / Grad-CAM。

2\. 默认训练配置与论文一致：

Adam，lr=0.001，batch_size=32，max_epochs=30，早停 patience=10，

ReduceLROnPlateau 学习率衰减 factor=0.5。

3\. 数据集支持两种常见结构：

A. data/dogs_vs_cats/cat/\*.jpg, data/dogs_vs_cats/dog/\*.jpg

B. data/dogs_vs_cats/cat.0.jpg, data/dogs_vs_cats/dog.0.jpg

4\. 正式复现实验请使用完整 Kaggle Dogs vs Cats 数据集。

"""

import argparse

import csv

import os

import random

import shutil

import time

import warnings

from dataclasses import dataclass

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PIL import Image, ImageFile

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset

try:

from torchvision import models, transforms

except Exception as exc:

raise ImportError(

"本代码需要 torchvision。请先安装：pip install torchvision"

) from exc

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore", category=UserWarning)

\# ============================================================

\# 1. 基础设置

\# ============================================================

CLASS_NAMES = \["Cat", "Dog"\]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_MEAN = \[0.485, 0.456, 0.406\] # ImageNet 均值

DEFAULT_STD = \[0.229, 0.224, 0.225\] # ImageNet 标准差

def set_seed(seed: int = 42) -> None:

"""固定随机种子，增强实验可复现性。"""

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed_all(seed)

\# 下面两项会提高可复现性，但可能降低速度

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:

return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: Path) -> None:

path.mkdir(parents=True, exist_ok=True)

def format_seconds(seconds: float) -> str:

seconds = int(seconds)

h = seconds // 3600

m = (seconds % 3600) // 60

s = seconds % 60

if h > 0:

return f"{h}h{m:02d}m{s:02d}s"

if m > 0:

return f"{m}m{s:02d}s"

return f"{s}s"

\# ============================================================

\# 2. 数据集扫描、清洗与划分

\# ============================================================

def scan_cat_dog_images(data_dir: Path, limit_per_class: Optional\[int\] = None) -> Tuple\[List\[Path\], List\[int\]\]:

"""

扫描猫狗图像数据。

支持结构：

1) root/cat/\*.jpg, root/dog/\*.jpg

2) root/train/cat/\*.jpg, root/train/dog/\*.jpg

3) root/cat.0.jpg, root/dog.0.jpg

"""

data_dir = Path(data_dir)

if not data_dir.exists():

raise FileNotFoundError(

f"数据集路径不存在：{data_dir}\\n"

"请将 Kaggle Dogs vs Cats 数据集解压到该目录，或通过 --data_dir 指定正确路径。"

)

candidates: List\[Tuple\[Path, int\]\] = \[\]

\# 结构一：cat / dog 子文件夹

possible_roots = \[data_dir, data_dir / "train"\]

for root in possible_roots:

cat_dir = root / "cat"

dog_dir = root / "dog"

if cat_dir.exists():

for p in sorted(cat_dir.rglob("\*")):

if p.suffix.lower() in IMAGE_EXTS:

candidates.append((p, 0))

if dog_dir.exists():

for p in sorted(dog_dir.rglob("\*")):

if p.suffix.lower() in IMAGE_EXTS:

candidates.append((p, 1))

\# 结构二：扁平文件名 cat.0.jpg / dog.0.jpg

if len(candidates) == 0:

for p in sorted(data_dir.rglob("\*")):

if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:

continue

name = p.name.lower()

if name.startswith("cat"):

candidates.append((p, 0))

elif name.startswith("dog"):

candidates.append((p, 1))

if len(candidates) == 0:

raise RuntimeError(

f"在 {data_dir} 下没有找到猫狗图像。\\n"

"请检查数据格式：应包含 cat/dog 子文件夹，或文件名以 cat / dog 开头。"

)

\# 限制每类数量，用于快速测试

if limit_per_class is not None and limit_per_class > 0:

selected: List\[Tuple\[Path, int\]\] = \[\]

for label in \[0, 1\]:

items = \[(p, y) for p, y in candidates if y == label\]

selected.extend(items\[:limit_per_class\])

candidates = selected

paths = \[p for p, _ in candidates\]

labels = \[y for \_, y in candidates\]

cat_count = sum(1 for y in labels if y == 0)

dog_count = sum(1 for y in labels if y == 1)

print(f"扫描完成：共 {len(paths)} 张图像，其中 Cat={cat_count}, Dog={dog_count}")

if cat_count == 0 or dog_count == 0:

raise RuntimeError("猫或狗其中一类数量为 0，无法进行二分类实验。")

return paths, labels

def verify_and_filter_images(paths: Sequence\[Path\], labels: Sequence\[int\], report_path: Path) -> Tuple\[List\[Path\], List\[int\]\]:

"""

剔除损坏或无法读取的图像。

这是论文 4.1.1 “数据清洗”的代码实现。

"""

valid_paths: List\[Path\] = \[\]

valid_labels: List\[int\] = \[\]

bad_rows: List\[List\[str\]\] = \[\]

for p, y in zip(paths, labels):

try:

with Image.open(p) as img:

img.verify()

\# 再次打开，确保可正常 convert

with Image.open(p) as img:

_ = img.convert("RGB")

valid_paths.append(p)

valid_labels.append(int(y))

except Exception as exc:

bad_rows.append(\[str(p), str(exc)\])

with open(report_path, "w", newline="", encoding="utf-8-sig") as f:

writer = csv.writer(f)

writer.writerow(\["bad_image_path", "error"\])

writer.writerows(bad_rows)

print(f"数据清洗完成：有效图像 {len(valid_paths)} 张，损坏图像 {len(bad_rows)} 张")

print(f"损坏图像报告：{report_path}")

return valid_paths, valid_labels

def stratified_split_indices(

labels: Sequence\[int\],

train_ratio: float = 0.8,

val_ratio: float = 0.1,

seed: int = 42,

) -> Tuple\[List\[int\], List\[int\], List\[int\]\]:

"""

按类别分层划分训练集、验证集和测试集，保证猫狗比例尽量一致。

默认比例为 8:1:1。

"""

rng = np.random.default_rng(seed)

labels_arr = np.array(labels)

train_idx: List\[int\] = \[\]

val_idx: List\[int\] = \[\]

test_idx: List\[int\] = \[\]

for c in sorted(np.unique(labels_arr)):

idx = np.where(labels_arr == c)\[0\]

rng.shuffle(idx)

n = len(idx)

n_train = int(n \* train_ratio)

n_val = int(n \* val_ratio)

train_idx.extend(idx\[:n_train\].tolist())

val_idx.extend(idx\[n_train:n_train + n_val\].tolist())

test_idx.extend(idx\[n_train + n_val:\].tolist())

rng.shuffle(train_idx)

rng.shuffle(val_idx)

rng.shuffle(test_idx)

return train_idx, val_idx, test_idx

def save_split_csv(

paths: Sequence\[Path\],

labels: Sequence\[int\],

train_idx: Sequence\[int\],

val_idx: Sequence\[int\],

test_idx: Sequence\[int\],

save_path: Path,

) -> None:

idx_to_split = {}

for i in train_idx:

idx_to_split\[i\] = "train"

for i in val_idx:

idx_to_split\[i\] = "val"

for i in test_idx:

idx_to_split\[i\] = "test"

with open(save_path, "w", newline="", encoding="utf-8-sig") as f:

writer = csv.writer(f)

writer.writerow(\["split", "path", "label", "class_name"\])

for i, p in enumerate(paths):

writer.writerow(\[idx_to_split\[i\], str(p), labels\[i\], CLASS_NAMES\[labels\[i\]\]\])

class CatsDogsDataset(Dataset):

"""猫狗二分类数据集。"""

def \__init_\_(

self,

image_paths: Sequence\[Path\],

labels: Sequence\[int\],

transform=None,

return_path: bool = False,

) -> None:

self.image_paths = list(image_paths)

self.labels = list(labels)

self.transform = transform

self.return_path = return_path

def \__len_\_(self) -> int:

return len(self.image_paths)

def \__getitem_\_(self, idx: int):

path = self.image_paths\[idx\]

label = int(self.labels\[idx\])

image = Image.open(path).convert("RGB")

if self.transform is not None:

image = self.transform(image)

if self.return_path:

return image, label, str(path)

return image, label

\# ============================================================

\# 3. 数据预处理与增强

\# ============================================================

def build_base_transform(image_size: int, mean: Sequence\[float\], std: Sequence\[float\]):

"""验证/测试集变换：尺寸统一 + ToTensor + 标准化。"""

return transforms.Compose(\[

transforms.Resize((image_size, image_size)),

transforms.ToTensor(), # 像素值从 \[0,255\] 映射到 \[0,1\]

transforms.Normalize(mean=mean, std=std),

\])

def build_aug_transform(image_size: int, mean: Sequence\[float\], std: Sequence\[float\]):

"""

训练集数据增强：

\- 随机水平翻转 p=0.5

\- 随机旋转 ±20°

\- 随机平移：宽高 20%

\- 随机缩放：80%—120%

\- 亮度/对比度 ±20%

"""

return transforms.Compose(\[

transforms.Resize((image_size, image_size)),

transforms.RandomHorizontalFlip(p=0.5),

transforms.RandomRotation(degrees=20),

transforms.RandomAffine(

degrees=0,

translate=(0.2, 0.2),

scale=(0.8, 1.2),

),

transforms.ColorJitter(brightness=0.2, contrast=0.2),

transforms.ToTensor(),

transforms.Normalize(mean=mean, std=std),

\])

def compute_dataset_mean_std(

image_paths: Sequence\[Path\],

labels: Sequence\[int\],

indices: Sequence\[int\],

image_size: int,

batch_size: int,

num_workers: int,

) -> Tuple\[List\[float\], List\[float\]\]:

"""

使用训练集统计量计算 RGB 三通道均值和标准差。

为避免增强影响统计，只使用 Resize + ToTensor。

"""

transform = transforms.Compose(\[

transforms.Resize((image_size, image_size)),

transforms.ToTensor(),

\])

sub_paths = \[image_paths\[i\] for i in indices\]

sub_labels = \[labels\[i\] for i in indices\]

dataset = CatsDogsDataset(sub_paths, sub_labels, transform=transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

channel_sum = torch.zeros(3)

channel_sq_sum = torch.zeros(3)

num_pixels = 0

print("正在计算训练集均值和标准差...")

for images, _ in loader:

\# images: \[B, 3, H, W\]

b, c, h, w = images.shape

channel_sum += images.sum(dim=\[0, 2, 3\])

channel_sq_sum += (images \*\* 2).sum(dim=\[0, 2, 3\])

num_pixels += b \* h \* w

mean = channel_sum / num_pixels

std = torch.sqrt(channel_sq_sum / num_pixels - mean \*\* 2)

mean_list = mean.tolist()

std_list = std.tolist()

print(f"训练集 mean = {\[round(x, 4) for x in mean_list\]}")

print(f"训练集 std = {\[round(x, 4) for x in std_list\]}")

return mean_list, std_list

@dataclass

class DataBundle:

train_loader: DataLoader

val_loader: DataLoader

test_loader: DataLoader

train_dataset: Dataset

val_dataset: Dataset

test_dataset: Dataset

mean: List\[float\]

std: List\[float\]

def create_dataloaders(

image_paths: Sequence\[Path\],

labels: Sequence\[int\],

train_idx: Sequence\[int\],

val_idx: Sequence\[int\],

test_idx: Sequence\[int\],

image_size: int,

batch_size: int,

num_workers: int,

mean: Sequence\[float\],

std: Sequence\[float\],

use_augmentation: bool,

) -> DataBundle:

"""

创建 DataLoader。

注意：训练集、验证集、测试集使用彼此独立的 Dataset，避免 transform 混用。

"""

train_paths = \[image_paths\[i\] for i in train_idx\]

train_labels = \[labels\[i\] for i in train_idx\]

val_paths = \[image_paths\[i\] for i in val_idx\]

val_labels = \[labels\[i\] for i in val_idx\]

test_paths = \[image_paths\[i\] for i in test_idx\]

test_labels = \[labels\[i\] for i in test_idx\]

train_transform = (

build_aug_transform(image_size, mean, std)

if use_augmentation

else build_base_transform(image_size, mean, std)

)

eval_transform = build_base_transform(image_size, mean, std)

train_dataset = CatsDogsDataset(train_paths, train_labels, transform=train_transform, return_path=True)

val_dataset = CatsDogsDataset(val_paths, val_labels, transform=eval_transform, return_path=True)

test_dataset = CatsDogsDataset(test_paths, test_labels, transform=eval_transform, return_path=True)

pin_memory = torch.cuda.is_available()

train_loader = DataLoader(

train_dataset,

batch_size=batch_size,

shuffle=True,

num_workers=num_workers,

pin_memory=pin_memory,

)

val_loader = DataLoader(

val_dataset,

batch_size=batch_size,

shuffle=False,

num_workers=num_workers,

pin_memory=pin_memory,

)

test_loader = DataLoader(

test_dataset,

batch_size=batch_size,

shuffle=False,

num_workers=num_workers,

pin_memory=pin_memory,

)

return DataBundle(

train_loader=train_loader,

val_loader=val_loader,

test_loader=test_loader,

train_dataset=train_dataset,

val_dataset=val_dataset,

test_dataset=test_dataset,

mean=list(mean),

std=list(std),

)

\# ============================================================

\# 4. 模型定义

\# ============================================================

class BasicCNN(nn.Module):

"""

基础 CNN：

四个卷积块，每个卷积块包含 Conv2d + BatchNorm + ReLU + MaxPool。

输入 224x224，四次池化后为 14x14。

fc1 采用 300 个神经元，使总参数量约 1500 万，贴合论文表述。

"""

def \__init_\_(self, num_classes: int = 2, dropout: float = 0.0) -> None:

super().\__init_\_()

self.conv_block1 = nn.Sequential(

nn.Conv2d(3, 32, kernel_size=3, padding=1),

nn.BatchNorm2d(32),

nn.ReLU(inplace=True),

nn.MaxPool2d(kernel_size=2, stride=2),

)

self.conv_block2 = nn.Sequential(

nn.Conv2d(32, 64, kernel_size=3, padding=1),

nn.BatchNorm2d(64),

nn.ReLU(inplace=True),

nn.MaxPool2d(kernel_size=2, stride=2),

)

self.conv_block3 = nn.Sequential(

nn.Conv2d(64, 128, kernel_size=3, padding=1),

nn.BatchNorm2d(128),

nn.ReLU(inplace=True),

nn.MaxPool2d(kernel_size=2, stride=2),

)

self.conv_block4 = nn.Sequential(

nn.Conv2d(128, 256, kernel_size=3, padding=1),

nn.BatchNorm2d(256),

nn.ReLU(inplace=True),

nn.MaxPool2d(kernel_size=2, stride=2),

)

fc_input_dim = 256 \* 14 \* 14

self.classifier = nn.Sequential(

nn.Dropout(dropout),

nn.Linear(fc_input_dim, 300),

nn.ReLU(inplace=True),

nn.Dropout(dropout),

nn.Linear(300, 128),

nn.ReLU(inplace=True),

nn.Linear(128, num_classes),

)

def forward(self, x: torch.Tensor) -> torch.Tensor:

x = self.conv_block1(x)

x = self.conv_block2(x)

x = self.conv_block3(x)

x = self.conv_block4(x)

x = torch.flatten(x, start_dim=1)

x = self.classifier(x)

return x

@property

def gradcam_target_layer(self):

return self.conv_block4\[0\]

class LeNet5Color(nn.Module):

"""

LeNet-5 变体：

论文中 LeNet-5 主要作为理论基础。本实现用于可选对比实验。

为适配彩色猫狗图像，这里输入为 3 通道，并统一缩放到 32x32。

"""

def \__init_\_(self, num_classes: int = 2) -> None:

super().\__init_\_()

self.conv1 = nn.Conv2d(3, 6, kernel_size=5) # 32 -> 28

self.pool1 = nn.MaxPool2d(2, 2) # 28 -> 14

self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 14 -> 10

self.pool2 = nn.MaxPool2d(2, 2) # 10 -> 5

self.fc1 = nn.Linear(16 \* 5 \* 5, 120)

self.fc2 = nn.Linear(120, 84)

self.fc3 = nn.Linear(84, num_classes)

self.relu = nn.ReLU(inplace=True)

def forward(self, x: torch.Tensor) -> torch.Tensor:

x = self.pool1(self.relu(self.conv1(x)))

x = self.pool2(self.relu(self.conv2(x)))

x = torch.flatten(x, start_dim=1)

x = self.relu(self.fc1(x))

x = self.relu(self.fc2(x))

return self.fc3(x)

class TransferResNet50(nn.Module):

"""

基于 ResNet50 的迁移学习模型。

默认使用 ImageNet 预训练权重，并冻结骨干网络，仅训练自定义分类头。

"""

def \__init_\_(

self,

num_classes: int = 2,

pretrained: bool = True,

freeze_backbone: bool = True,

) -> None:

super().\__init_\_()

if pretrained:

try:

weights = models.ResNet50_Weights.IMAGENET1K_V1

self.backbone = models.resnet50(weights=weights)

print("已加载 ResNet50 ImageNet 预训练权重。")

except Exception as exc:

print("警告：ResNet50 预训练权重加载失败，将使用随机初始化权重。")

print("原因：", str(exc))

self.backbone = models.resnet50(weights=None)

else:

self.backbone = models.resnet50(weights=None)

if freeze_backbone:

for param in self.backbone.parameters():

param.requires_grad = False

in_features = self.backbone.fc.in_features

self.backbone.fc = nn.Sequential(

nn.Linear(in_features, 512),

nn.ReLU(inplace=True),

nn.Dropout(0.5),

nn.Linear(512, 256),

nn.ReLU(inplace=True),

nn.Dropout(0.3),

nn.Linear(256, num_classes),

)

\# 分类头必须可训练

for param in self.backbone.fc.parameters():

param.requires_grad = True

def forward(self, x: torch.Tensor) -> torch.Tensor:

return self.backbone(x)

def unfreeze_deep_layers(self) -> None:

"""可选微调：解冻 layer4 和分类头。"""

for param in self.backbone.layer4.parameters():

param.requires_grad = True

for param in self.backbone.fc.parameters():

param.requires_grad = True

@property

def gradcam_target_layer(self):

return self.backbone.layer4\[-1\]

\# ============================================================

\# 5. 指标计算

\# ============================================================

def count_trainable_params(model: nn.Module) -> int:

return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model: nn.Module) -> int:

return sum(p.numel() for p in model.parameters())

def compute_confusion_matrix(y_true: Sequence\[int\], y_pred: Sequence\[int\], num_classes: int = 2) -> np.ndarray:

cm = np.zeros((num_classes, num_classes), dtype=np.int64)

for t, p in zip(y_true, y_pred):

cm\[int(t), int(p)\] += 1

return cm

def compute_metrics(y_true: Sequence\[int\], y_pred: Sequence\[int\], num_classes: int = 2) -> Dict\[str, object\]:

y_true_arr = np.array(y_true)

y_pred_arr = np.array(y_pred)

cm = compute_confusion_matrix(y_true_arr, y_pred_arr, num_classes=num_classes)

accuracy = float(np.mean(y_true_arr == y_pred_arr))

precisions = \[\]

recalls = \[\]

f1s = \[\]

for c in range(num_classes):

tp = cm\[c, c\]

fp = cm\[:, c\].sum() - tp

fn = cm\[c, :\].sum() - tp

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

f1 = (

2 \* precision \* recall / (precision + recall)

if (precision + recall) > 0

else 0.0

)

precisions.append(float(precision))

recalls.append(float(recall))

f1s.append(float(f1))

macro_precision = float(np.mean(precisions))

macro_recall = float(np.mean(recalls))

macro_f1 = float(np.mean(f1s))

return {

"accuracy": accuracy,

"precision": macro_precision,

"recall": macro_recall,

"f1": macro_f1,

"confusion_matrix": cm,

"class_precision": precisions,

"class_recall": recalls,

"class_f1": f1s,

}

\# ============================================================

\# 6. 训练与评估

\# ============================================================

def unpack_batch(batch):

"""兼容 Dataset 返回 (x,y) 或 (x,y,path)。"""

if len(batch) == 3:

images, labels, paths = batch

return images, labels, paths

images, labels = batch

return images, labels, None

def evaluate_model(

model: nn.Module,

loader: DataLoader,

device: torch.device,

criterion: Optional\[nn.Module\] = None,

) -> Dict\[str, object\]:

model.eval()

all_preds: List\[int\] = \[\]

all_labels: List\[int\] = \[\]

total_loss = 0.0

n_batches = 0

with torch.no_grad():

for batch in loader:

images, labels, _ = unpack_batch(batch)

images = images.to(device)

labels = labels.to(device)

outputs = model(images)

if criterion is not None:

loss = criterion(outputs, labels)

total_loss += float(loss.item())

n_batches += 1

preds = torch.argmax(outputs, dim=1)

all_preds.extend(preds.cpu().numpy().tolist())

all_labels.extend(labels.cpu().numpy().tolist())

metrics = compute_metrics(all_labels, all_preds, num_classes=2)

metrics\["loss"\] = total_loss / max(n_batches, 1)

metrics\["predictions"\] = all_preds

metrics\["labels"\] = all_labels

return metrics

def train_model(

model: nn.Module,

train_loader: DataLoader,

val_loader: DataLoader,

device: torch.device,

output_dir: Path,

model_name: str,

epochs: int = 30,

lr: float = 0.001,

patience: int = 10,

weight_decay: float = 0.0,

) -> Tuple\[nn.Module, Dict\[str, List\[float\]\], Dict\[str, object\]\]:

"""

训练模型：

\- CrossEntropyLoss

\- Adam

\- ReduceLROnPlateau：验证损失停滞时学习率乘以 0.5

\- Early Stopping：验证准确率连续 patience 轮不提升则停止

"""

ensure_dir(output_dir)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

filter(lambda p: p.requires_grad, model.parameters()),

lr=lr,

weight_decay=weight_decay,

)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

optimizer,

mode="min",

factor=0.5,

patience=5,

)

history = {

"train_loss": \[\],

"train_acc": \[\],

"val_loss": \[\],

"val_acc": \[\],

"lr": \[\],

}

best_val_acc = -1.0

best_epoch = 0

patience_counter = 0

best_model_path = output_dir / f"best_{model_name}.pth"

print("=" \* 70)

print(f"开始训练：{model_name}")

print(f"总参数量：{count_total_params(model):,}")

print(f"可训练参数：{count_trainable_params(model):,}")

print(f"设备：{device}")

print("=" \* 70)

start_time = time.time()

for epoch in range(1, epochs + 1):

model.train()

train_loss_sum = 0.0

train_correct = 0

train_total = 0

for batch in train_loader:

images, labels, _ = unpack_batch(batch)

images = images.to(device)

labels = labels.to(device)

optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()

train_loss_sum += float(loss.item())

preds = torch.argmax(outputs, dim=1)

train_correct += int((preds == labels).sum().item())

train_total += int(labels.size(0))

train_loss = train_loss_sum / max(len(train_loader), 1)

train_acc = train_correct / max(train_total, 1)

val_metrics = evaluate_model(model, val_loader, device, criterion=criterion)

val_loss = float(val_metrics\["loss"\])

val_acc = float(val_metrics\["accuracy"\])

scheduler.step(val_loss)

current_lr = float(optimizer.param_groups\[0\]\["lr"\])

history\["train_loss"\].append(train_loss)

history\["train_acc"\].append(train_acc)

history\["val_loss"\].append(val_loss)

history\["val_acc"\].append(val_acc)

history\["lr"\].append(current_lr)

print(

f"Epoch \[{epoch:02d}/{epochs}\] "

f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "

f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}"

)

if val_acc > best_val_acc:

best_val_acc = val_acc

best_epoch = epoch

patience_counter = 0

torch.save(model.state_dict(), best_model_path)

print(f" -> 保存最佳模型：{best_model_path.name}，Val Acc={best_val_acc:.4f}")

else:

patience_counter += 1

if patience_counter >= patience:

print(f"早停触发：验证准确率连续 {patience} 轮未提升。")

break

train_time = time.time() - start_time

\# 加载最佳模型

if best_model_path.exists():

model.load_state_dict(torch.load(best_model_path, map_location=device))

best_info = {

"best_val_acc": best_val_acc,

"best_epoch": best_epoch,

"train_time_sec": train_time,

"train_time": format_seconds(train_time),

"best_model_path": str(best_model_path),

}

print(f"训练完成：{model_name}")

print(f"最佳验证准确率：{best_val_acc:.4f}，出现在 Epoch {best_epoch}")

print(f"训练耗时：{format_seconds(train_time)}")

return model, history, best_info

\# ============================================================

\# 7. 绘图与结果保存

\# ============================================================

def inverse_normalize_tensor(tensor: torch.Tensor, mean: Sequence\[float\], std: Sequence\[float\]) -> np.ndarray:

"""

将标准化后的 Tensor \[3,H,W\] 反标准化为 numpy 图像 \[H,W,3\]，范围 \[0,1\]。

"""

x = tensor.detach().cpu().clone()

for c in range(3):

x\[c\] = x\[c\] \* std\[c\] + mean\[c\]

x = torch.clamp(x, 0.0, 1.0)

return x.permute(1, 2, 0).numpy()

def plot_training_history(history: Dict\[str, List\[float\]\], save_path: Path, title: str) -> None:

epochs = range(1, len(history\["train_loss"\]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes\[0\].plot(epochs, history\["train_loss"\], marker="o", label="Train Loss")

axes\[0\].plot(epochs, history\["val_loss"\], marker="s", label="Val Loss")

axes\[0\].set_xlabel("Epoch")

axes\[0\].set_ylabel("Loss")

axes\[0\].set_title(f"{title} Loss")

axes\[0\].grid(True, alpha=0.3)

axes\[0\].legend()

axes\[1\].plot(epochs, history\["train_acc"\], marker="o", label="Train Acc")

axes\[1\].plot(epochs, history\["val_acc"\], marker="s", label="Val Acc")

axes\[1\].set_xlabel("Epoch")

axes\[1\].set_ylabel("Accuracy")

axes\[1\].set_title(f"{title} Accuracy")

axes\[1\].grid(True, alpha=0.3)

axes\[1\].legend()

plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.close()

def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str) -> None:

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, interpolation="nearest")

plt.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(CLASS_NAMES)))

ax.set_yticks(np.arange(len(CLASS_NAMES)))

ax.set_xticklabels(CLASS_NAMES)

ax.set_yticklabels(CLASS_NAMES)

ax.set_xlabel("Predicted Label")

ax.set_ylabel("True Label")

ax.set_title(title)

for i in range(cm.shape\[0\]):

for j in range(cm.shape\[1\]):

ax.text(j, i, str(cm\[i, j\]), ha="center", va="center", fontsize=12)

plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.close()

def save_history_csv(history: Dict\[str, List\[float\]\], save_path: Path) -> None:

with open(save_path, "w", newline="", encoding="utf-8-sig") as f:

writer = csv.writer(f)

writer.writerow(\["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"\])

for i in range(len(history\["train_loss"\])):

writer.writerow(\[

i + 1,

history\["train_loss"\]\[i\],

history\["train_acc"\]\[i\],

history\["val_loss"\]\[i\],

history\["val_acc"\]\[i\],

history\["lr"\]\[i\],

\])

def save_result_csv(results: List\[Dict\[str, object\]\], save_path: Path) -> None:

fieldnames = \[

"model",

"train_acc_last",

"val_acc_best",

"test_accuracy",

"test_precision",

"test_recall",

"test_f1",

"train_time",

"best_epoch",

"total_params",

"trainable_params",

\]

with open(save_path, "w", newline="", encoding="utf-8-sig") as f:

writer = csv.DictWriter(f, fieldnames=fieldnames)

writer.writeheader()

for r in results:

writer.writerow({k: r.get(k, "") for k in fieldnames})

def plot_model_comparison(results: List\[Dict\[str, object\]\], save_path: Path) -> None:

names = \[str(r\["model"\]) for r in results\]

metrics = \["test_accuracy", "test_precision", "test_recall", "test_f1"\]

metric_labels = \["Accuracy", "Precision", "Recall", "F1"\]

x = np.arange(len(names))

width = 0.18

fig, ax = plt.subplots(figsize=(12, 6))

for k, metric in enumerate(metrics):

values = \[float(r\[metric\]) for r in results\]

ax.bar(x + (k - 1.5) \* width, values, width, label=metric_labels\[k\])

ax.set_xticks(x)

ax.set_xticklabels(names, rotation=15, ha="right")

ax.set_ylim(0, 1.0)

ax.set_ylabel("Score")

ax.set_title("Model Performance Comparison")

ax.grid(axis="y", alpha=0.3)

ax.legend()

plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.close()

def plot_all_loss_curves(histories: Dict\[str, Dict\[str, List\[float\]\]\], save_path: Path) -> None:

plt.figure(figsize=(10, 6))

for name, history in histories.items():

epochs = range(1, len(history\["val_loss"\]) + 1)

plt.plot(epochs, history\["val_loss"\], marker="o", label=name)

plt.xlabel("Epoch")

plt.ylabel("Validation Loss")

plt.title("Validation Loss Curves of Different Experiments")

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.close()

\# ============================================================

\# 8. Grad-CAM 可视化

\# ============================================================

class GradCAM:

"""

Grad-CAM 实现。

用目标类别对目标卷积层特征图的梯度加权，生成空间注意力热力图。

"""

def \__init_\_(self, model: nn.Module, target_layer: nn.Module) -> None:

self.model = model

self.target_layer = target_layer

self.activations: Optional\[torch.Tensor\] = None

self.gradients: Optional\[torch.Tensor\] = None

self.forward_handle = None

self.backward_handle = None

self.\_register_hooks()

def \_register_hooks(self) -> None:

def forward_hook(module, inputs, output):

self.activations = output

def backward_hook(module, grad_input, grad_output):

self.gradients = grad_output\[0\]

self.forward_handle = self.target_layer.register_forward_hook(forward_hook)

self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

def close(self) -> None:

if self.forward_handle is not None:

self.forward_handle.remove()

if self.backward_handle is not None:

self.backward_handle.remove()

def generate(self, input_tensor: torch.Tensor, target_class: Optional\[int\] = None) -> Tuple\[np.ndarray, int\]:

self.model.eval()

output = self.model(input_tensor)

if target_class is None:

target_class = int(torch.argmax(output, dim=1).item())

self.model.zero_grad()

score = output\[0, target_class\]

score.backward()

if self.gradients is None or self.activations is None:

raise RuntimeError("Grad-CAM hook 没有获取到梯度或激活值。")

gradients = self.gradients\[0\] # \[C,H,W\]

activations = self.activations\[0\] # \[C,H,W\]

weights = gradients.mean(dim=(1, 2)) # \[C\]

cam = torch.zeros(activations.shape\[1:\], dtype=torch.float32, device=activations.device)

for i, w in enumerate(weights):

cam += w \* activations\[i\]

cam = torch.relu(cam)

if torch.max(cam) > 0:

cam = cam / torch.max(cam)

return cam.detach().cpu().numpy(), target_class

def resize_heatmap(heatmap: np.ndarray, target_hw: Tuple\[int, int\]) -> np.ndarray:

h, w = target_hw

heatmap_uint8 = np.uint8(255 \* heatmap)

pil = Image.fromarray(heatmap_uint8).resize((w, h), Image.BILINEAR)

return np.array(pil).astype(np.float32) / 255.0

def save_gradcam_figure(

model: nn.Module,

target_layer: nn.Module,

sample_tensor: torch.Tensor,

true_label: int,

mean: Sequence\[float\],

std: Sequence\[float\],

device: torch.device,

save_path: Path,

) -> None:

"""

保存 Original / Heatmap / Overlay 三联图。

"""

model.eval()

input_tensor = sample_tensor.unsqueeze(0).to(device)

gradcam = GradCAM(model, target_layer)

try:

heatmap, pred_class = gradcam.generate(input_tensor)

finally:

gradcam.close()

original = inverse_normalize_tensor(sample_tensor, mean, std)

heatmap = resize_heatmap(heatmap, target_hw=(original.shape\[0\], original.shape\[1\]))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes\[0\].imshow(original)

axes\[0\].set_title(f"Original\\nTrue: {CLASS_NAMES\[true_label\]}")

axes\[0\].axis("off")

axes\[1\].imshow(heatmap, cmap="jet")

axes\[1\].set_title("Grad-CAM Heatmap")

axes\[1\].axis("off")

axes\[2\].imshow(original)

axes\[2\].imshow(heatmap, cmap="jet", alpha=0.45)

axes\[2\].set_title(f"Overlay\\nPred: {CLASS_NAMES\[pred_class\]}")

axes\[2\].axis("off")

plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.close()

def save_gradcam_samples(

model: nn.Module,

target_layer: nn.Module,

test_loader: DataLoader,

mean: Sequence\[float\],

std: Sequence\[float\],

device: torch.device,

output_dir: Path,

prefix: str,

max_samples: int = 4,

) -> None:

ensure_dir(output_dir)

count = 0

for batch in test_loader:

images, labels, _ = unpack_batch(batch)

for i in range(images.size(0)):

save_path = output_dir / f"{prefix}\_gradcam_{count + 1}.png"

save_gradcam_figure(

model=model,

target_layer=target_layer,

sample_tensor=images\[i\],

true_label=int(labels\[i\].item()),

mean=mean,

std=std,

device=device,

save_path=save_path,

)

count += 1

if count >= max_samples:

print(f"Grad-CAM 图已保存到：{output_dir}")

return

\# ============================================================

\# 9. 单个实验封装

\# ============================================================

def run_one_experiment(

experiment_name: str,

model: nn.Module,

data: DataBundle,

device: torch.device,

output_dir: Path,

epochs: int,

lr: float,

patience: int,

weight_decay: float = 0.0,

run_gradcam: bool = False,

) -> Tuple\[nn.Module, Dict\[str, List\[float\]\], Dict\[str, object\]\]:

exp_dir = output_dir / experiment_name

ensure_dir(exp_dir)

model, history, best_info = train_model(

model=model,

train_loader=data.train_loader,

val_loader=data.val_loader,

device=device,

output_dir=exp_dir,

model_name=experiment_name,

epochs=epochs,

lr=lr,

patience=patience,

weight_decay=weight_decay,

)

criterion = nn.CrossEntropyLoss()

train_metrics = evaluate_model(model, data.train_loader, device, criterion=criterion)

val_metrics = evaluate_model(model, data.val_loader, device, criterion=criterion)

test_metrics = evaluate_model(model, data.test_loader, device, criterion=criterion)

plot_training_history(history, exp_dir / f"{experiment_name}\_history.png", title=experiment_name)

plot_confusion_matrix(

test_metrics\["confusion_matrix"\],

exp_dir / f"{experiment_name}\_confusion_matrix.png",

title=f"{experiment_name} Test Confusion Matrix",

)

save_history_csv(history, exp_dir / f"{experiment_name}\_history.csv")

if run_gradcam and hasattr(model, "gradcam_target_layer"):

save_gradcam_samples(

model=model,

target_layer=model.gradcam_target_layer,

test_loader=data.test_loader,

mean=data.mean,

std=data.std,

device=device,

output_dir=exp_dir,

prefix=experiment_name,

max_samples=4,

)

result = {

"model": experiment_name,

"train_acc_last": history\["train_acc"\]\[-1\] if len(history\["train_acc"\]) > 0 else "",

"val_acc_best": best_info\["best_val_acc"\],

"test_accuracy": test_metrics\["accuracy"\],

"test_precision": test_metrics\["precision"\],

"test_recall": test_metrics\["recall"\],

"test_f1": test_metrics\["f1"\],

"train_time": best_info\["train_time"\],

"best_epoch": best_info\["best_epoch"\],

"total_params": count_total_params(model),

"trainable_params": count_trainable_params(model),

}

print("-" \* 70)

print(f"{experiment_name} 测试集结果：")

print(f"Accuracy : {float(test_metrics\['accuracy'\]):.4f}")

print(f"Precision: {float(test_metrics\['precision'\]):.4f}")

print(f"Recall : {float(test_metrics\['recall'\]):.4f}")

print(f"F1-score : {float(test_metrics\['f1'\]):.4f}")

print("Confusion Matrix:")

print(test_metrics\["confusion_matrix"\])

print("-" \* 70)

return model, history, result

\# ============================================================

\# 10. 主实验：基础 CNN、数据增强 CNN、迁移学习 CNN

\# ============================================================

def run_paper_main_experiments(args) -> None:

set_seed(args.seed)

device = get_device()

output_dir = Path(args.output_dir)

ensure_dir(output_dir)

\# 数据扫描与清洗

paths, labels = scan_cat_dog_images(Path(args.data_dir), limit_per_class=args.limit_per_class)

paths, labels = verify_and_filter_images(paths, labels, output_dir / "bad_images_report.csv")

\# 分层划分 8:1:1

train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=args.seed)

save_split_csv(paths, labels, train_idx, val_idx, test_idx, output_dir / "dataset_split.csv")

print(f"训练集：{len(train_idx)}，验证集：{len(val_idx)}，测试集：{len(test_idx)}")

print("划分文件已保存：dataset_split.csv")

\# 标准化参数：默认按论文要求使用训练集统计量；也可用 ImageNet 统计量

if args.norm == "dataset":

mean, std = compute_dataset_mean_std(

paths,

labels,

train_idx,

image_size=args.image_size,

batch_size=args.batch_size,

num_workers=args.num_workers,

)

else:

mean, std = DEFAULT_MEAN, DEFAULT_STD

print(f"使用 ImageNet mean/std：mean={mean}, std={std}")

\# 保存标准化参数

with open(output_dir / "normalization_stats.csv", "w", newline="", encoding="utf-8-sig") as f:

writer = csv.writer(f)

writer.writerow(\["channel", "mean", "std"\])

for c, (m, s) in enumerate(zip(mean, std)):

writer.writerow(\[c, m, s\])

\# DataLoader：基础 CNN 不使用数据增强；数据增强 CNN 和迁移学习 CNN 使用数据增强

base_data = create_dataloaders(

paths, labels, train_idx, val_idx, test_idx,

image_size=args.image_size,

batch_size=args.batch_size,

num_workers=args.num_workers,

mean=mean,

std=std,

use_augmentation=False,

)

aug_data = create_dataloaders(

paths, labels, train_idx, val_idx, test_idx,

image_size=args.image_size,

batch_size=args.batch_size,

num_workers=args.num_workers,

mean=mean,

std=std,

use_augmentation=True,

)

all_results: List\[Dict\[str, object\]\] = \[\]

histories: Dict\[str, Dict\[str, List\[float\]\]\] = {}

\# 实验一：基础 CNN。作为性能基线，不使用数据增强。

if args.mode in \["main", "basic", "all"\]:

model_basic = BasicCNN(num_classes=2, dropout=0.0)

\_, hist_basic, result_basic = run_one_experiment(

experiment_name="basic_cnn",

model=model_basic,

data=base_data,

device=device,

output_dir=output_dir,

epochs=args.epochs,

lr=args.lr,

patience=args.patience,

weight_decay=0.0,

run_gradcam=args.gradcam,

)

all_results.append(result_basic)

histories\["Basic CNN"\] = hist_basic

\# 实验二：数据增强 CNN。采用与基础 CNN 相同主体结构，并加入 Dropout 作为论文消融中最优组合之一。

if args.mode in \["main", "augmented", "all"\]:

model_aug = BasicCNN(num_classes=2, dropout=0.5)

\_, hist_aug, result_aug = run_one_experiment(

experiment_name="augmented_cnn",

model=model_aug,

data=aug_data,

device=device,

output_dir=output_dir,

epochs=args.epochs,

lr=args.lr,

patience=args.patience,

weight_decay=0.0,

run_gradcam=args.gradcam,

)

all_results.append(result_aug)

histories\["Augmented CNN"\] = hist_aug

\# 实验三：迁移学习 CNN。ResNet50 + 自定义分类头，冻结骨干网络。

if args.mode in \["main", "resnet", "all"\]:

model_resnet = TransferResNet50(

num_classes=2,

pretrained=args.pretrained,

freeze_backbone=True,

)

\_, hist_resnet, result_resnet = run_one_experiment(

experiment_name="transfer_resnet50",

model=model_resnet,

data=aug_data,

device=device,

output_dir=output_dir,

epochs=args.epochs,

lr=args.lr,

patience=args.patience,

weight_decay=0.0,

run_gradcam=True if args.gradcam else False,

)

all_results.append(result_resnet)

histories\["Transfer CNN (ResNet50)"\] = hist_resnet

\# 可选：LeNet-5 彩色图像对比，仅用于补充。

if args.mode == "lenet":

lenet_mean, lenet_std = \[0.5, 0.5, 0.5\], \[0.5, 0.5, 0.5\]

lenet_base = create_dataloaders(

paths, labels, train_idx, val_idx, test_idx,

image_size=32,

batch_size=args.batch_size,

num_workers=args.num_workers,

mean=lenet_mean,

std=lenet_std,

use_augmentation=False,

)

model_lenet = LeNet5Color(num_classes=2)

\_, hist_lenet, result_lenet = run_one_experiment(

experiment_name="lenet5_color",

model=model_lenet,

data=lenet_base,

device=device,

output_dir=output_dir,

epochs=args.epochs,

lr=args.lr,

patience=args.patience,

weight_decay=0.0,

run_gradcam=False,

)

all_results.append(result_lenet)

histories\["LeNet-5"\] = hist_lenet

\# 汇总结果

if len(all_results) > 0:

save_result_csv(all_results, output_dir / "paper_experiment_results.csv")

plot_model_comparison(all_results, output_dir / "model_comparison.png")

plot_all_loss_curves(histories, output_dir / "validation_loss_curves.png")

print("\\n全部实验完成。主要输出文件：")

print(f"1. {output_dir / 'dataset_split.csv'}")

print(f"2. {output_dir / 'paper_experiment_results.csv'}")

print(f"3. {output_dir / 'model_comparison.png'}")

print(f"4. {output_dir / 'validation_loss_curves.png'}")

print(f"5. 每个实验子文件夹中的 best_\*.pth、训练曲线、混淆矩阵、Grad-CAM 图")

\# ============================================================

\# 11. 消融实验：数据增强、Dropout、L2 正则化、迁移学习

\# ============================================================

def run_ablation_experiments(args) -> None:

"""

对应论文 4.3.5 消融研究：

\- 基线：无增强、无正则化

\- + 数据增强

\- + Dropout(0.5)

\- + L2 正则化(0.001)

\- + 数据增强 + Dropout

\- + 预训练 ResNet50

"""

set_seed(args.seed)

device = get_device()

output_dir = Path(args.output_dir) / "ablation"

ensure_dir(output_dir)

paths, labels = scan_cat_dog_images(Path(args.data_dir), limit_per_class=args.limit_per_class)

paths, labels = verify_and_filter_images(paths, labels, output_dir / "bad_images_report.csv")

train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=args.seed)

if args.norm == "dataset":

mean, std = compute_dataset_mean_std(

paths, labels, train_idx,

image_size=args.image_size,

batch_size=args.batch_size,

num_workers=args.num_workers,

)

else:

mean, std = DEFAULT_MEAN, DEFAULT_STD

base_data = create_dataloaders(

paths, labels, train_idx, val_idx, test_idx,

image_size=args.image_size,

batch_size=args.batch_size,

num_workers=args.num_workers,

mean=mean,

std=std,

use_augmentation=False,

)

aug_data = create_dataloaders(

paths, labels, train_idx, val_idx, test_idx,

image_size=args.image_size,

batch_size=args.batch_size,

num_workers=args.num_workers,

mean=mean,

std=std,

use_augmentation=True,

)

experiments = \[

("ablation_baseline", BasicCNN(num_classes=2, dropout=0.0), base_data, 0.0),

("ablation_plus_augmentation", BasicCNN(num_classes=2, dropout=0.0), aug_data, 0.0),

("ablation_plus_dropout", BasicCNN(num_classes=2, dropout=0.5), base_data, 0.0),

("ablation_plus_l2", BasicCNN(num_classes=2, dropout=0.0), base_data, 0.001),

("ablation_aug_dropout", BasicCNN(num_classes=2, dropout=0.5), aug_data, 0.0),

\]

results: List\[Dict\[str, object\]\] = \[\]

histories: Dict\[str, Dict\[str, List\[float\]\]\] = {}

for name, model, data, weight_decay in experiments:

\_, hist, result = run_one_experiment(

experiment_name=name,

model=model,

data=data,

device=device,

output_dir=output_dir,

epochs=args.epochs,

lr=args.lr,

patience=args.patience,

weight_decay=weight_decay,

run_gradcam=False,

)

results.append(result)

histories\[name\] = hist

\# 迁移学习作为消融最后一项

model_resnet = TransferResNet50(num_classes=2, pretrained=args.pretrained, freeze_backbone=True)

\_, hist_resnet, result_resnet = run_one_experiment(

experiment_name="ablation_pretrained_resnet50",

model=model_resnet,

data=aug_data,

device=device,

output_dir=output_dir,

epochs=args.epochs,

lr=args.lr,

patience=args.patience,

weight_decay=0.0,

run_gradcam=args.gradcam,

)

results.append(result_resnet)

histories\["ablation_pretrained_resnet50"\] = hist_resnet

\# 计算相对基线提升

baseline_acc = float(results\[0\]\["test_accuracy"\])

for r in results:

r\["relative_change"\] = float(r\["test_accuracy"\]) - baseline_acc

\# 保存消融结果

ablation_csv = output_dir / "paper_ablation_results.csv"

with open(ablation_csv, "w", newline="", encoding="utf-8-sig") as f:

fieldnames = \[

"model",

"test_accuracy",

"test_precision",

"test_recall",

"test_f1",

"relative_change",

"train_time",

"best_epoch",

\]

writer = csv.DictWriter(f, fieldnames=fieldnames)

writer.writeheader()

for r in results:

writer.writerow({k: r.get(k, "") for k in fieldnames})

plot_model_comparison(results, output_dir / "ablation_model_comparison.png")

plot_all_loss_curves(histories, output_dir / "ablation_validation_loss_curves.png")

print(f"\\n消融实验完成，结果已保存：{ablation_csv}")

\# ============================================================

\# 12. 单张图像预测

\# ============================================================

def predict_single_image(

model: nn.Module,

image_path: Path,

image_size: int,

mean: Sequence\[float\],

std: Sequence\[float\],

device: torch.device,

) -> Tuple\[str, float\]:

transform = build_base_transform(image_size, mean, std)

model.eval()

img = Image.open(image_path).convert("RGB")

x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():

out = model(x)

prob = torch.softmax(out, dim=1)\[0\]

pred = int(torch.argmax(prob).item())

conf = float(prob\[pred\].item())

return CLASS_NAMES\[pred\], conf

\# ============================================================

\# 13. 命令行入口

\# ============================================================

def parse_args():

parser = argparse.ArgumentParser(

description="基于 CNN 的猫狗图像二分类：基础 CNN、数据增强 CNN、ResNet50 迁移学习、Grad-CAM。"

)

parser.add_argument("--data_dir", type=str,

default=r"C:\\\\Users\\\\QJX03\\\\data\\\\dogs_vs_cats\\\\PetImages",

help="猫狗数据集目录。")

parser.add_argument("--output_dir", type=str,

default=r"C:\\\\Users\\\\QJX03\\\\results_paper_cnn",

help="结果输出目录。")

parser.add_argument("--mode", type=str, default="main",

choices=\["main", "basic", "augmented", "resnet", "lenet", "all", "ablation"\],

help="运行模式：main=论文三组主实验；ablation=消融实验。")

parser.add_argument("--epochs", type=int, default=30,

help="最大训练轮数，论文设置为 30。")

parser.add_argument("--batch_size", type=int, default=32,

help="批量大小，论文设置为 32。")

parser.add_argument("--lr", type=float, default=0.001,

help="学习率，论文设置为 0.001。")

parser.add_argument("--patience", type=int, default=10,

help="早停 patience，论文设置为验证损失/验证准确率连续 10 轮未改善时停止。")

parser.add_argument("--image_size", type=int, default=224,

help="输入图像尺寸，论文使用 224x224。")

parser.add_argument("--num_workers", type=int, default=0,

help="DataLoader 进程数。Windows 下建议先用 0，稳定后可改为 2 或 4。")

parser.add_argument("--seed", type=int, default=42,

help="随机种子。")

parser.add_argument("--norm", type=str, default="dataset", choices=\["dataset", "imagenet"\],

help="标准化方式：dataset=计算训练集均值方差；imagenet=使用 ImageNet 均值方差。")

parser.add_argument("--limit_per_class", type=int, default=None,

help="每类最多使用多少张图像。正式实验不要设置；快速调试可设为 100。")

parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,

help="ResNet50 是否使用 ImageNet 预训练权重。默认启用。")

parser.add_argument("--gradcam", action=argparse.BooleanOptionalAction, default=False,

help="是否保存 Grad-CAM 可视化图。默认关闭；需要时可添加 --gradcam 启用。")

return parser.parse_args()

def main() -> None:

args = parse_args()

print("=" \* 70)

print("基于卷积神经网络的图像分类应用研究")

print("——从基础 CNN 到迁移学习的递进实践")

print("=" \* 70)

print("运行参数：")

for k, v in vars(args).items():

print(f"{k}: {v}")

print("=" \* 70)

if args.mode == "ablation":

run_ablation_experiments(args)

else:

run_paper_main_experiments(args)

if \__name__ == "\__main_\_":

main()