# -*- coding: utf-8 -*-
"""Quick Cats vs Dogs CNN experiments.

This script implements the workflow described in the markdown note:
scan Kaggle Dogs vs Cats images, clean readable files, make an 8:1:1
stratified split, train Basic CNN / augmented CNN / ResNet50 transfer
models, then save metrics, curves, confusion matrices, and checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import models, transforms
except Exception as exc:  # pragma: no cover - import failure is environment specific.
    raise ImportError("This script requires torchvision. Install requirements first.") from exc


ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

CLASS_NAMES = ["Cat", "Dog"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def scan_cat_dog_images(
    data_dir: Path, limit_per_class: Optional[int] = None
) -> Tuple[List[Path], List[int]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    candidates: List[Tuple[Path, int]] = []
    possible_roots = [data_dir, data_dir / "train"]

    for root in possible_roots:
        for dirname, label in [("cat", 0), ("dog", 1)]:
            class_dir = root / dirname
            if class_dir.exists():
                for path in sorted(class_dir.rglob("*")):
                    if path.suffix.lower() in IMAGE_EXTS:
                        candidates.append((path, label))

    if not candidates:
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            name = path.name.lower()
            if name.startswith("cat"):
                candidates.append((path, 0))
            elif name.startswith("dog"):
                candidates.append((path, 1))

    if not candidates:
        raise RuntimeError(f"No cat/dog images found under {data_dir}")

    if limit_per_class and limit_per_class > 0:
        limited: List[Tuple[Path, int]] = []
        for label in [0, 1]:
            limited.extend([(p, y) for p, y in candidates if y == label][:limit_per_class])
        candidates = limited

    paths = [path for path, _ in candidates]
    labels = [label for _, label in candidates]
    cat_count = sum(label == 0 for label in labels)
    dog_count = sum(label == 1 for label in labels)
    print(f"扫描完成：共 {len(paths)} 张图像，其中 Cat={cat_count}, Dog={dog_count}")
    if cat_count == 0 or dog_count == 0:
        raise RuntimeError("Both cat and dog classes are required.")
    return paths, labels


def verify_and_filter_images(
    paths: Sequence[Path], labels: Sequence[int], report_path: Path
) -> Tuple[List[Path], List[int]]:
    valid_paths: List[Path] = []
    valid_labels: List[int] = []
    bad_rows: List[List[str]] = []

    for path, label in zip(paths, labels):
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB")
            valid_paths.append(path)
            valid_labels.append(int(label))
        except Exception as exc:
            bad_rows.append([str(path), str(exc)])

    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bad_image_path", "error"])
        writer.writerows(bad_rows)

    print(f"数据清洗完成：有效图像 {len(valid_paths)} 张，损坏图像 {len(bad_rows)} 张")
    print(f"损坏图像报告：{report_path}")
    return valid_paths, valid_labels


def stratified_split_indices(
    labels: Sequence[int], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for class_id in sorted(np.unique(labels_arr)):
        idx = np.where(labels_arr == class_id)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def save_split_csv(
    paths: Sequence[Path],
    labels: Sequence[int],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    save_path: Path,
) -> None:
    idx_to_split = {i: "train" for i in train_idx}
    idx_to_split.update({i: "val" for i in val_idx})
    idx_to_split.update({i: "test" for i in test_idx})

    with save_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "path", "label", "class_name"])
        for i, path in enumerate(paths):
            writer.writerow([idx_to_split[i], str(path), labels[i], CLASS_NAMES[labels[i]]])


class CatsDogsDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform=None,
        return_path: bool = False,
    ) -> None:
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = int(self.labels[idx])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, label, str(path)
        return image, label


def build_base_transform(image_size: int, mean: Sequence[float], std: Sequence[float]):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_aug_transform(image_size: int, mean: Sequence[float], std: Sequence[float]):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def compute_dataset_mean_std(
    image_paths: Sequence[Path],
    labels: Sequence[int],
    indices: Sequence[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[float], List[float]]:
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    sub_paths = [image_paths[i] for i in indices]
    sub_labels = [labels[i] for i in indices]
    dataset = CatsDogsDataset(sub_paths, sub_labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0
    print("正在计算训练集均值和标准差...")
    for images, _ in loader:
        b, _, h, w = images.shape
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images**2).sum(dim=[0, 2, 3])
        num_pixels += b * h * w

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sq_sum / num_pixels - mean**2)
    print(f"训练集 mean = {[round(x, 4) for x in mean.tolist()]}")
    print(f"训练集 std = {[round(x, 4) for x in std.tolist()]}")
    return mean.tolist(), std.tolist()


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    mean: List[float]
    std: List[float]


def create_dataloaders(
    image_paths: Sequence[Path],
    labels: Sequence[int],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: Sequence[float],
    std: Sequence[float],
    use_augmentation: bool,
) -> DataBundle:
    def select(indices: Sequence[int]) -> Tuple[List[Path], List[int]]:
        return [image_paths[i] for i in indices], [labels[i] for i in indices]

    train_paths, train_labels = select(train_idx)
    val_paths, val_labels = select(val_idx)
    test_paths, test_labels = select(test_idx)
    train_transform = (
        build_aug_transform(image_size, mean, std)
        if use_augmentation
        else build_base_transform(image_size, mean, std)
    )
    eval_transform = build_base_transform(image_size, mean, std)
    pin_memory = torch.cuda.is_available()

    return DataBundle(
        train_loader=DataLoader(
            CatsDogsDataset(train_paths, train_labels, train_transform, return_path=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        val_loader=DataLoader(
            CatsDogsDataset(val_paths, val_labels, eval_transform, return_path=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        test_loader=DataLoader(
            CatsDogsDataset(test_paths, test_labels, eval_transform, return_path=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        mean=list(mean),
        std=list(std),
    )


class BasicCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 14 * 14, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(300, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, start_dim=1))

    @property
    def gradcam_target_layer(self):
        return self.features[12]


class TransferResNet50(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = None
        try:
            self.backbone = models.resnet50(weights=weights)
            if weights is not None:
                print("已加载 ResNet50 ImageNet 预训练权重。")
        except Exception as exc:
            print("警告：ResNet50 预训练权重加载失败，将使用随机初始化权重。")
            print(f"原因：{exc}")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def gradcam_target_layer(self):
        return self.backbone.layer4[-1]


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, object]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    cm = compute_confusion_matrix(y_true_arr, y_pred_arr)
    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for class_id in range(2):
        tp = cm[class_id, class_id]
        fp = cm[:, class_id].sum() - tp
        fn = cm[class_id, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "confusion_matrix": cm,
    }


def unpack_batch(batch):
    if len(batch) == 3:
        return batch
    images, labels = batch
    return images, labels, None


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: Optional[nn.Module] = None
) -> Dict[str, object]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            images, labels, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if criterion is not None:
                total_loss += float(criterion(outputs, labels).item())
                n_batches += 1
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    model_name: str,
    epochs: int,
    lr: float,
    patience: int,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, object]]:
    ensure_dir(output_dir)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = output_dir / f"best_{model_name}.pth"

    print("=" * 70)
    print(f"开始训练：{model_name}")
    print(f"总参数量：{count_total_params(model):,}")
    print(f"可训练参数：{count_trainable_params(model):,}")
    print(f"设备：{device}")
    print("=" * 70)

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
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["accuracy"])
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
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
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    best_info = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_time": format_seconds(train_time),
        "best_model_path": str(best_model_path),
    }
    print(f"训练完成：{model_name}")
    print(f"最佳验证准确率：{best_val_acc:.4f}，出现在 Epoch {best_epoch}")
    print(f"训练耗时：{format_seconds(train_time)}")
    return model, history, best_info


def plot_training_history(history: Dict[str, List[float]], save_path: Path, title: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], marker="s", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
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
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_history_csv(history: Dict[str, List[float]], save_path: Path) -> None:
    with save_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        for i in range(len(history["train_loss"])):
            writer.writerow(
                [
                    i + 1,
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["val_loss"][i],
                    history["val_acc"][i],
                    history["lr"][i],
                ]
            )


def save_result_csv(results: List[Dict[str, object]], save_path: Path) -> None:
    fieldnames = [
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
    ]
    with save_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in fieldnames})


def plot_model_comparison(results: List[Dict[str, object]], save_path: Path) -> None:
    names = [str(result["model"]) for result in results]
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [float(r[metric]) for r in results], width, label=labels[i])
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


def plot_all_loss_curves(histories: Dict[str, Dict[str, List[float]]], save_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        plt.plot(epochs, history["val_loss"], marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Curves of Different Experiments")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_one_experiment(
    experiment_name: str,
    model: nn.Module,
    data: DataBundle,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    lr: float,
    patience: int,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, object]]:
    exp_dir = output_dir / experiment_name
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
    )

    criterion = nn.CrossEntropyLoss()
    train_metrics = evaluate_model(model, data.train_loader, device, criterion)
    test_metrics = evaluate_model(model, data.test_loader, device, criterion)
    plot_training_history(history, exp_dir / f"{experiment_name}_history.png", experiment_name)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        exp_dir / f"{experiment_name}_confusion_matrix.png",
        f"{experiment_name} Test Confusion Matrix",
    )
    save_history_csv(history, exp_dir / f"{experiment_name}_history.csv")

    result = {
        "model": experiment_name,
        "train_acc_last": train_metrics["accuracy"],
        "val_acc_best": best_info["best_val_acc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "train_time": best_info["train_time"],
        "best_epoch": best_info["best_epoch"],
        "total_params": count_total_params(model),
        "trainable_params": count_trainable_params(model),
    }
    print("-" * 70)
    print(f"{experiment_name} 测试集结果：")
    print(f"Accuracy : {float(test_metrics['accuracy']):.4f}")
    print(f"Precision: {float(test_metrics['precision']):.4f}")
    print(f"Recall   : {float(test_metrics['recall']):.4f}")
    print(f"F1-score : {float(test_metrics['f1']):.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])
    print("-" * 70)
    return model, history, result


def run_main_experiments(args) -> None:
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    paths, labels = scan_cat_dog_images(Path(args.data_dir), limit_per_class=args.limit_per_class)
    paths, labels = verify_and_filter_images(paths, labels, output_dir / "bad_images_report.csv")
    train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=args.seed)
    save_split_csv(paths, labels, train_idx, val_idx, test_idx, output_dir / "dataset_split.csv")
    print(f"训练集：{len(train_idx)}，验证集：{len(val_idx)}，测试集：{len(test_idx)}")
    print(f"划分文件已保存：{output_dir / 'dataset_split.csv'}")

    if args.norm == "dataset":
        mean, std = compute_dataset_mean_std(
            paths, labels, train_idx, args.image_size, args.batch_size, args.num_workers
        )
    else:
        mean, std = DEFAULT_MEAN, DEFAULT_STD
        print(f"使用 ImageNet mean/std：mean={mean}, std={std}")

    with (output_dir / "normalization_stats.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "mean", "std"])
        for channel, (m, s) in enumerate(zip(mean, std)):
            writer.writerow([channel, m, s])

    base_data = create_dataloaders(
        paths,
        labels,
        train_idx,
        val_idx,
        test_idx,
        args.image_size,
        args.batch_size,
        args.num_workers,
        mean,
        std,
        use_augmentation=False,
    )
    aug_data = create_dataloaders(
        paths,
        labels,
        train_idx,
        val_idx,
        test_idx,
        args.image_size,
        args.batch_size,
        args.num_workers,
        mean,
        std,
        use_augmentation=True,
    )

    results: List[Dict[str, object]] = []
    histories: Dict[str, Dict[str, List[float]]] = {}

    if args.mode in ["main", "basic", "all"]:
        _, history, result = run_one_experiment(
            "basic_cnn",
            BasicCNN(num_classes=2, dropout=0.0),
            base_data,
            device,
            output_dir,
            args.epochs,
            args.lr,
            args.patience,
        )
        results.append(result)
        histories["Basic CNN"] = history

    if args.mode in ["main", "augmented", "all"]:
        _, history, result = run_one_experiment(
            "augmented_cnn",
            BasicCNN(num_classes=2, dropout=0.5),
            aug_data,
            device,
            output_dir,
            args.epochs,
            args.lr,
            args.patience,
        )
        results.append(result)
        histories["Augmented CNN"] = history

    if args.mode in ["main", "resnet", "all"]:
        _, history, result = run_one_experiment(
            "transfer_resnet50",
            TransferResNet50(num_classes=2, pretrained=args.pretrained, freeze_backbone=True),
            aug_data,
            device,
            output_dir,
            args.epochs,
            args.lr,
            args.patience,
        )
        results.append(result)
        histories["Transfer CNN (ResNet50)"] = history

    if results:
        save_result_csv(results, output_dir / "paper_experiment_results.csv")
        plot_model_comparison(results, output_dir / "model_comparison.png")
        plot_all_loss_curves(histories, output_dir / "validation_loss_curves.png")
        print("\n全部实验完成。主要输出文件：")
        print(output_dir / "dataset_split.csv")
        print(output_dir / "paper_experiment_results.csv")
        print(output_dir / "model_comparison.png")
        print(output_dir / "validation_loss_curves.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Cats vs Dogs CNN quick experiments.")
    parser.add_argument("--data_dir", type=str, default="doc/dogs-vs-cats-redux-kernels-edition/train")
    parser.add_argument("--output_dir", type=str, default="runs/quick_verify")
    parser.add_argument("--mode", type=str, default="main", choices=["main", "basic", "augmented", "resnet", "all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm", type=str, default="dataset", choices=["dataset", "imagenet"])
    parser.add_argument("--limit_per_class", type=int, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradcam", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("基于卷积神经网络的图像分类应用研究")
    print("——从基础 CNN 到迁移学习的递进实践")
    print("=" * 70)
    print("运行参数：")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 70)
    if args.gradcam:
        print("提示：当前快速脚本接受 --gradcam 参数，但快速验证默认不生成 Grad-CAM。")
    run_main_experiments(args)


if __name__ == "__main__":
    main()
