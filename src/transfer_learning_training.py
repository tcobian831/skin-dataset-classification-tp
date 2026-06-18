# src/transfer_learning_training.py
#
# Transfer learning para Parte 2 del TP.
#
# Objetivo:
# - Probar ResNet18 preentrenada en ImageNet.
# - Comparar contra la mejor CNN entrenada desde cero.
# - Registrar todo en TensorBoard y MLflow.
#
# Estrategias:
# - freeze: se congelan todas las capas convolucionales y se entrena solo la capa final.
# - fine_tune_last_block: se entrena layer4 + capa final.
# - fine_tune_all: se entrena toda la red.

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from cnn_training import (
    SkinDataset,
    build_class_mapping,
    compute_metrics,
    count_trainable_params,
    get_device,
    safe_log_params,
    set_seed,
    setup_mlflow,
)


# ============================================================
# Transforms
# ============================================================

def get_transfer_transforms(image_size: int = 224, augmentation: str = "minimal"):
    """
    Transforms para ResNet18 preentrenada.

    Se usa normalización ImageNet porque la red fue preentrenada con esa escala.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if augmentation == "minimal":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif augmentation == "light":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif augmentation == "medium":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0),
                ratio=(0.90, 1.10),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.12,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    else:
        raise ValueError(f"augmentation no reconocida: {augmentation}")

    return train_transform, val_transform


# ============================================================
# Modelo transfer learning
# ============================================================

def build_resnet18_transfer(
    num_classes: int,
    strategy: str = "freeze",
    dropout: float = 0.3,
) -> nn.Module:
    """
    Construye ResNet18 preentrenada y reemplaza la capa final.

    strategy:
    - freeze: entrena solo clasificador final.
    - fine_tune_last_block: entrena layer4 + clasificador final.
    - fine_tune_all: entrena toda la red.
    """

    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    if strategy == "freeze":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "fine_tune_last_block":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "fine_tune_all":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"strategy no reconocida: {strategy}")

    return model


def build_optimizer(model: nn.Module, config: Dict[str, Any]):
    """
    Optimizer solo sobre parámetros entrenables.
    """

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer_name = config.get("optimizer", "adamw").lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    raise ValueError(f"Optimizador no reconocido: {optimizer_name}")


# ============================================================
# DataLoaders
# ============================================================

def make_transfer_dataloaders_for_fold(
    df: pd.DataFrame,
    fold: int,
    class_to_idx: Dict[str, int],
    config: Dict[str, Any],
):
    """
    Crea train_loader y val_loader para un fold.
    """

    trainval_df = df[df["subset"] == "trainval"].copy()
    trainval_df["fold"] = trainval_df["fold"].astype(int)

    train_df = trainval_df[trainval_df["fold"] != fold].copy()
    val_df = trainval_df[trainval_df["fold"] == fold].copy()

    train_transform, val_transform = get_transfer_transforms(
        image_size=config["image_size"],
        augmentation=config["augmentation"],
    )

    train_dataset = SkinDataset(train_df, class_to_idx, transform=train_transform)
    val_dataset = SkinDataset(val_df, class_to_idx, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, train_df, val_df


# ============================================================
# Entrenamiento
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred)

    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred)

    return epoch_loss, metrics, y_true, y_pred


def train_one_fold_transfer(
    df: pd.DataFrame,
    fold: int,
    class_to_idx: Dict[str, int],
    config: Dict[str, Any],
    device: torch.device,
):
    """
    Entrena ResNet18 transfer learning para un fold.
    """

    set_seed(config.get("seed", 42) + fold)

    train_loader, val_loader, train_df, val_df = make_transfer_dataloaders_for_fold(
        df=df,
        fold=fold,
        class_to_idx=class_to_idx,
        config=config,
    )

    model = build_resnet18_transfer(
        num_classes=len(class_to_idx),
        strategy=config["strategy"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    writer = None

    if config.get("tensorboard", True) and SummaryWriter is not None:
        log_dir = Path("runs") / "transfer_learning" / config["experiment"] / f"fold_{fold}"
        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_text("config", json.dumps(config, indent=2, ensure_ascii=False))

    best_val_accuracy = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    history_rows = []

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_metrics, val_true, val_pred = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "experiment": config["experiment"],
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }

        history_rows.append(row)

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar("accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("macro_f1/train", train_metrics["macro_f1"], epoch)
            writer.add_scalar("macro_f1/val", val_metrics["macro_f1"], epoch)

        if mlflow.active_run() is not None:
            mlflow.log_metric(f"fold_{fold}_train_loss", train_loss, step=epoch)
            mlflow.log_metric(f"fold_{fold}_val_loss", val_loss, step=epoch)
            mlflow.log_metric(f"fold_{fold}_train_accuracy", train_metrics["accuracy"], step=epoch)
            mlflow.log_metric(f"fold_{fold}_val_accuracy", val_metrics["accuracy"], step=epoch)

        print(
            f"[{config['experiment']}] fold {fold} | "
            f"epoch {epoch:02d}/{config['epochs']} | "
            f"train_acc={train_metrics['accuracy']:.3f} | "
            f"val_acc={val_metrics['accuracy']:.3f} | "
            f"val_f1={val_metrics['macro_f1']:.3f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.get("early_stopping", True):
            patience = config.get("patience", 5)

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping en fold {fold}. "
                    f"Mejor epoch: {best_epoch}, mejor val_acc: {best_val_accuracy:.3f}"
                )
                break

    model.load_state_dict(best_state)

    val_loss, val_metrics, val_true, val_pred = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    if writer is not None:
        writer.close()

    fold_result = {
        "experiment": config["experiment"],
        "fold": fold,
        "best_epoch": best_epoch,
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "n_train": len(train_df),
        "n_val": len(val_df),
    }

    if mlflow.active_run() is not None:
        mlflow.log_metric(f"fold_{fold}_best_epoch", best_epoch)
        mlflow.log_metric(f"fold_{fold}_best_val_accuracy", val_metrics["accuracy"])
        mlflow.log_metric(f"fold_{fold}_best_val_macro_f1", val_metrics["macro_f1"])
        mlflow.log_metric(
            f"fold_{fold}_best_val_balanced_accuracy",
            val_metrics["balanced_accuracy"],
        )

    return {
        "model": model,
        "fold_result": fold_result,
        "history": history_rows,
        "y_true": val_true,
        "y_pred": val_pred,
    }


def build_summary(
    config: Dict[str, Any],
    fold_results_df: pd.DataFrame,
    folds_to_run: List[int],
    best_fold_id: Optional[int],
    mlflow_run_id: Optional[str],
    num_params: int,
):
    """
    Summary final del experimento.
    """

    return {
        "experiment": config["experiment"],
        "model": "ResNet18_pretrained",
        "strategy": config["strategy"],
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "lr": config["lr"],
        "optimizer": config["optimizer"],
        "dropout": config["dropout"],
        "weight_decay": config["weight_decay"],
        "augmentation": config["augmentation"],
        "early_stopping": config["early_stopping"],
        "folds_run": folds_to_run,
        "best_fold": best_fold_id,
        "num_trainable_params": num_params,
        "val_accuracy_mean": float(fold_results_df["val_accuracy"].mean()),
        "val_accuracy_std": float(fold_results_df["val_accuracy"].std(ddof=0)),
        "val_macro_f1_mean": float(fold_results_df["val_macro_f1"].mean()),
        "val_macro_f1_std": float(fold_results_df["val_macro_f1"].std(ddof=0)),
        "val_balanced_accuracy_mean": float(fold_results_df["val_balanced_accuracy"].mean()),
        "val_balanced_accuracy_std": float(fold_results_df["val_balanced_accuracy"].std(ddof=0)),
        "mlflow_run_id": mlflow_run_id,
    }


def run_transfer_cross_validation(
    config: Dict[str, Any],
    split_csv: str | Path = "data/splits/final_split_5fold.csv",
    folds_to_run: Optional[List[int]] = None,
):
    """
    Corre validación cruzada usando transfer learning.
    """

    set_seed(config.get("seed", 42))

    df = pd.read_csv(split_csv)

    trainval_df = df[df["subset"] == "trainval"].copy()
    class_to_idx = build_class_mapping(trainval_df)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if folds_to_run is None:
        folds_to_run = sorted(trainval_df["fold"].astype(int).unique().tolist())

    device = get_device()

    print(f"Device usado: {device}")
    print(f"Folds a correr: {folds_to_run}")
    print(f"Clases: {idx_to_class}")

    dummy_model = build_resnet18_transfer(
        num_classes=len(class_to_idx),
        strategy=config["strategy"],
        dropout=config["dropout"],
    )

    num_params = count_trainable_params(dummy_model)

    print(f"Parámetros entrenables: {num_params:,}")

    setup_mlflow(config.get("mlflow_experiment", "TP_skin_transfer_learning"))

    all_fold_results = []
    all_history = []
    all_y_true = []
    all_y_pred = []

    best_fold_accuracy = -np.inf
    best_fold_model = None
    best_fold_id = None

    mlflow_run_id = None

    with mlflow.start_run(run_name=config["experiment"]) as run:
        mlflow_run_id = run.info.run_id

        safe_log_params(config)
        mlflow.log_param("split_csv", str(split_csv))
        mlflow.log_param("folds_to_run", json.dumps(folds_to_run))
        mlflow.log_param("num_classes", len(class_to_idx))
        mlflow.log_param("num_trainable_params", num_params)
        mlflow.log_param(
            "classes",
            json.dumps(
                [idx_to_class[i] for i in range(len(idx_to_class))],
                ensure_ascii=False,
            ),
        )

        mlflow.set_tag("part", "part2_transfer_learning")
        mlflow.set_tag("model_family", "ResNet18")
        mlflow.set_tag("pretrained", "ImageNet")
        mlflow.set_tag("uses_test_set", "false")

        for fold in folds_to_run:
            output = train_one_fold_transfer(
                df=df,
                fold=fold,
                class_to_idx=class_to_idx,
                config=config,
                device=device,
            )

            fold_result = output["fold_result"]

            all_fold_results.append(fold_result)
            all_history.extend(output["history"])
            all_y_true.extend(output["y_true"])
            all_y_pred.extend(output["y_pred"])

            if fold_result["val_accuracy"] > best_fold_accuracy:
                best_fold_accuracy = fold_result["val_accuracy"]
                best_fold_model = copy.deepcopy(output["model"]).cpu()
                best_fold_id = fold

        fold_results_df = pd.DataFrame(all_fold_results)

        summary = build_summary(
            config=config,
            fold_results_df=fold_results_df,
            folds_to_run=folds_to_run,
            best_fold_id=best_fold_id,
            mlflow_run_id=mlflow_run_id,
            num_params=num_params,
        )

        mlflow.log_metric("cv_val_accuracy_mean", summary["val_accuracy_mean"])
        mlflow.log_metric("cv_val_accuracy_std", summary["val_accuracy_std"])
        mlflow.log_metric("cv_val_macro_f1_mean", summary["val_macro_f1_mean"])
        mlflow.log_metric("cv_val_balanced_accuracy_mean", summary["val_balanced_accuracy_mean"])

        if best_fold_model is not None and config.get("log_pytorch_model", False):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "best_transfer_state_dict.pth"
                torch.save(best_fold_model.state_dict(), model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model_state_dict")

            mlflow.pytorch.log_model(
                pytorch_model=best_fold_model,
                artifact_path="best_pytorch_model",
            )

    cv_output = {
        "config": config,
        "summary": summary,
        "fold_results": pd.DataFrame(all_fold_results),
        "history": pd.DataFrame(all_history),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "classes": [idx_to_class[i] for i in range(len(idx_to_class))],
        "mlflow_run_id": mlflow_run_id,
    }

    cv_output["results_df"] = cv_output["fold_results"]
    cv_output["history_df"] = cv_output["history"]

    return cv_output