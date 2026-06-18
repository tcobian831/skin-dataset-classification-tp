# src/final_test_training.py
#
# Test final del TP.
#
# En esta etapa:
# 1. Se toman los hiperparámetros elegidos por validación cruzada.
# 2. Se reentrena cada modelo usando TODO trainval.
# 3. Se evalúa una única vez sobre el test fijo.
#
# Modelos evaluados:
# - Mejor MLP.
# - Mejor CNN desde cero.
# - Mejor transfer learning.
#
# Importante:
# - No se usa test para elegir hiperparámetros.
# - No se usa early stopping con test.
# - El test se mira una sola vez al final.

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Reutilizamos código ya validado en las partes anteriores.
from mlp_training import (
    SkinDataset as MLPSkinDataset,
    MLPClassifier,
    build_class_mapping,
    get_transforms as get_mlp_transforms,
    initialize_weights,
    build_optimizer as build_mlp_optimizer,
    train_one_epoch as train_one_epoch_mlp,
    evaluate as evaluate_mlp,
)

from cnn_training import (
    SkinDataset as CNNSkinDataset,
    get_transforms as get_cnn_transforms,
    build_model as build_cnn_model,
    build_optimizer as build_cnn_optimizer,
    train_one_epoch as train_one_epoch_cnn,
    evaluate as evaluate_cnn,
)

from transfer_learning_training import (
    get_transfer_transforms,
    build_resnet18_transfer,
    build_optimizer as build_transfer_optimizer,
    train_one_epoch as train_one_epoch_transfer,
    evaluate as evaluate_transfer,
)


# ============================================================
# Utilidades generales
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Fija semillas pseudoaleatorias.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Usa GPU si está disponible; si no, CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> None:
    """
    Crea carpeta si no existe.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_mlflow_final(experiment_name: str = "TP_skin_final_test") -> None:
    """
    Configura MLflow local para test final.
    """
    mlruns_path = Path("mlruns").resolve()
    mlruns_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment(experiment_name)


def safe_log_params(params: Dict[str, Any]) -> None:
    """
    Loguea params en MLflow convirtiendo listas/dicts a JSON.
    """
    for key, value in params.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value)
        mlflow.log_param(key, value)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Métricas principales para comparar modelos.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def count_trainable_params(model: nn.Module) -> int:
    """
    Cuenta parámetros entrenables.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# DataLoaders para test final
# ============================================================

def make_final_dataloaders(
    split_csv: str | Path,
    config: Dict[str, Any],
    model_family: str,
):
    """
    Construye loaders finales.

    train_loader:
        usa TODO subset == trainval.

    test_loader:
        usa subset == test.

    No hay validación en esta etapa.
    """

    df = pd.read_csv(split_csv)

    train_df = df[df["subset"] == "trainval"].copy()
    test_df = df[df["subset"] == "test"].copy()

    class_to_idx = build_class_mapping(train_df)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    image_size = config["image_size"]
    augmentation = config["augmentation"]

    if model_family == "MLP":
        train_transform, test_transform = get_mlp_transforms(
            image_size=image_size,
            augmentation=augmentation,
        )
        dataset_class = MLPSkinDataset

    elif model_family == "CNN":
        train_transform, test_transform = get_cnn_transforms(
            image_size=image_size,
            augmentation=augmentation,
        )
        dataset_class = CNNSkinDataset

    elif model_family == "TL":
        train_transform, test_transform = get_transfer_transforms(
            image_size=image_size,
            augmentation=augmentation,
        )
        dataset_class = CNNSkinDataset

    else:
        raise ValueError(f"model_family no reconocido: {model_family}")

    train_dataset = dataset_class(
        train_df,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )

    test_dataset = dataset_class(
        test_df,
        class_to_idx=class_to_idx,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader, train_df, test_df, class_to_idx, classes


# ============================================================
# Construcción de modelos finales
# ============================================================

def build_final_model(
    config: Dict[str, Any],
    model_family: str,
    num_classes: int,
) -> nn.Module:
    """
    Construye MLP, CNN o transfer learning según model_family.
    """

    if model_family == "MLP":
        input_dim = 3 * config["image_size"] * config["image_size"]

        model = MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=config.get("hidden_dims", [512, 128]),
            dropout=config["dropout"],
            batch_norm=config["batch_norm"],
        )

        initialize_weights(
            model,
            method=config.get("initialization", "default"),
        )

        return model

    if model_family == "CNN":
        return build_cnn_model(
            config=config,
            num_classes=num_classes,
        )

    if model_family == "TL":
        return build_resnet18_transfer(
            num_classes=num_classes,
            strategy=config["strategy"],
            dropout=config["dropout"],
        )

    raise ValueError(f"model_family no reconocido: {model_family}")


def build_final_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    model_family: str,
):
    """
    Construye optimizer correcto para cada familia de modelos.
    """

    if model_family == "MLP":
        return build_mlp_optimizer(model, config)

    if model_family == "CNN":
        return build_cnn_optimizer(model, config)

    if model_family == "TL":
        return build_transfer_optimizer(model, config)

    raise ValueError(f"model_family no reconocido: {model_family}")


def get_train_eval_functions(model_family: str):
    """
    Devuelve funciones de entrenamiento/evaluación adecuadas.
    """

    if model_family == "MLP":
        return train_one_epoch_mlp, evaluate_mlp

    if model_family == "CNN":
        return train_one_epoch_cnn, evaluate_cnn

    if model_family == "TL":
        return train_one_epoch_transfer, evaluate_transfer

    raise ValueError(f"model_family no reconocido: {model_family}")


# ============================================================
# Gráficos y guardado
# ============================================================

def plot_training_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    """
    Grafica train_loss y train_accuracy del entrenamiento final.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["train_accuracy"], label="train_accuracy")
    ax2.set_ylabel("Accuracy")

    plt.title("Final training curve")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    output_path: Path,
) -> None:
    """
    Guarda matriz de confusión del test.
    """
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Final test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=90)
    plt.yticks(ticks=np.arange(len(classes)), labels=classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_final_outputs(
    output: Dict[str, Any],
    output_prefix: str,
) -> List[Path]:
    """
    Guarda artifacts del test final.
    """

    ensure_dir("experiments/final_test")
    ensure_dir("results/final_test/training_curves")
    ensure_dir("results/final_test/confusion_matrices")
    ensure_dir("results/final_test/classification_reports")
    ensure_dir("models/final_test")

    prefix = output_prefix

    history_path = Path("experiments/final_test") / f"{prefix}_history.csv"
    summary_path = Path("experiments/final_test") / f"{prefix}_summary.json"
    predictions_path = Path("experiments/final_test") / f"{prefix}_predictions.csv"

    curve_path = Path("results/final_test/training_curves") / f"{prefix}_training_curve.png"
    cm_path = Path("results/final_test/confusion_matrices") / f"{prefix}.png"
    report_path = Path("results/final_test/classification_reports") / f"{prefix}_classification_report.txt"
    model_path = Path("models/final_test") / f"{prefix}_state_dict.pth"

    output["history"].to_csv(history_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output["summary"], f, indent=2, ensure_ascii=False)

    output["predictions"].to_csv(predictions_path, index=False)

    plot_training_curve(
        history_df=output["history"],
        output_path=curve_path,
    )

    save_confusion_matrix(
        y_true=output["y_true"],
        y_pred=output["y_pred"],
        classes=output["classes"],
        output_path=cm_path,
    )

    labels = list(range(len(output["classes"])))

    report = classification_report(
        output["y_true"],
        output["y_pred"],
        labels=labels,
        target_names=output["classes"],
        zero_division=0,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    torch.save(output["model_state_dict"], model_path)

    output_paths = [
        history_path,
        summary_path,
        predictions_path,
        curve_path,
        cm_path,
        report_path,
        model_path,
    ]

    return output_paths


def log_artifacts_to_mlflow(run_id: str, artifact_paths: List[Path]) -> None:
    """
    Loguea artifacts ya guardados al run de MLflow.
    """
    with mlflow.start_run(run_id=run_id):
        for path in artifact_paths:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="outputs")


# ============================================================
# Entrenamiento final y test
# ============================================================

def run_final_test_experiment(
    config: Dict[str, Any],
    model_family: str,
    split_csv: str | Path = "data/splits/final_split_5fold.csv",
):
    """
    Entrena un modelo final sobre todo trainval y evalúa una vez sobre test.
    """

    set_seed(config.get("seed", 42))

    device = get_device()

    print("=" * 80)
    print(f"Experimento final: {config['experiment']}")
    print(f"Familia: {model_family}")
    print(f"Device: {device}")
    print("=" * 80)

    train_loader, test_loader, train_df, test_df, class_to_idx, classes = make_final_dataloaders(
        split_csv=split_csv,
        config=config,
        model_family=model_family,
    )

    model = build_final_model(
        config=config,
        model_family=model_family,
        num_classes=len(class_to_idx),
    ).to(device)

    optimizer = build_final_optimizer(
        model=model,
        config=config,
        model_family=model_family,
    )

    criterion = nn.CrossEntropyLoss()
    train_one_epoch_fn, evaluate_fn = get_train_eval_functions(model_family)

    num_trainable_params = count_trainable_params(model)

    setup_mlflow_final(config.get("mlflow_experiment", "TP_skin_final_test"))

    writer = None
    if SummaryWriter is not None and config.get("tensorboard", True):
        log_dir = Path("runs") / "final_test" / config["experiment"]
        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_text("config", json.dumps(config, indent=2, ensure_ascii=False))

    history_rows = []

    with mlflow.start_run(run_name=config["experiment"]) as run:
        run_id = run.info.run_id

        safe_log_params(config)
        mlflow.log_param("model_family", model_family)
        mlflow.log_param("split_csv", str(split_csv))
        mlflow.log_param("n_trainval", len(train_df))
        mlflow.log_param("n_test", len(test_df))
        mlflow.log_param("num_classes", len(classes))
        mlflow.log_param("classes", json.dumps(classes, ensure_ascii=False))
        mlflow.log_param("num_trainable_params", num_trainable_params)

        mlflow.set_tag("part", "final_test")
        mlflow.set_tag("uses_test_set", "true")
        mlflow.set_tag("test_usage", "single_final_evaluation")

        for epoch in range(1, config["epochs"] + 1):
            train_loss, train_metrics = train_one_epoch_fn(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            row = {
                "experiment": config["experiment"],
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            }

            history_rows.append(row)

            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
                writer.add_scalar("macro_f1/train", train_metrics["macro_f1"], epoch)
                writer.add_scalar(
                    "balanced_accuracy/train",
                    train_metrics["balanced_accuracy"],
                    epoch,
                )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_metrics["accuracy"], step=epoch)
            mlflow.log_metric("train_macro_f1", train_metrics["macro_f1"], step=epoch)
            mlflow.log_metric(
                "train_balanced_accuracy",
                train_metrics["balanced_accuracy"],
                step=epoch,
            )

            print(
                f"[{config['experiment']}] "
                f"epoch {epoch:02d}/{config['epochs']} | "
                f"train_acc={train_metrics['accuracy']:.3f} | "
                f"train_f1={train_metrics['macro_f1']:.3f}"
            )

        # Única evaluación final sobre test.
        test_loss, test_metrics, y_true, y_pred = evaluate_fn(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
        mlflow.log_metric("test_macro_f1", test_metrics["macro_f1"])
        mlflow.log_metric("test_balanced_accuracy", test_metrics["balanced_accuracy"])

        if writer is not None:
            writer.add_scalar("loss/test_final", test_loss, config["epochs"])
            writer.add_scalar("accuracy/test_final", test_metrics["accuracy"], config["epochs"])
            writer.add_scalar("macro_f1/test_final", test_metrics["macro_f1"], config["epochs"])
            writer.close()

        predictions_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "true_class": [classes[i] for i in y_true],
            "pred_class": [classes[i] for i in y_pred],
        })

        summary = {
            "experiment": config["experiment"],
            "model_family": model_family,
            "model": config.get("model", None),
            "strategy": config.get("strategy", None),
            "epochs": config["epochs"],
            "image_size": config["image_size"],
            "batch_size": config["batch_size"],
            "lr": config["lr"],
            "optimizer": config["optimizer"],
            "dropout": config.get("dropout", None),
            "batch_norm": config.get("batch_norm", None),
            "weight_decay": config["weight_decay"],
            "augmentation": config["augmentation"],
            "initialization": config.get("initialization", None),
            "hidden_dims": config.get("hidden_dims", None),
            "n_trainval": len(train_df),
            "n_test": len(test_df),
            "num_trainable_params": num_trainable_params,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "mlflow_run_id": run_id,
        }

        output = {
            "config": config,
            "summary": summary,
            "history": pd.DataFrame(history_rows),
            "predictions": predictions_df,
            "y_true": y_true,
            "y_pred": y_pred,
            "classes": classes,
            "model_state_dict": model.detach_state_dict() if False else {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "mlflow_run_id": run_id,
        }

        output_prefix = config["experiment"].lower()
        artifact_paths = save_final_outputs(
            output=output,
            output_prefix=output_prefix,
        )

        # Logueamos artefactos y modelo en MLflow.
        for path in artifact_paths:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="outputs")

        if config.get("log_pytorch_model", True):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "final_model_state_dict.pth"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model_state_dict")

            mlflow.pytorch.log_model(
                pytorch_model=model.cpu(),
                artifact_path="final_pytorch_model",
            )

    print("Test final:")
    print(summary)

    return output