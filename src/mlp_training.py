# ============================================================
# MLP TRAINING UTILITIES
# Skin Dataset Classification - Parte 1 del TP
# ============================================================
#
# Este módulo contiene todo lo necesario para entrenar modelos MLP:
#
# 1. Lectura del split final:
#    - test fijo excluido
#    - trainval con 5 folds estratificados
#
# 2. Dataset de PyTorch.
#
# 3. Modelo MLP configurable:
#    - Dropout
#    - Batch Normalization
#    - Weight Decay
#    - Data Augmentation
#    - inicialización: default, Xavier, He/Kaiming, uniforme
#    - Early Stopping
#
# 4. Logging:
#    - CSV con métricas
#    - curvas loss/accuracy
#    - matriz de confusión
#    - TensorBoard
#    - histogramas de pesos
#
# ============================================================


# ============================================================
# 0. Imports
# ============================================================

from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


# ============================================================
# 1. Reproducibilidad
# ============================================================
# Fijamos semillas para que los resultados sean más repetibles.
# No garantiza igualdad absoluta en todos los sistemas, pero reduce
# la variabilidad entre corridas.
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Dataset
# ============================================================
# Este Dataset usa el CSV final_split_5fold.csv.
#
# Cada fila del CSV contiene:
# - path: ruta de la imagen
# - class: etiqueta textual
# - subset: trainval o test
# - fold: fold asignado dentro de trainval
#
# Para Parte G solo se usa subset == "trainval".
# ============================================================

class SkinDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, class_to_idx: dict, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # La ruta ya viene relativa al root del repo, por ejemplo:
        # data/Split_smol/train/clase/imagen.jpg
        img_path = Path(row["path"])

        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["class"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# 3. Transformaciones y Data Augmentation
# ============================================================
# Para el MLP se redimensiona a 64x64.
#
# minimal:
#   Resize + ToTensor + Normalize.
#
# medium:
#   Transformaciones suaves:
#   - flip horizontal
#   - rotación leve
#   - cambios leves de brillo/contraste/saturación
#
# Esto cumple la actividad de modificación relacionada con
# data augmentation.
# ============================================================

def get_transforms(image_size: int = 64, augmentation: str = "minimal"):
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if augmentation == "minimal":
        train_transform = val_transform

    elif augmentation == "medium":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),

            # Aumento geométrico leve.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),

            # Aumento fotométrico leve.
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    else:
        raise ValueError(f"Augmentation desconocida: {augmentation}")

    return train_transform, val_transform


# ============================================================
# 4. Modelo MLP
# ============================================================
# El MLP aplana la imagen.
#
# Imagen 64x64 RGB:
#   64 * 64 * 3 = 12288 entradas.
#
# Arquitectura:
#   Flatten
#   Linear(12288 -> 512)
#   ReLU
#   Dropout opcional
#   Linear(512 -> 128)
#   ReLU
#   Dropout opcional
#   Linear(128 -> num_classes)
#
# BatchNorm se puede activar después de las capas lineales.
# Dropout se puede activar después de ReLU.
# ============================================================

class MLPClassifier(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        input_dim = image_size * image_size * 3

        layers = []

        # ----------------------------
        # Bloque inicial: aplanado
        # ----------------------------
        layers.append(nn.Flatten())

        # ----------------------------
        # Bloque fully-connected 1
        # ----------------------------
        layers.append(nn.Linear(input_dim, 512))

        if batch_norm:
            layers.append(nn.BatchNorm1d(512))

        layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # ----------------------------
        # Bloque fully-connected 2
        # ----------------------------
        layers.append(nn.Linear(512, 128))

        if batch_norm:
            layers.append(nn.BatchNorm1d(128))

        layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # ----------------------------
        # Capa de clasificación final
        # ----------------------------
        layers.append(nn.Linear(128, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================
# 5. Inicialización de pesos
# ============================================================
# Preguntas.md pide comparar inicializaciones.
#
# Métodos implementados:
#
# default:
#   Inicialización default de PyTorch.
#
# xavier:
#   Xavier/Glorot uniforme.
#
# he:
#   He/Kaiming normal, adecuada para ReLU.
#
# uniform:
#   Inicialización uniforme manual en [-0.1, 0.1].
# ============================================================

def initialize_weights(model: nn.Module, method: str = "default"):
    if method == "default":
        return model

    for module in model.modules():
        if isinstance(module, nn.Linear):

            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)

            elif method == "he":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

            elif method == "uniform":
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)

            else:
                raise ValueError(f"Inicialización desconocida: {method}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return model


# ============================================================
# 6. TensorBoard: histogramas de pesos
# ============================================================
# Preguntas.md pide visualizar pesos.
#
# Esta función manda a TensorBoard histogramas de:
# - pesos
# - biases
#
# Se llama durante el entrenamiento cada ciertas epochs.
# ============================================================

def log_weight_histograms(writer: SummaryWriter, model: nn.Module, epoch: int, prefix: str):
    if writer is None:
        return

    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(
                tag=f"{prefix}/{name}",
                values=param.detach().cpu(),
                global_step=epoch,
            )


# ============================================================
# 7. Métricas
# ============================================================
# Se usan métricas adecuadas para clasificación multiclase:
#
# accuracy:
#   proporción total de aciertos.
#
# macro_f1:
#   promedio del F1 por clase, pesando todas las clases igual.
#
# balanced_accuracy:
#   útil si hay desbalance de clases.
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


# ============================================================
# 8. Evaluación
# ============================================================
# Evalúa el modelo sin actualizar pesos.
#
# Se usa tanto para validación por epoch como para la evaluación
# final del mejor modelo de cada fold.
# ============================================================

def evaluate_model(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)


# ============================================================
# 9. Entrenamiento de un fold
# ============================================================
# En cada fold:
#
# - validación = imágenes de trainval con fold == k
# - entrenamiento = imágenes de trainval con fold != k
#
# El test fijo nunca entra acá.
#
# Esta función implementa:
# - entrenamiento epoch por epoch
# - regularización por Dropout/BatchNorm/Weight Decay
# - Early Stopping opcional
# - logging en TensorBoard
# - guardado del mejor estado según val_accuracy
# ============================================================

def train_one_fold(
    df: pd.DataFrame,
    fold: int,
    class_to_idx: dict,
    config: dict,
    device,
    writer: SummaryWriter = None,
):
    # --------------------------------------------------------
    # 9.1. Separar train y validación del fold
    # --------------------------------------------------------
    train_df = df[
        (df["subset"] == "trainval") &
        (df["fold"].astype(int) != int(fold))
    ].copy()

    val_df = df[
        (df["subset"] == "trainval") &
        (df["fold"].astype(int) == int(fold))
    ].copy()

    # --------------------------------------------------------
    # 9.2. Transformaciones
    # --------------------------------------------------------
    train_transform, val_transform = get_transforms(
        image_size=config["image_size"],
        augmentation=config["augmentation"],
    )

    # --------------------------------------------------------
    # 9.3. Datasets y DataLoaders
    # --------------------------------------------------------
    train_dataset = SkinDataset(
        dataframe=train_df,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )

    val_dataset = SkinDataset(
        dataframe=val_df,
        class_to_idx=class_to_idx,
        transform=val_transform,
    )

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

    # --------------------------------------------------------
    # 9.4. Crear modelo
    # --------------------------------------------------------
    model = MLPClassifier(
        image_size=config["image_size"],
        num_classes=len(class_to_idx),
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    ).to(device)

    # --------------------------------------------------------
    # 9.5. Inicialización manual si corresponde
    # --------------------------------------------------------
    model = initialize_weights(
        model=model,
        method=config["initialization"],
    )

    # Log inicial de pesos antes de entrenar.
    log_weight_histograms(
        writer=writer,
        model=model,
        epoch=0,
        prefix=f"{config['experiment']}/fold_{fold}/weights_initial",
    )

    # --------------------------------------------------------
    # 9.6. Loss y optimizador
    # --------------------------------------------------------
    # CrossEntropyLoss es la pérdida estándar para clasificación
    # multiclase con logits.
    #
    # Adam usa weight_decay si config["weight_decay"] > 0.
    # Eso implementa regularización L2.
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # --------------------------------------------------------
    # 9.7. Variables de control
    # --------------------------------------------------------
    best_val_accuracy = -1.0
    best_epoch = -1
    best_state = None

    history = []

    patience_counter = 0
    early_stopping_enabled = config.get("early_stopping", False)
    patience = config.get("patience", 5)

    # --------------------------------------------------------
    # 9.8. Loop de entrenamiento
    # --------------------------------------------------------
    for epoch in range(1, config["epochs"] + 1):

        # ----------------------------
        # Modo entrenamiento
        # ----------------------------
        model.train()

        train_loss_total = 0.0
        train_preds = []
        train_labels = []

        # ----------------------------
        # Iterar por batches
        # ----------------------------
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Resetear gradientes acumulados.
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(images)

            # Calcular loss.
            loss = criterion(outputs, labels)

            # Backpropagation.
            loss.backward()

            # Actualizar pesos.
            optimizer.step()

            # Guardar métricas del batch.
            train_loss_total += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        # ----------------------------
        # Métricas de entrenamiento
        # ----------------------------
        train_loss = train_loss_total / len(train_loader.dataset)
        train_metrics = compute_metrics(train_labels, train_preds)

        # ----------------------------
        # Validación
        # ----------------------------
        val_loss, val_metrics, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ----------------------------
        # Guardar historial de la epoch
        # ----------------------------
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

        history.append(row)

        # ----------------------------
        # TensorBoard: scalars
        # ----------------------------
        if writer is not None:
            writer.add_scalar(
                f"{config['experiment']}/fold_{fold}/loss/train",
                train_loss,
                epoch,
            )
            writer.add_scalar(
                f"{config['experiment']}/fold_{fold}/loss/val",
                val_loss,
                epoch,
            )
            writer.add_scalar(
                f"{config['experiment']}/fold_{fold}/accuracy/train",
                train_metrics["accuracy"],
                epoch,
            )
            writer.add_scalar(
                f"{config['experiment']}/fold_{fold}/accuracy/val",
                val_metrics["accuracy"],
                epoch,
            )
            writer.add_scalar(
                f"{config['experiment']}/fold_{fold}/macro_f1/val",
                val_metrics["macro_f1"],
                epoch,
            )

        # ----------------------------
        # TensorBoard: histogramas de pesos
        # ----------------------------
        histogram_every = config.get("histogram_every", 5)

        if epoch % histogram_every == 0 or epoch == config["epochs"]:
            log_weight_histograms(
                writer=writer,
                model=model,
                epoch=epoch,
                prefix=f"{config['experiment']}/fold_{fold}/weights",
            )

        # ----------------------------
        # Guardar mejor modelo del fold
        # ----------------------------
        improved = val_metrics["accuracy"] > best_val_accuracy

        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # ----------------------------
        # Early Stopping opcional
        # ----------------------------
        if early_stopping_enabled and patience_counter >= patience:
            print(
                f"Early stopping en fold {fold}, epoch {epoch}. "
                f"Mejor epoch: {best_epoch}"
            )
            break

    # --------------------------------------------------------
    # 9.9. Cargar mejor modelo del fold
    # --------------------------------------------------------
    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # 9.10. Evaluación final del fold con el mejor modelo
    # --------------------------------------------------------
    val_loss, val_metrics, y_true, y_pred = evaluate_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )

    # --------------------------------------------------------
    # 9.11. Resultado final del fold
    # --------------------------------------------------------
    result = {
        "experiment": config["experiment"],
        "fold": fold,
        "best_epoch": best_epoch,

        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "lr": config["lr"],

        "dropout": config["dropout"],
        "batch_norm": config["batch_norm"],
        "weight_decay": config["weight_decay"],
        "augmentation": config["augmentation"],
        "initialization": config["initialization"],
        "early_stopping": config.get("early_stopping", False),

        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
    }

    return result, pd.DataFrame(history), y_true, y_pred


# ============================================================
# 10. Cross-validation
# ============================================================
# Ejecuta una configuración en todos los folds.
#
# Ejemplo:
# - fold 0: valida con fold 0, entrena con 1-4
# - fold 1: valida con fold 1, entrena con 0,2,3,4
# - etc.
#
# El resultado resume la performance promedio del modelo.
# ============================================================

def run_cross_validation(config: dict, split_csv="data/splits/final_split_5fold.csv"):
    set_seed(config.get("seed", 42))

    # --------------------------------------------------------
    # 10.1. Cargar split final
    # --------------------------------------------------------
    df = pd.read_csv(split_csv)

    # Se excluye test fijo en Parte G.
    trainval_df = df[df["subset"] == "trainval"].copy()

    # --------------------------------------------------------
    # 10.2. Codificación de clases
    # --------------------------------------------------------
    classes = sorted(trainval_df["class"].unique())
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    # --------------------------------------------------------
    # 10.3. Lista de folds
    # --------------------------------------------------------
    folds = sorted(trainval_df["fold"].astype(int).unique())

    # --------------------------------------------------------
    # 10.4. Dispositivo
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Experimento: {config['experiment']}")
    print(f"Dispositivo: {device}")
    print(f"Folds usados: {folds}")
    print(f"Clases: {classes}")

    # --------------------------------------------------------
    # 10.5. TensorBoard writer
    # --------------------------------------------------------
    writer = None

    if config.get("tensorboard", True):
        log_dir = Path("runs") / "mlp" / config["experiment"]
        writer = SummaryWriter(log_dir=str(log_dir))

        writer.add_text(
            tag=f"{config['experiment']}/config",
            text_string=json.dumps(config, indent=4),
            global_step=0,
        )

    # --------------------------------------------------------
    # 10.6. Ejecutar folds
    # --------------------------------------------------------
    fold_results = []
    histories = {}
    all_y_true = []
    all_y_pred = []

    for fold in folds:
        print("\n" + "=" * 70)
        print(f"Entrenando fold {fold}")
        print("=" * 70)

        result, history, y_true, y_pred = train_one_fold(
            df=df,
            fold=fold,
            class_to_idx=class_to_idx,
            config=config,
            device=device,
            writer=writer,
        )

        fold_results.append(result)
        histories[fold] = history

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(
            f"Fold {fold} terminado | "
            f"val_acc={result['val_accuracy']:.4f} | "
            f"macro_f1={result['val_macro_f1']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)

    # --------------------------------------------------------
    # 10.7. Resumen promedio de cross-validation
    # --------------------------------------------------------
    summary = {
        "experiment": config["experiment"],

        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "lr": config["lr"],

        "dropout": config["dropout"],
        "batch_norm": config["batch_norm"],
        "weight_decay": config["weight_decay"],
        "augmentation": config["augmentation"],
        "initialization": config["initialization"],
        "early_stopping": config.get("early_stopping", False),

        "val_accuracy_mean": results_df["val_accuracy"].mean(),
        "val_accuracy_std": results_df["val_accuracy"].std(),

        "val_macro_f1_mean": results_df["val_macro_f1"].mean(),
        "val_macro_f1_std": results_df["val_macro_f1"].std(),

        "val_balanced_accuracy_mean": results_df["val_balanced_accuracy"].mean(),
        "val_balanced_accuracy_std": results_df["val_balanced_accuracy"].std(),
    }

    # --------------------------------------------------------
    # 10.8. TensorBoard: métricas resumen
    # --------------------------------------------------------
    if writer is not None:
        writer.add_scalar(
            f"{config['experiment']}/cv_summary/val_accuracy_mean",
            summary["val_accuracy_mean"],
            0,
        )
        writer.add_scalar(
            f"{config['experiment']}/cv_summary/val_macro_f1_mean",
            summary["val_macro_f1_mean"],
            0,
        )
        writer.flush()
        writer.close()

    return {
        "config": config,
        "results_df": results_df,
        "summary": summary,
        "histories": histories,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }


# ============================================================
# 11. Guardado de resultados
# ============================================================
# Guarda los resultados de un experimento:
#
# experiments/
#   *_fold_results.csv
#   *_history.csv
#   *_summary.json
#   *_classification_report.txt
#
# results/training_curves/
#   *_loss.png
#   *_accuracy.png
#
# results/confusion_matrices/
#   *.png
# ============================================================

def save_experiment_outputs(cv_output, output_prefix: str):
    output_prefix = str(output_prefix)

    experiments_dir = Path("experiments")
    curves_dir = Path("results/training_curves")
    cm_dir = Path("results/confusion_matrices")
    reports_dir = Path("results/classification_reports")

    experiments_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = cv_output["config"]["experiment"]

    # --------------------------------------------------------
    # 11.1. CSV con resultados por fold
    # --------------------------------------------------------
    fold_csv = experiments_dir / f"{output_prefix}_fold_results.csv"
    cv_output["results_df"].to_csv(fold_csv, index=False)

    # --------------------------------------------------------
    # 11.2. CSV con historial epoch por epoch
    # --------------------------------------------------------
    history_df = pd.concat(
        [
            history.assign(fold=fold)
            for fold, history in cv_output["histories"].items()
        ],
        ignore_index=True,
    )

    history_csv = experiments_dir / f"{output_prefix}_history.csv"
    history_df.to_csv(history_csv, index=False)

    # --------------------------------------------------------
    # 11.3. JSON con resumen del experimento
    # --------------------------------------------------------
    summary_json = experiments_dir / f"{output_prefix}_summary.json"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(cv_output["summary"], f, indent=4)

    # --------------------------------------------------------
    # 11.4. Curvas promedio de loss
    # --------------------------------------------------------
    mean_history = history_df.groupby("epoch").mean(numeric_only=True)

    plt.figure(figsize=(8, 5))
    plt.plot(mean_history.index, mean_history["train_loss"], label="train_loss")
    plt.plot(mean_history.index, mean_history["val_loss"], label="val_loss")
    plt.title(f"Loss promedio por epoch - {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / f"{output_prefix}_loss.png", dpi=300)
    plt.show()

    # --------------------------------------------------------
    # 11.5. Curvas promedio de accuracy
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(mean_history.index, mean_history["train_accuracy"], label="train_accuracy")
    plt.plot(mean_history.index, mean_history["val_accuracy"], label="val_accuracy")
    plt.title(f"Accuracy promedio por epoch - {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / f"{output_prefix}_accuracy.png", dpi=300)
    plt.show()

    # --------------------------------------------------------
    # 11.6. Matriz de confusión agregada
    # --------------------------------------------------------
    cm = confusion_matrix(cv_output["y_true"], cv_output["y_pred"])

    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=cv_output["classes"],
    )

    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(f"Matriz de confusión agregada - {experiment_name}")
    plt.tight_layout()
    plt.savefig(cm_dir / f"{output_prefix}.png", dpi=300)
    plt.show()

    # --------------------------------------------------------
    # 11.7. Classification report agregado
    # --------------------------------------------------------
    report_txt = classification_report(
        cv_output["y_true"],
        cv_output["y_pred"],
        target_names=cv_output["classes"],
        zero_division=0,
    )

    report_path = reports_dir / f"{output_prefix}_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    # --------------------------------------------------------
    # 11.8. Mensajes finales
    # --------------------------------------------------------
    print(f"Guardado: {fold_csv}")
    print(f"Guardado: {history_csv}")
    print(f"Guardado: {summary_json}")
    print(f"Guardado: {report_path}")