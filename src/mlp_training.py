# src/mlp_training.py
#
# Entrenamiento MLP para la Parte 1 del TP.
#
# Este archivo centraliza:
# 1. Carga del dataset desde data/splits/final_split_5fold.csv.
# 2. Definición del Dataset de PyTorch.
# 3. Definición del modelo MLP.
# 4. Entrenamiento con validación cruzada de 5 folds.
# 5. Regularización: Dropout, BatchNorm, Weight Decay, Data Augmentation.
# 6. Inicialización de pesos.
# 7. Logging en TensorBoard.
# 8. Logging en MLflow, generando la carpeta mlruns/.
# 9. Guardado de métricas, curvas, matriz de confusión y classification report.
#
# Importante:
# - El test fijo NO se usa para elegir hiperparámetros.
# - La comparación de modelos se hace solo sobre trainval usando folds.
# - MLflow se usa para dejar registro de hiperparámetros, métricas y artefactos.

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# TensorBoard es opcional, pero debería estar instalado.
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# MLflow es obligatorio para esta versión del TP.
# Lo dejamos con try/except para que el error sea claro si falta instalarlo.
try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    mlflow = None


# ============================================================
# Utilidades generales
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Fija semillas pseudoaleatorias para mejorar la reproducibilidad.

    Esto no garantiza determinismo perfecto en todos los sistemas, pero reduce
    la variabilidad entre corridas.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    """
    Crea una carpeta si no existe.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_path(path: str | Path) -> Path:
    """
    Convierte paths guardados con backslashes de Windows a paths compatibles
    con el sistema operativo actual.

    El CSV puede tener paths como:
        data\\Split_smol\\train\\...
    En Windows funciona, pero esto lo hace más robusto.
    """
    return Path(str(path).replace("\\", os.sep))


def get_device() -> torch.device:
    """
    Usa GPU si hay CUDA disponible. Si no, usa CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_mlflow(experiment_name: str = "TP_skin_MLP") -> None:
    """
    Configura MLflow para guardar los experimentos localmente en mlruns/.

    Esta función hace dos cosas:
    1. Define el tracking URI local.
    2. Define o crea el experimento de MLflow.

    Resultado esperado:
        mlruns/
            <experiment_id>/
                <run_id>/
                    params/
                    metrics/
                    artifacts/
                    meta.yaml
    """
    if mlflow is None:
        raise ImportError(
            "MLflow no está instalado en este entorno. "
            "Instalalo con: python -m pip install mlflow==2.22.0"
        )

    mlruns_path = Path("mlruns").resolve()
    mlruns_path.mkdir(parents=True, exist_ok=True)

    # MLflow necesita un URI. En local usamos file:///...
    mlflow.set_tracking_uri(mlruns_path.as_uri())

    # Si el experimento no existe, MLflow lo crea.
    mlflow.set_experiment(experiment_name)


def safe_log_params(params: Dict[str, Any]) -> None:
    """
    Loguea hiperparámetros en MLflow.

    MLflow acepta strings, ints, floats y bools.
    Para listas o diccionarios, los convertimos a JSON string.
    """
    for key, value in params.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value)
        mlflow.log_param(key, value)


# ============================================================
# Dataset
# ============================================================

class SkinDataset(Dataset):
    """
    Dataset de imágenes dermatológicas para PyTorch.

    Cada fila del dataframe debe tener al menos:
    - path: ruta a la imagen.
    - class: nombre de la clase.

    Devuelve:
    - image: tensor normalizado.
    - label: entero asociado a la clase.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: Dict[str, int],
        transform: Optional[Any] = None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = normalize_path(row["path"])
        label_name = row["class"]
        label = self.class_to_idx[label_name]

        # Convertimos siempre a RGB para que todas las imágenes tengan 3 canales.
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_class_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """
    Construye el diccionario clase -> índice.

    Se ordenan las clases alfabéticamente para que el mapping sea determinístico.
    """
    classes = sorted(df["class"].unique().tolist())
    return {class_name: idx for idx, class_name in enumerate(classes)}


def get_transforms(image_size: int = 64, augmentation: str = "minimal"):
    """
    Define transformaciones para entrenamiento y validación.

    Para MLP:
    - Todas las imágenes deben tener tamaño fijo.
    - Luego se aplanan dentro del modelo.
    - Usamos normalización tipo ImageNet porque es una práctica estándar
      y estabiliza el rango de entrada.

    augmentation:
    - minimal: sin aumento de datos.
    - light: aumento suave.
    - medium: aumento más marcado.
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
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    else:
        raise ValueError(f"augmentation no reconocida: {augmentation}")

    return train_transform, val_transform


# ============================================================
# Modelo MLP
# ============================================================

class MLPClassifier(nn.Module):
    """
    MLP para clasificación multiclase de imágenes.

    Flujo:
    1. Flatten: imagen 3 x H x W -> vector.
    2. Capas lineales.
    3. ReLU.
    4. Opcionalmente BatchNorm.
    5. Opcionalmente Dropout.
    6. Capa final con num_classes salidas.

    Nota:
    - La salida final NO tiene softmax.
    - CrossEntropyLoss aplica internamente log-softmax.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 128]

        layers: List[nn.Module] = []

        # La MLP no opera sobre estructura espacial.
        # Primero aplana la imagen.
        layers.append(nn.Flatten())

        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))

            # BatchNorm puede estabilizar la distribución de activaciones.
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU(inplace=True))

            # Dropout apaga neuronas durante entrenamiento para reducir overfitting.
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            previous_dim = hidden_dim

        # Capa final: una salida por clase.
        layers.append(nn.Linear(previous_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def initialize_weights(model: nn.Module, method: str = "default") -> None:
    """
    Inicializa pesos de las capas Linear.

    method:
    - default: inicialización estándar de PyTorch.
    - xavier: útil para mantener varianza estable en activaciones.
    - he: recomendada para redes con ReLU.
    - uniform: inicialización uniforme simple.
    """

    if method == "default":
        return

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)

            elif method == "he":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

            elif method == "uniform":
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)

            else:
                raise ValueError(f"Inicialización no reconocida: {method}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)


# ============================================================
# Métricas
# ============================================================

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calcula métricas principales.

    accuracy:
        proporción total de aciertos.

    macro_f1:
        F1 promedio por clase, sin ponderar por frecuencia.
        Es útil si hay desbalance.

    balanced_accuracy:
        recall promedio por clase.
        También es útil si hay desbalance.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


# ============================================================
# Entrenamiento y evaluación
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
):
    """
    Ejecuta una época de entrenamiento.

    En esta etapa:
    - model.train() activa Dropout y BatchNorm en modo entrenamiento.
    - Se hace forward.
    - Se calcula loss.
    - Se hace backpropagation.
    - Se actualizan pesos.
    """

    model.train()

    running_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
):
    """
    Evalúa el modelo en validación.

    En esta etapa:
    - model.eval() desactiva Dropout.
    - BatchNorm usa estadísticas acumuladas.
    - torch.no_grad() evita guardar gradientes y ahorra memoria.
    """

    model.eval()

    running_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

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


def make_dataloaders_for_fold(
    df: pd.DataFrame,
    fold: int,
    class_to_idx: Dict[str, int],
    config: Dict[str, Any],
):
    """
    Construye train_loader y val_loader para un fold.

    Se usa solo subset == trainval.

    train:
        imágenes con fold distinto del fold actual.

    val:
        imágenes con fold igual al fold actual.

    El test fijo queda excluido. Esto evita usar el test para elegir HP.
    """

    trainval_df = df[df["subset"] == "trainval"].copy()
    trainval_df["fold"] = trainval_df["fold"].astype(int)

    train_df = trainval_df[trainval_df["fold"] != fold].copy()
    val_df = trainval_df[trainval_df["fold"] == fold].copy()

    train_transform, val_transform = get_transforms(
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


def build_optimizer(model: nn.Module, config: Dict[str, Any]):
    """
    Crea el optimizador.

    Por defecto se usa Adam, como en la mayoría de los ensayos previos.
    También se deja soporte para AdamW si se quiere probar más adelante.
    """

    optimizer_name = config.get("optimizer", "adam").lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    raise ValueError(f"Optimizador no reconocido: {optimizer_name}")


def train_one_fold(
    df: pd.DataFrame,
    fold: int,
    class_to_idx: Dict[str, int],
    config: Dict[str, Any],
    device: torch.device,
):
    """
    Entrena un fold completo.

    Además de entrenar:
    - Registra curvas en TensorBoard.
    - Registra métricas por epoch en MLflow.
    - Guarda el mejor estado del modelo según val_accuracy.
    """

    set_seed(config.get("seed", 42) + fold)

    train_loader, val_loader, train_df, val_df = make_dataloaders_for_fold(
        df=df,
        fold=fold,
        class_to_idx=class_to_idx,
        config=config,
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    input_dim = 3 * config["image_size"] * config["image_size"]

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=len(class_to_idx),
        hidden_dims=config.get("hidden_dims", [512, 128]),
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    ).to(device)

    initialize_weights(model, method=config.get("initialization", "default"))

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    # TensorBoard: un directorio por experimento y fold.
    writer = None
    if config.get("tensorboard", True) and SummaryWriter is not None:
        log_dir = Path("runs") / "mlp" / config["experiment"] / f"fold_{fold}"
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

        # ---------------- TensorBoard ----------------
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar("accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("macro_f1/train", train_metrics["macro_f1"], epoch)
            writer.add_scalar("macro_f1/val", val_metrics["macro_f1"], epoch)
            writer.add_scalar(
                "balanced_accuracy/train",
                train_metrics["balanced_accuracy"],
                epoch,
            )
            writer.add_scalar(
                "balanced_accuracy/val",
                val_metrics["balanced_accuracy"],
                epoch,
            )

            # Histogramas: útiles para ver distribución de pesos.
            # No se guardan en cada epoch para evitar logs excesivamente pesados.
            if epoch == 1 or epoch % config.get("histogram_every", 5) == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(name, param.detach().cpu(), epoch)

        # ---------------- MLflow ----------------
        if config.get("mlflow", True) and mlflow is not None and mlflow.active_run() is not None:
            mlflow.log_metric(f"fold_{fold}_train_loss", train_loss, step=epoch)
            mlflow.log_metric(f"fold_{fold}_val_loss", val_loss, step=epoch)
            mlflow.log_metric(f"fold_{fold}_train_accuracy", train_metrics["accuracy"], step=epoch)
            mlflow.log_metric(f"fold_{fold}_val_accuracy", val_metrics["accuracy"], step=epoch)
            mlflow.log_metric(f"fold_{fold}_train_macro_f1", train_metrics["macro_f1"], step=epoch)
            mlflow.log_metric(f"fold_{fold}_val_macro_f1", val_metrics["macro_f1"], step=epoch)
            mlflow.log_metric(
                f"fold_{fold}_train_balanced_accuracy",
                train_metrics["balanced_accuracy"],
                step=epoch,
            )
            mlflow.log_metric(
                f"fold_{fold}_val_balanced_accuracy",
                val_metrics["balanced_accuracy"],
                step=epoch,
            )

        print(
            f"[{config['experiment']}] fold {fold} | "
            f"epoch {epoch:02d}/{config['epochs']} | "
            f"train_acc={train_metrics['accuracy']:.3f} | "
            f"val_acc={val_metrics['accuracy']:.3f} | "
            f"val_f1={val_metrics['macro_f1']:.3f}"
        )

        # Selección del mejor modelo del fold según accuracy de validación.
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping opcional.
        if config.get("early_stopping", False):
            patience = config.get("patience", 5)
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping en fold {fold}. "
                    f"Mejor epoch: {best_epoch}, mejor val_acc: {best_val_accuracy:.3f}"
                )
                break

    # Restauramos el mejor estado del fold.
    model.load_state_dict(best_state)

    # Recalculamos métricas finales del mejor modelo.
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

    # Métricas finales del fold en MLflow.
    if config.get("mlflow", True) and mlflow is not None and mlflow.active_run() is not None:
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
        "idx_to_class": idx_to_class,
    }


def build_summary(
    config: Dict[str, Any],
    fold_results_df: pd.DataFrame,
    folds_to_run: List[int],
    best_fold_id: Optional[int],
    mlflow_run_id: Optional[str],
) -> Dict[str, Any]:
    """
    Construye el resumen final de una corrida de validación cruzada.
    """

    return {
        "experiment": config["experiment"],
        "model": config.get("model", "MLPClassifier"),
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "lr": config["lr"],
        "optimizer": config.get("optimizer", "adam"),
        "dropout": config["dropout"],
        "batch_norm": config["batch_norm"],
        "weight_decay": config["weight_decay"],
        "augmentation": config["augmentation"],
        "initialization": config.get("initialization", "default"),
        "hidden_dims": config.get("hidden_dims", [512, 128]),
        "early_stopping": config.get("early_stopping", False),
        "folds_run": folds_to_run,
        "best_fold": best_fold_id,
        "val_accuracy_mean": float(fold_results_df["val_accuracy"].mean()),
        "val_accuracy_std": float(fold_results_df["val_accuracy"].std(ddof=0)),
        "val_macro_f1_mean": float(fold_results_df["val_macro_f1"].mean()),
        "val_macro_f1_std": float(fold_results_df["val_macro_f1"].std(ddof=0)),
        "val_balanced_accuracy_mean": float(fold_results_df["val_balanced_accuracy"].mean()),
        "val_balanced_accuracy_std": float(fold_results_df["val_balanced_accuracy"].std(ddof=0)),
        "mlflow_run_id": mlflow_run_id,
    }


def run_cross_validation(
    config: Dict[str, Any],
    split_csv: str | Path = "data/splits/final_split_5fold.csv",
    folds_to_run: Optional[List[int]] = None,
):
    """
    Corre validación cruzada sobre trainval.

    Esta función crea un run de MLflow por experimento completo.
    Dentro del run se registran:
    - hiperparámetros;
    - métricas por epoch y fold;
    - métricas promedio de validación cruzada;
    - mejor modelo encontrado entre los folds.

    El test fijo no se usa.
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

    use_mlflow = config.get("mlflow", True)

    if use_mlflow:
        setup_mlflow(config.get("mlflow_experiment", "TP_skin_MLP"))

    all_fold_results = []
    all_history = []
    all_y_true = []
    all_y_pred = []

    best_fold_accuracy = -np.inf
    best_fold_model = None
    best_fold_id = None

    mlflow_run_id = None

    def training_loop():
        nonlocal best_fold_accuracy
        nonlocal best_fold_model
        nonlocal best_fold_id

        for fold in folds_to_run:
            output = train_one_fold(
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

    if use_mlflow:
        with mlflow.start_run(run_name=config["experiment"]) as run:
            mlflow_run_id = run.info.run_id

            # Hiperparámetros principales.
            safe_log_params(config)

            # Metadata adicional.
            mlflow.log_param("split_csv", str(split_csv))
            mlflow.log_param("folds_to_run", json.dumps(folds_to_run))
            mlflow.log_param("num_classes", len(class_to_idx))
            mlflow.log_param(
                "classes",
                json.dumps(
                    [idx_to_class[i] for i in range(len(idx_to_class))],
                    ensure_ascii=False,
                ),
            )

            # Tags para ordenar la interfaz de MLflow.
            mlflow.set_tag("part", "part1_mlp")
            mlflow.set_tag("model_family", "MLP")
            mlflow.set_tag("uses_test_set", "false")

            training_loop()

            fold_results_df = pd.DataFrame(all_fold_results)

            summary = build_summary(
                config=config,
                fold_results_df=fold_results_df,
                folds_to_run=folds_to_run,
                best_fold_id=best_fold_id,
                mlflow_run_id=mlflow_run_id,
            )

            # Métricas agregadas de validación cruzada.
            mlflow.log_metric("cv_val_accuracy_mean", summary["val_accuracy_mean"])
            mlflow.log_metric("cv_val_accuracy_std", summary["val_accuracy_std"])
            mlflow.log_metric("cv_val_macro_f1_mean", summary["val_macro_f1_mean"])
            mlflow.log_metric("cv_val_macro_f1_std", summary["val_macro_f1_std"])
            mlflow.log_metric(
                "cv_val_balanced_accuracy_mean",
                summary["val_balanced_accuracy_mean"],
            )
            mlflow.log_metric(
                "cv_val_balanced_accuracy_std",
                summary["val_balanced_accuracy_std"],
            )

            # Guardamos el mejor modelo como artifact de MLflow.
            # Esto es lo que hace que mlruns pueda pesar bastante.
            if best_fold_model is not None and config.get("log_pytorch_model", True):
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = Path(tmpdir) / "best_model_state_dict.pth"
                    torch.save(best_fold_model.state_dict(), model_path)
                    mlflow.log_artifact(str(model_path), artifact_path="model_state_dict")

                mlflow.pytorch.log_model(
                    pytorch_model=best_fold_model,
                    artifact_path="best_pytorch_model",
                )

    else:
        training_loop()

        fold_results_df = pd.DataFrame(all_fold_results)

        summary = build_summary(
            config=config,
            fold_results_df=fold_results_df,
            folds_to_run=folds_to_run,
            best_fold_id=best_fold_id,
            mlflow_run_id=None,
        )

    return {
        "config": config,
        "summary": summary,
        "fold_results": pd.DataFrame(all_fold_results),
        "history": pd.DataFrame(all_history),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "classes": [idx_to_class[i] for i in range(len(idx_to_class))],
        "mlflow_run_id": mlflow_run_id,
    }


# ============================================================
# Guardado de artifacts locales y MLflow
# ============================================================

def plot_metric(
    history_df: pd.DataFrame,
    metric_train: str,
    metric_val: str,
    title: str,
    output_path: Path,
):
    """
    Grafica una métrica promedio por epoch.

    Se promedia sobre folds para obtener una curva general del experimento.
    """
    grouped = history_df.groupby("epoch")[[metric_train, metric_val]].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["epoch"], grouped[metric_train], label=metric_train)
    plt.plot(grouped["epoch"], grouped[metric_val], label=metric_val)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    output_path: Path,
):
    """
    Guarda la matriz de confusión como imagen PNG.
    """
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion matrix")
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


def log_outputs_to_mlflow(
    cv_output: Dict[str, Any],
    output_paths: List[Path],
) -> None:
    """
    Agrega artifacts ya guardados localmente al run de MLflow.

    save_experiment_outputs() se ejecuta después de run_cross_validation().
    Por eso reabrimos el run usando su run_id.
    """
    run_id = cv_output.get("mlflow_run_id")

    if run_id is None:
        return

    if mlflow is None:
        return

    if not cv_output["config"].get("mlflow", True):
        return

    setup_mlflow(cv_output["config"].get("mlflow_experiment", "TP_skin_MLP"))

    with mlflow.start_run(run_id=run_id):
        for path in output_paths:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="outputs")


def save_experiment_outputs(cv_output: Dict[str, Any], output_prefix: str):
    """
    Guarda los resultados del experimento en carpetas locales y en MLflow.

    Salidas locales:
    - experiments/*_fold_results.csv
    - experiments/*_history.csv
    - experiments/*_summary.json
    - results/training_curves/*.png
    - results/confusion_matrices/*.png
    - results/classification_reports/*.txt

    Además, esos mismos archivos se agregan como artifacts del run MLflow.
    """

    ensure_dir("experiments")
    ensure_dir("results/training_curves")
    ensure_dir("results/confusion_matrices")
    ensure_dir("results/classification_reports")

    prefix = output_prefix

    fold_results_path = Path("experiments") / f"{prefix}_fold_results.csv"
    history_path = Path("experiments") / f"{prefix}_history.csv"
    summary_path = Path("experiments") / f"{prefix}_summary.json"

    cv_output["fold_results"].to_csv(fold_results_path, index=False)
    cv_output["history"].to_csv(history_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(cv_output["summary"], f, indent=2, ensure_ascii=False)

    loss_plot_path = Path("results/training_curves") / f"{prefix}_loss.png"
    accuracy_plot_path = Path("results/training_curves") / f"{prefix}_accuracy.png"
    cm_path = Path("results/confusion_matrices") / f"{prefix}.png"
    report_path = Path("results/classification_reports") / f"{prefix}_classification_report.txt"

    plot_metric(
        history_df=cv_output["history"],
        metric_train="train_loss",
        metric_val="val_loss",
        title="Loss",
        output_path=loss_plot_path,
    )

    plot_metric(
        history_df=cv_output["history"],
        metric_train="train_accuracy",
        metric_val="val_accuracy",
        title="Accuracy",
        output_path=accuracy_plot_path,
    )

    save_confusion_matrix(
        y_true=cv_output["y_true"],
        y_pred=cv_output["y_pred"],
        classes=cv_output["classes"],
        output_path=cm_path,
    )

    labels = list(range(len(cv_output["classes"])))

    report = classification_report(
        cv_output["y_true"],
        cv_output["y_pred"],
        labels=labels,
        target_names=cv_output["classes"],
        zero_division=0,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    output_paths = [
        fold_results_path,
        history_path,
        summary_path,
        loss_plot_path,
        accuracy_plot_path,
        cm_path,
        report_path,
    ]

    # Copia los archivos anteriores al run correspondiente de MLflow.
    log_outputs_to_mlflow(cv_output=cv_output, output_paths=output_paths)

    print("Archivos guardados:")
    for path in output_paths:
        print(f"- {path}")