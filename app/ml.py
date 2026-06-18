from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path
from random import Random
from typing import Any
from zoneinfo import ZoneInfo
import math

import torch
from torch import nn
import torch.nn.functional as F

from app.models import PopulationObservation, TrainingConfig


SEOUL_TZ = ZoneInfo("Asia/Seoul")
CONGESTION_LABELS = ["여유", "보통", "약간 붐빔", "붐빔"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(CONGESTION_LABELS)}


class CrowdNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = len(CONGESTION_LABELS)) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def train_crowd_model(
    observations: list[PopulationObservation],
    model_path: Path,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TrainingConfig()
    _validate_training_config(cfg)
    examples = _usable_observations(observations)
    if len(examples) < 3:
        raise ValueError("At least 3 labeled observations are required for train/validation/test split")

    area_names = sorted({item.area_name for item in examples})
    train_rows, validation_rows, test_rows = split_observations(
        examples,
        train_ratio=cfg.train_ratio,
        validation_ratio=cfg.validation_ratio,
        seed=cfg.seed,
    )

    torch.manual_seed(cfg.seed)
    device = _select_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    input_dim = len(area_names) + 7
    model = CrowdNet(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
        min_lr=cfg.min_learning_rate,
    )

    train_x, train_y = _tensor_dataset(train_rows, area_names, device=device)
    validation_x, validation_y = _tensor_dataset(validation_rows, area_names, device=device)
    test_x, test_y = _tensor_dataset(test_rows, area_names, device=device)

    history = []
    best_state_dict = _state_dict_on_cpu(model)
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    epochs_ran = 0
    stopped_reason = "max_epochs"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()

        epochs_ran = epoch
        validation_metrics = evaluate_model(model, validation_x, validation_y)
        scheduler.step(validation_metrics["loss"])

        if validation_metrics["loss"] < best_validation_loss - cfg.early_stopping_min_delta:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state_dict = _state_dict_on_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch == cfg.epochs
            or epoch % max(1, cfg.epochs // 10) == 0
            or epochs_without_improvement == 0
        ):
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": _current_learning_rate(optimizer),
                    "train": evaluate_model(model, train_x, train_y),
                    "validation": validation_metrics,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            stopped_reason = "early_stopping"
            break

    model.load_state_dict(best_state_dict)
    train_metrics = evaluate_model(model, train_x, train_y)
    validation_metrics = evaluate_model(model, validation_x, validation_y)
    test_metrics = evaluate_model(model, test_x, test_y)

    artifact = {
        "state_dict": _state_dict_on_cpu(model),
        "area_names": area_names,
        "input_dim": input_dim,
        "labels": CONGESTION_LABELS,
        "trained_at": datetime.now(tz=SEOUL_TZ).isoformat(),
        "device": str(device),
        "config": asdict(cfg),
        "split_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "history": history,
        "training_summary": {
            "epochs_requested": cfg.epochs,
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "early_stopped": stopped_reason == "early_stopping",
            "stopped_reason": stopped_reason,
            "best_validation_loss": best_validation_loss,
            "final_learning_rate": _current_learning_rate(optimizer),
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_path)

    return {
        "model_path": str(model_path),
        "trained_at": artifact["trained_at"],
        "device": artifact["device"],
        "labels": CONGESTION_LABELS,
        "area_names": area_names,
        "split_counts": artifact["split_counts"],
        "metrics": artifact["metrics"],
        "history": history,
        "training_summary": artifact["training_summary"],
    }


def predict_crowd(
    model_path: Path,
    area_name: str,
    target_date: date,
    target_time: time,
) -> dict[str, Any]:
    artifact = load_artifact(model_path)
    area_names = artifact["area_names"]
    if area_name not in area_names:
        raise ValueError(f"Area '{area_name}' was not included in the trained model")

    target_dt = datetime.combine(target_date, target_time.replace(tzinfo=None), tzinfo=SEOUL_TZ)
    if target_dt <= datetime.now(tz=SEOUL_TZ):
        raise ValueError("target_date and target_time must be in the future in Asia/Seoul")

    device = _select_device()
    model = CrowdNet(input_dim=int(artifact["input_dim"])).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    feature = torch.tensor(
        [build_features(area_name, target_dt, area_names)],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits = model(feature)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()
        predicted_index = int(torch.argmax(probabilities).item())

    return {
        "area_name": area_name,
        "target_datetime": target_dt.isoformat(),
        "predicted_congestion_level": CONGESTION_LABELS[predicted_index],
        "confidence": round(float(probabilities[predicted_index].item()), 4),
        "probabilities": {
            label: round(float(probabilities[index].item()), 4)
            for index, label in enumerate(CONGESTION_LABELS)
        },
        "model_trained_at": artifact["trained_at"],
        "device": str(device),
        "metrics": artifact["metrics"],
    }


def load_artifact(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return torch.load(model_path, map_location="cpu", weights_only=False)


def split_observations(
    observations: list[PopulationObservation],
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[list[PopulationObservation], list[PopulationObservation], list[PopulationObservation]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    rows = list(observations)
    Random(seed).shuffle(rows)
    n = len(rows)
    train_count = max(1, int(n * train_ratio))
    validation_count = max(1, int(n * validation_ratio))
    if train_count + validation_count >= n:
        train_count = n - 2
        validation_count = 1

    train_rows = rows[:train_count]
    validation_rows = rows[train_count : train_count + validation_count]
    test_rows = rows[train_count + validation_count :]
    return train_rows, validation_rows, test_rows


def evaluate_model(model: CrowdNet, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = F.cross_entropy(logits, labels).item()
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean().item()
    return {"loss": round(loss, 6), "accuracy": round(accuracy, 6)}


def build_features(area_name: str, observed_at: datetime, area_names: list[str]) -> list[float]:
    local_time = observed_at.astimezone(SEOUL_TZ)
    area_features = [1.0 if area_name == candidate else 0.0 for candidate in area_names]
    hour = local_time.hour + local_time.minute / 60
    weekday = local_time.weekday()
    month = local_time.month
    return area_features + [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * weekday / 7),
        math.cos(2 * math.pi * weekday / 7),
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        1.0 if weekday >= 5 else 0.0,
    ]


def _tensor_dataset(
    rows: list[PopulationObservation],
    area_names: list[str],
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [build_features(row.area_name, row.observed_at, area_names) for row in rows]
    labels = [LABEL_TO_INDEX[row.congestion_level] for row in rows if row.congestion_level in LABEL_TO_INDEX]
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _state_dict_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return round(float(optimizer.param_groups[0]["lr"]), 10)


def _validate_training_config(config: TrainingConfig) -> None:
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be greater than or equal to 0")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be greater than or equal to 0")
    if config.lr_scheduler_patience < 1:
        raise ValueError("lr_scheduler_patience must be at least 1")
    if not 0 < config.lr_scheduler_factor < 1:
        raise ValueError("lr_scheduler_factor must be between 0 and 1")
    if config.min_learning_rate <= 0:
        raise ValueError("min_learning_rate must be greater than 0")


def _usable_observations(observations: list[PopulationObservation]) -> list[PopulationObservation]:
    return [
        item
        for item in observations
        if item.congestion_level in LABEL_TO_INDEX and item.observed_at is not None
    ]
