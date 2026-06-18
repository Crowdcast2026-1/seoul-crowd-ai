from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PopulationObservation:
    area_name: str
    observed_at: datetime
    area_code: str | None = None
    source_updated_at: datetime | None = None
    congestion_level: str | None = None
    congestion_message: str | None = None
    population_min: int | None = None
    population_max: int | None = None
    male_rate: float | None = None
    female_rate: float | None = None
    resident_rate: float | None = None
    non_resident_rate: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def population_midpoint(self) -> float | None:
        if self.population_min is None or self.population_max is None:
            return None
        return (self.population_min + self.population_max) / 2


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    learning_rate: float = 0.1
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    seed: int = 42
    weight_decay: float = 0.0001
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 0.0001
    lr_scheduler_patience: int = 15
    lr_scheduler_factor: float = 0.5
    min_learning_rate: float = 0.0001
