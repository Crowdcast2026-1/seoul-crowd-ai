from __future__ import annotations

from datetime import date, time
from typing import Any

from app.areas import all_area_codes
from app.config import Settings
from app.database import PopulationDatabase, observation_to_dict
from app.ml import predict_crowd, train_crowd_model
from app.models import TrainingConfig
from app.seoul_api import SeoulPopulationClient


def make_client(settings: Settings) -> SeoulPopulationClient:
    return SeoulPopulationClient(
        api_key=settings.seoul_api_key,
        base_url=settings.seoul_api_base_url,
        timeout_seconds=settings.seoul_api_timeout_seconds,
    )


def collect_once(settings: Settings, database: PopulationDatabase, areas: list[str]) -> dict[str, Any]:
    client = make_client(settings)
    collected = []
    errors = []
    for area in areas:
        try:
            observation = client.fetch_population(area)
            observation_id = database.save_observation(observation)
            collected.append(observation_to_dict(observation, observation_id=observation_id))
        except Exception as exc:
            errors.append({"area": area, "error": str(exc)})
    return {"collected": collected, "errors": errors}


def collect_all_areas(settings: Settings, database: PopulationDatabase) -> dict[str, Any]:
    result = collect_once(settings, database, all_area_codes())
    result["requested_area_count"] = len(all_area_codes())
    return result


def train_model(
    settings: Settings,
    database: PopulationDatabase,
    area_name: str | None,
    config: TrainingConfig,
) -> dict[str, Any]:
    observations = database.list_observations(area_name=area_name)
    return train_crowd_model(observations, model_path=settings.model_path, config=config)


def predict_model(settings: Settings, area_name: str, target_date: date, target_time: time) -> dict[str, Any]:
    return predict_crowd(
        model_path=settings.model_path,
        area_name=area_name,
        target_date=target_date,
        target_time=target_time,
    )
