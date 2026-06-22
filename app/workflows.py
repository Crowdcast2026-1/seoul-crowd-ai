from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import date, time
from typing import Any

from app.areas import all_area_codes
from app.config import Settings
from app.database import PopulationDatabase, observation_to_dict
from app.ml import predict_crowd, train_crowd_model
from app.models import TrainingConfig
from app.seoul_api import SeoulPopulationClient


def make_client(settings: Settings, timeout_seconds: float | None = None) -> SeoulPopulationClient:
    return SeoulPopulationClient(
        api_key=settings.seoul_api_key,
        base_url=settings.seoul_api_base_url,
        timeout_seconds=timeout_seconds or settings.seoul_api_timeout_seconds,
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


def fetch_current_all_areas(
    settings: Settings,
    fallback_database: PopulationDatabase | None = None,
) -> dict[str, Any]:
    areas = all_area_codes()
    deadline_seconds = max(1, settings.seoul_api_all_deadline_seconds)
    client = make_client(settings, timeout_seconds=min(settings.seoul_api_timeout_seconds, deadline_seconds))
    max_workers = max(1, min(settings.seoul_api_max_workers, len(areas)))
    live_by_area: dict[str, dict[str, Any]] = {}
    errors_by_area: dict[str, str] = {}

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {executor.submit(client.fetch_population, area): area for area in areas}
        try:
            completed = as_completed(futures, timeout=deadline_seconds)
            for future in completed:
                area = futures[future]
                try:
                    payload = observation_to_dict(future.result())
                    payload["data_source"] = "live"
                    payload["has_data"] = True
                    live_by_area[area] = payload
                except Exception as exc:
                    errors_by_area[area] = str(exc)
        except FuturesTimeoutError:
            pass
        finally:
            for future, area in futures.items():
                if area in live_by_area or area in errors_by_area:
                    continue
                if future.cancelled() or not future.done():
                    errors_by_area[area] = f"live fetch did not finish within {deadline_seconds:g}s"
                else:
                    try:
                        payload = observation_to_dict(future.result())
                        payload["data_source"] = "live"
                        payload["has_data"] = True
                        live_by_area[area] = payload
                    except Exception as exc:
                        errors_by_area[area] = str(exc)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    fallback_by_area: dict[str, dict[str, Any]] = {}
    missing_areas = [area for area in areas if area not in live_by_area]
    if fallback_database is not None and missing_areas:
        latest_observations = fallback_database.list_latest_observations_by_area_codes(missing_areas)
        for area, observation in latest_observations.items():
            payload = observation_to_dict(observation)
            payload["data_source"] = "database_fallback"
            payload["has_data"] = True
            if area in errors_by_area:
                payload["live_error"] = errors_by_area[area]
            fallback_by_area[area] = payload

    area_payloads = []
    for area in areas:
        if area in live_by_area:
            area_payloads.append(live_by_area[area])
        elif area in fallback_by_area:
            area_payloads.append(fallback_by_area[area])
        else:
            area_payloads.append(
                {
                    "area_code": area,
                    "data_source": "unavailable",
                    "has_data": False,
                    "live_error": errors_by_area.get(area, "no live or fallback data available"),
                }
            )

    errors = [
        {"area": area, "error": errors_by_area[area]}
        for area in areas
        if area in errors_by_area and area not in fallback_by_area
    ]

    return {
        "requested_area_count": len(areas),
        "live_success_count": len(live_by_area),
        "live_error_count": len(errors_by_area),
        "fallback_count": len(fallback_by_area),
        "success_count": len(live_by_area) + len(fallback_by_area),
        "error_count": len(errors),
        "returned_area_count": len(area_payloads),
        "unavailable_count": len(errors),
        "is_complete_live": not errors_by_area,
        "is_complete": not errors,
        "live_deadline_seconds": deadline_seconds,
        "areas": area_payloads,
        "errors": errors,
    }


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
