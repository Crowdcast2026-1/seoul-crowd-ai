from __future__ import annotations

from datetime import date, time

from fastapi import FastAPI, HTTPException, Query

from app.areas import all_area_codes
from app.collection_job import CollectionJobRunner
from app.config import DEFAULT_AREA, get_settings
from app.database import PopulationDatabase, observation_to_dict
from app.models import TrainingConfig
from app.seoul_api import SeoulApiError
from app.workflows import collect_all_areas, collect_once, make_client, predict_model, train_model


settings = get_settings()
database = PopulationDatabase(settings.database_path)
collection_jobs = CollectionJobRunner()

app = FastAPI(
    title="Seoul Crowd Forecast API",
    description="서울 실시간 인구 API 수집, PyTorch 학습, 미래 날짜/시간대 혼잡도 예측 API",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    database.init()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "database_path": str(settings.database_path),
        "model_path": str(settings.model_path),
        "stored_observations": database.count_observations(),
    }


@app.get("/areas")
def areas() -> dict:
    codes = all_area_codes()
    return {
        "default_areas": settings.default_areas,
        "major_area_count": len(codes),
        "major_area_codes": codes,
    }


@app.get("/population/current")
def current_population(area: str = Query(DEFAULT_AREA, description="장소명 또는 장소 코드")) -> dict:
    try:
        observation = make_client(settings).fetch_population(area)
    except SeoulApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"Seoul API request failed: {exc}") from exc
    return observation_to_dict(observation)


@app.post("/collect")
def collect(area: list[str] | None = Query(default=None, description="반복 지정 가능한 장소명 또는 장소 코드")) -> dict:
    selected_areas = area or list(settings.default_areas)
    result = collect_once(settings, database, selected_areas)
    collected = result["collected"]
    errors = result["errors"]
    if errors and not collected:
        raise HTTPException(status_code=502, detail={"errors": errors})
    return result


@app.post("/collect/all")
def collect_all() -> dict:
    result = collect_all_areas(settings, database)
    if result["errors"] and not result["collected"]:
        raise HTTPException(status_code=502, detail={"errors": result["errors"]})
    return result


@app.post("/collect/all/continuous")
def collect_all_continuous(
    interval_minutes: float = Query(5, gt=0, le=1440),
) -> dict:
    try:
        status = collection_jobs.start(
            settings,
            database,
            interval_seconds=max(1, round(interval_minutes * 60)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "message": "continuous collection job started",
        "status": status,
    }


@app.get("/collect/all/continuous/status")
def collect_all_continuous_status() -> dict:
    return collection_jobs.status()


@app.post("/collect/all/continuous/stop")
def stop_collect_all_continuous() -> dict:
    return collection_jobs.stop()


@app.get("/observations")
def observations(
    area: str | None = Query(None, description="장소명"),
    limit: int = Query(100, ge=1, le=5000),
) -> dict:
    rows = database.list_observations(area_name=area, limit=limit)
    return {
        "area_name": area,
        "count": len(rows),
        "observations": [observation_to_dict(row) for row in rows],
    }


@app.post("/train")
def train(
    area: str | None = Query(None, description="특정 장소만 학습하려면 지정"),
    epochs: int = Query(80, ge=1, le=5000),
    learning_rate: float = Query(0.01, gt=0, le=1),
    train_ratio: float = Query(0.7, gt=0, lt=1),
    validation_ratio: float = Query(0.15, gt=0, lt=1),
    seed: int = Query(42),
) -> dict:
    try:
        return train_model(
            settings,
            database,
            area_name=area,
            config=TrainingConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                seed=seed,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/predictions")
def predictions(
    area: str = Query(DEFAULT_AREA, description="장소명"),
    target_date: date = Query(..., description="예측할 미래 날짜, 예: 2026-06-15"),
    target_time: time = Query(..., description="예측할 시간, 예: 18:30"),
) -> dict:
    try:
        return predict_model(settings, area_name=area, target_date=target_date, target_time=target_time)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
