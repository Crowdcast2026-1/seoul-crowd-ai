from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AREA = "광화문·덕수궁"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _split_areas(value: str) -> tuple[str, ...]:
    areas = tuple(area.strip() for area in value.split(",") if area.strip())
    return areas or (DEFAULT_AREA,)


@dataclass(frozen=True)
class Settings:
    seoul_api_key: str
    seoul_api_base_url: str
    default_areas: tuple[str, ...]
    database_path: Path
    model_dir: Path
    seoul_api_timeout_seconds: float
    seoul_api_max_workers: int
    seoul_api_all_deadline_seconds: float
    model_path: Path


def get_settings() -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")

    model_dir = _project_path(os.getenv("MODEL_DIR", "models"))
    return Settings(
        seoul_api_key=os.getenv("SEOUL_API_KEY", "sample"),
        seoul_api_base_url=os.getenv("SEOUL_API_BASE_URL", "http://openapi.seoul.go.kr:8088"),
        default_areas=_split_areas(os.getenv("SEOUL_DEFAULT_AREAS", DEFAULT_AREA)),
        database_path=_project_path(os.getenv("DATABASE_PATH", "data/seoul_population.sqlite3")),
        model_dir=model_dir,
        seoul_api_timeout_seconds=float(os.getenv("SEOUL_API_TIMEOUT_SECONDS", "10")),
        seoul_api_max_workers=int(os.getenv("SEOUL_API_MAX_WORKERS", "8")),
        seoul_api_all_deadline_seconds=float(os.getenv("SEOUL_API_ALL_DEADLINE_SECONDS", "5")),
        model_path=_project_path(os.getenv("MODEL_PATH", str(model_dir / "crowd_model.pt"))),
    )
