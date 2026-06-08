from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
import json
import sqlite3

from app.models import PopulationObservation


SCHEMA = """
CREATE TABLE IF NOT EXISTS population_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    area_name TEXT NOT NULL,
    area_code TEXT,
    observed_at TEXT NOT NULL,
    source_updated_at TEXT,
    congestion_level TEXT,
    congestion_message TEXT,
    population_min INTEGER,
    population_max INTEGER,
    male_rate REAL,
    female_rate REAL,
    resident_rate REAL,
    non_resident_rate REAL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(area_name, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_population_observations_area_time
ON population_observations(area_name, observed_at);
"""


class PopulationDatabase:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def init(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def save_observation(self, observation: PopulationObservation) -> int:
        self.init()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO population_observations (
                    area_name, area_code, observed_at, source_updated_at,
                    congestion_level, congestion_message, population_min, population_max,
                    male_rate, female_rate, resident_rate, non_resident_rate, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.area_name,
                    observation.area_code,
                    _dt(observation.observed_at),
                    _dt(observation.source_updated_at),
                    observation.congestion_level,
                    observation.congestion_message,
                    observation.population_min,
                    observation.population_max,
                    observation.male_rate,
                    observation.female_rate,
                    observation.resident_rate,
                    observation.non_resident_rate,
                    json.dumps(observation.raw, ensure_ascii=False, sort_keys=True),
                ),
            )
            row = conn.execute(
                """
                SELECT id FROM population_observations
                WHERE area_name = ? AND observed_at = ?
                """,
                (observation.area_name, _dt(observation.observed_at)),
            ).fetchone()
        return int(row["id"])

    def list_observations(self, area_name: str | None = None, limit: int | None = None) -> list[PopulationObservation]:
        self.init()
        sql = "SELECT * FROM population_observations"
        params: list[Any] = []
        if area_name:
            sql += " WHERE area_name = ?"
            params.append(area_name)
        sql += " ORDER BY observed_at ASC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_observation(row) for row in rows]

    def count_observations(self, area_name: str | None = None) -> int:
        self.init()
        with self.connect() as conn:
            if area_name:
                row = conn.execute(
                    "SELECT COUNT(*) AS count FROM population_observations WHERE area_name = ?",
                    (area_name,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS count FROM population_observations").fetchone()
        return int(row["count"])

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def observation_to_dict(observation: PopulationObservation, observation_id: int | None = None) -> dict[str, Any]:
    payload = {
        "id": observation_id,
        "area_name": observation.area_name,
        "area_code": observation.area_code,
        "observed_at": _dt(observation.observed_at),
        "source_updated_at": _dt(observation.source_updated_at),
        "congestion_level": observation.congestion_level,
        "congestion_message": observation.congestion_message,
        "population_min": observation.population_min,
        "population_max": observation.population_max,
        "population_midpoint": observation.population_midpoint,
        "male_rate": observation.male_rate,
        "female_rate": observation.female_rate,
        "resident_rate": observation.resident_rate,
        "non_resident_rate": observation.non_resident_rate,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _row_to_observation(row: sqlite3.Row) -> PopulationObservation:
    return PopulationObservation(
        area_name=row["area_name"],
        area_code=row["area_code"],
        observed_at=_parse_dt(row["observed_at"]),
        source_updated_at=_parse_dt(row["source_updated_at"]),
        congestion_level=row["congestion_level"],
        congestion_message=row["congestion_message"],
        population_min=row["population_min"],
        population_max=row["population_max"],
        male_rate=row["male_rate"],
        female_rate=row["female_rate"],
        resident_rate=row["resident_rate"],
        non_resident_rate=row["non_resident_rate"],
        raw=json.loads(row["raw_json"] or "{}"),
    )


def _dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)
