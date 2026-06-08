from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.database import PopulationDatabase
from app.workflows import collect_all_areas


@dataclass
class CollectionJobStatus:
    running: bool = False
    job_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: int = 0
    rounds_completed: int = 0
    requested_area_count: int = 0
    collected_total: int = 0
    errors_total: int = 0
    last_result: dict[str, Any] | None = None
    recent_errors: list[dict[str, Any]] = field(default_factory=list)


class CollectionJobRunner:
    def __init__(self) -> None:
        self._lock = Lock()
        self._status = CollectionJobStatus()

    def start(self, settings: Settings, database: PopulationDatabase, duration_seconds: int) -> dict[str, Any]:
        if duration_seconds < 1:
            raise ValueError("duration_seconds must be at least 1")

        with self._lock:
            if self._status.running:
                raise ValueError("A collection job is already running")

            self._status = CollectionJobStatus(
                running=True,
                job_id=uuid4().hex,
                started_at=_now_iso(),
                duration_seconds=duration_seconds,
            )
            return self._status_dict()

    def run(self, settings: Settings, database: PopulationDatabase) -> None:
        with self._lock:
            duration_seconds = self._status.duration_seconds

        deadline = monotonic() + duration_seconds
        try:
            while monotonic() < deadline:
                try:
                    result = collect_all_areas(settings, database)
                except Exception as exc:
                    result = {
                        "requested_area_count": 0,
                        "collected": [],
                        "errors": [{"area": "all", "error": str(exc)}],
                    }

                errors = result.get("errors", [])
                with self._lock:
                    self._status.rounds_completed += 1
                    self._status.requested_area_count = int(result.get("requested_area_count", 0))
                    self._status.collected_total += len(result.get("collected", []))
                    self._status.errors_total += len(errors)
                    self._status.last_result = {
                        "collected_count": len(result.get("collected", [])),
                        "error_count": len(errors),
                        "requested_area_count": result.get("requested_area_count", 0),
                    }
                    self._status.recent_errors = (self._status.recent_errors + errors)[-20:]

                if monotonic() < deadline:
                    sleep(0.1)

        finally:
            with self._lock:
                self._status.running = False
                self._status.finished_at = _now_iso()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_dict()

    def _status_dict(self) -> dict[str, Any]:
        return {
            "running": self._status.running,
            "job_id": self._status.job_id,
            "started_at": self._status.started_at,
            "finished_at": self._status.finished_at,
            "duration_seconds": self._status.duration_seconds,
            "rounds_completed": self._status.rounds_completed,
            "requested_area_count": self._status.requested_area_count,
            "collected_total": self._status.collected_total,
            "errors_total": self._status.errors_total,
            "last_result": self._status.last_result,
            "recent_errors": list(self._status.recent_errors),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
