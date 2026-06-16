from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Any
from uuid import uuid4
import logging

from app.config import Settings
from app.database import PopulationDatabase
from app.workflows import collect_all_areas


LOGGER = logging.getLogger("uvicorn.error")


@dataclass
class CollectionJobStatus:
    running: bool = False
    job_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    stop_requested: bool = False
    interval_seconds: int = 300
    next_run_at: str | None = None
    rounds_completed: int = 0
    requested_area_count: int = 0
    collected_total: int = 0
    errors_total: int = 0
    last_result: dict[str, Any] | None = None
    recent_errors: list[dict[str, Any]] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)


class CollectionJobRunner:
    def __init__(self) -> None:
        self._lock = Lock()
        self._status = CollectionJobStatus()
        self._thread: Thread | None = None

    def start(self, settings: Settings, database: PopulationDatabase, interval_seconds: int = 300) -> dict[str, Any]:
        if interval_seconds < 1:
            raise ValueError("interval_seconds must be at least 1")

        with self._lock:
            if self._status.running:
                raise ValueError("A collection job is already running")

            self._status = CollectionJobStatus(
                running=True,
                job_id=uuid4().hex,
                started_at=_now_iso(),
                interval_seconds=interval_seconds,
            )
            self._append_event_locked(
                "collection job started",
                job_id=self._status.job_id,
                interval_seconds=interval_seconds,
            )
            LOGGER.info(
                "Seoul population collection job started: job_id=%s interval_seconds=%s",
                self._status.job_id,
                interval_seconds,
            )
            status = self._status_dict()

            self._thread = Thread(
                target=self._run_forever,
                args=(settings, database),
                name="seoul-population-collector",
                daemon=True,
            )
            self._thread.start()
            return status

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._status.running:
                return self._status_dict()
            self._status.stop_requested = True
            self._append_event_locked("collection job stop requested", job_id=self._status.job_id)
            LOGGER.info("Seoul population collection stop requested: job_id=%s", self._status.job_id)
            return self._status_dict()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_dict()

    def _run_forever(self, settings: Settings, database: PopulationDatabase) -> None:
        try:
            while True:
                with self._lock:
                    if self._status.stop_requested:
                        break
                    interval_seconds = self._status.interval_seconds
                    self._status.next_run_at = None
                    job_id = self._status.job_id
                    round_number = self._status.rounds_completed + 1
                    self._append_event_locked("collection round started", job_id=job_id, round=round_number)

                LOGGER.info("Seoul population collection round started: job_id=%s round=%s", job_id, round_number)

                started = monotonic()
                try:
                    result = collect_all_areas(settings, database)
                except Exception as exc:
                    result = {
                        "requested_area_count": 0,
                        "collected": [],
                        "errors": [{"area": "all", "error": str(exc)}],
                    }

                errors = result.get("errors", [])
                collected_count = len(result.get("collected", []))
                error_count = len(errors)
                requested_area_count = int(result.get("requested_area_count", 0))
                elapsed = monotonic() - started
                with self._lock:
                    self._status.rounds_completed += 1
                    self._status.requested_area_count = requested_area_count
                    self._status.collected_total += collected_count
                    self._status.errors_total += error_count
                    self._status.last_result = {
                        "collected_count": collected_count,
                        "error_count": error_count,
                        "requested_area_count": requested_area_count,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                    self._status.recent_errors = (self._status.recent_errors + errors)[-20:]
                    self._append_event_locked(
                        "collection round finished",
                        job_id=self._status.job_id,
                        round=self._status.rounds_completed,
                        collected_count=collected_count,
                        error_count=error_count,
                        requested_area_count=requested_area_count,
                        elapsed_seconds=round(elapsed, 3),
                    )

                LOGGER.info(
                    "Seoul population collection round finished: job_id=%s round=%s collected=%s errors=%s requested=%s elapsed=%.3fs",
                    job_id,
                    round_number,
                    collected_count,
                    error_count,
                    requested_area_count,
                    elapsed,
                )
                if errors:
                    error_areas = [str(item.get("area")) for item in errors[:10]]
                    LOGGER.warning(
                        "Seoul population collection round had errors: job_id=%s round=%s error_count=%s sample_areas=%s",
                        job_id,
                        round_number,
                        error_count,
                        ",".join(error_areas),
                    )

                wait_seconds = max(0.0, interval_seconds - elapsed)
                next_run_at = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
                with self._lock:
                    if self._status.stop_requested:
                        break
                    self._status.next_run_at = next_run_at.isoformat()
                    self._append_event_locked(
                        "next collection round scheduled",
                        job_id=self._status.job_id,
                        next_run_at=self._status.next_run_at,
                        wait_seconds=round(wait_seconds, 3),
                    )

                LOGGER.info(
                    "Next Seoul population collection round scheduled: job_id=%s next_run_at=%s wait_seconds=%.3f",
                    job_id,
                    next_run_at.isoformat(),
                    wait_seconds,
                )

                self._sleep_until_next_run(wait_seconds)

        finally:
            with self._lock:
                self._status.running = False
                self._status.finished_at = _now_iso()
                self._status.next_run_at = None
                self._append_event_locked("collection job stopped", job_id=self._status.job_id)
                LOGGER.info("Seoul population collection job stopped: job_id=%s", self._status.job_id)

    def _sleep_until_next_run(self, wait_seconds: float) -> None:
        deadline = monotonic() + wait_seconds
        while monotonic() < deadline:
            with self._lock:
                if self._status.stop_requested:
                    return
            sleep(min(1.0, max(0.0, deadline - monotonic())))

    def _status_dict(self) -> dict[str, Any]:
        return {
            "running": self._status.running,
            "job_id": self._status.job_id,
            "started_at": self._status.started_at,
            "finished_at": self._status.finished_at,
            "stop_requested": self._status.stop_requested,
            "interval_seconds": self._status.interval_seconds,
            "next_run_at": self._status.next_run_at,
            "rounds_completed": self._status.rounds_completed,
            "requested_area_count": self._status.requested_area_count,
            "collected_total": self._status.collected_total,
            "errors_total": self._status.errors_total,
            "last_result": self._status.last_result,
            "recent_errors": list(self._status.recent_errors),
            "recent_events": list(self._status.recent_events),
        }

    def _append_event_locked(self, message: str, **fields: Any) -> None:
        event = {"at": _now_iso(), "message": message}
        event.update(fields)
        self._status.recent_events = (self._status.recent_events + [event])[-50:]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
