from __future__ import annotations

from datetime import datetime
from typing import Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

from app.models import PopulationObservation


SEOUL_TZ = ZoneInfo("Asia/Seoul")
SUCCESS_CODES = {"INFO-000", "INFO-100"}


class SeoulApiError(RuntimeError):
    """Raised when Seoul Open API returns an invalid or failed response."""


class SeoulPopulationClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://openapi.seoul.go.kr:8088",
        timeout_seconds: float = 10,
        service: str = "citydata_ppltn",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.service = service

    def build_url(self, area_name_or_code: str) -> str:
        return (
            f"{self.base_url}/{quote(self.api_key, safe='')}/xml/"
            f"{self.service}/1/5/{quote(area_name_or_code, safe='')}"
        )

    def fetch_population(self, area_name_or_code: str) -> PopulationObservation:
        url = self.build_url(area_name_or_code)
        request = Request(url, headers={"User-Agent": "seoul-crowd-forecast/0.1"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read()
        return parse_population_xml(body)

    def fetch_many(self, areas: Iterable[str]) -> list[PopulationObservation]:
        return [self.fetch_population(area) for area in areas]


def parse_population_xml(body: bytes | str) -> PopulationObservation:
    if isinstance(body, bytes):
        text = body.decode("utf-8")
    else:
        text = body

    root = ET.fromstring(text)
    result_code = _text(root, ("RESULT/CODE", ".//RESULT/CODE", "RESULT/RESULT.CODE", ".//RESULT.CODE"))
    result_message = _text(
        root,
        ("RESULT/MESSAGE", ".//RESULT/MESSAGE", "RESULT/RESULT.MESSAGE", ".//RESULT.MESSAGE"),
    )
    if result_code and result_code not in SUCCESS_CODES:
        raise SeoulApiError(f"Seoul API error {result_code}: {result_message or 'unknown error'}")

    row = root.find(".//row")
    if row is None:
        row = _find_population_node(root)
        if row is None:
            row = root
    fields = {
        "AREA_NM": _text(row, ("AREA_NM",)),
        "AREA_CD": _text(row, ("AREA_CD",)),
        "AREA_CONGEST_LVL": _text(row, ("AREA_CONGEST_LVL",)),
        "AREA_CONGEST_MSG": _text(row, ("AREA_CONGEST_MSG",)),
        "AREA_PPLTN_MIN": _text(row, ("AREA_PPLTN_MIN",)),
        "AREA_PPLTN_MAX": _text(row, ("AREA_PPLTN_MAX",)),
        "MALE_PPLTN_RATE": _text(row, ("MALE_PPLTN_RATE",)),
        "FEMALE_PPLTN_RATE": _text(row, ("FEMALE_PPLTN_RATE",)),
        "RESNT_PPLTN_RATE": _text(row, ("RESNT_PPLTN_RATE",)),
        "NON_RESNT_PPLTN_RATE": _text(row, ("NON_RESNT_PPLTN_RATE",)),
        "PPLTN_TIME": _text(row, ("PPLTN_TIME",)),
    }

    area_name = fields["AREA_NM"]
    if not area_name:
        raise SeoulApiError("Seoul API response did not include AREA_NM")

    source_updated_at = _parse_datetime(fields["PPLTN_TIME"])
    observed_at = source_updated_at or datetime.now(tz=SEOUL_TZ)

    return PopulationObservation(
        area_name=area_name,
        area_code=fields["AREA_CD"],
        observed_at=observed_at,
        source_updated_at=source_updated_at,
        congestion_level=fields["AREA_CONGEST_LVL"],
        congestion_message=fields["AREA_CONGEST_MSG"],
        population_min=_parse_int(fields["AREA_PPLTN_MIN"]),
        population_max=_parse_int(fields["AREA_PPLTN_MAX"]),
        male_rate=_parse_float(fields["MALE_PPLTN_RATE"]),
        female_rate=_parse_float(fields["FEMALE_PPLTN_RATE"]),
        resident_rate=_parse_float(fields["RESNT_PPLTN_RATE"]),
        non_resident_rate=_parse_float(fields["NON_RESNT_PPLTN_RATE"]),
        raw={key: value for key, value in fields.items() if value is not None},
    )


def _find_population_node(root: ET.Element) -> ET.Element | None:
    for node in root.iter():
        if node.find("AREA_NM") is not None:
            return node
    return None


def _text(node: ET.Element, paths: tuple[str, ...]) -> str | None:
    for path in paths:
        found = node.find(path)
        if found is not None and found.text is not None:
            value = found.text.strip()
            if value:
                return value
    return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace("%", "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = value.strip().replace("T", " ")
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(normalized, fmt)
            return parsed.replace(tzinfo=SEOUL_TZ)
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=SEOUL_TZ)
    return parsed.astimezone(SEOUL_TZ)
