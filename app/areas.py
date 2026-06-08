from __future__ import annotations


SEOUL_MAJOR_AREA_CODES = tuple(f"POI{index:03d}" for index in range(1, 122))


def all_area_codes() -> list[str]:
    return list(SEOUL_MAJOR_AREA_CODES)

