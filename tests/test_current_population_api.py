import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.areas import all_area_codes
from app.main import app, current_population_all
from app.models import PopulationObservation
from app.workflows import fetch_current_all_areas


class FakePopulationClient:
    def fetch_population(self, area: str) -> PopulationObservation:
        return PopulationObservation(
            area_name=f"Area {area}",
            area_code=area,
            observed_at=datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc),
            congestion_level="normal",
            population_min=100,
            population_max=200,
        )


class FailingPopulationClient:
    def fetch_population(self, area: str) -> PopulationObservation:
        raise TimeoutError("timed out")


class FakeFallbackDatabase:
    def list_latest_observations_by_area_codes(self, area_codes: list[str]) -> dict[str, PopulationObservation]:
        return {
            area: PopulationObservation(
                area_name=f"Cached {area}",
                area_code=area,
                observed_at=datetime(2026, 6, 21, 11, 55, tzinfo=timezone.utc),
                congestion_level="cached",
                population_min=50,
                population_max=150,
            )
            for area in area_codes
        }


class CurrentPopulationApiTest(unittest.TestCase):
    def test_fetch_current_all_areas_returns_121_frontend_items_in_order(self):
        settings = SimpleNamespace(
            seoul_api_timeout_seconds=10,
            seoul_api_max_workers=8,
            seoul_api_all_deadline_seconds=12,
        )

        with patch("app.workflows.make_client", return_value=FakePopulationClient()):
            result = fetch_current_all_areas(settings)

        self.assertEqual(result["requested_area_count"], 121)
        self.assertEqual(result["returned_area_count"], 121)
        self.assertEqual(result["success_count"], 121)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["unavailable_count"], 0)
        self.assertTrue(result["is_complete"])
        self.assertTrue(result["is_complete_live"])
        self.assertEqual(len(result["areas"]), 121)
        self.assertEqual(result["areas"][0]["area_code"], "POI001")
        self.assertEqual(result["areas"][-1]["area_code"], "POI121")
        self.assertEqual([item["area_code"] for item in result["areas"]], all_area_codes())
        self.assertTrue(all(item["data_source"] == "live" for item in result["areas"]))

    def test_fetch_current_all_areas_uses_database_fallback_for_live_failures(self):
        settings = SimpleNamespace(
            seoul_api_timeout_seconds=10,
            seoul_api_max_workers=8,
            seoul_api_all_deadline_seconds=12,
        )

        with patch("app.workflows.make_client", return_value=FailingPopulationClient()):
            result = fetch_current_all_areas(settings, fallback_database=FakeFallbackDatabase())

        self.assertEqual(result["requested_area_count"], 121)
        self.assertEqual(result["returned_area_count"], 121)
        self.assertEqual(result["live_success_count"], 0)
        self.assertEqual(result["live_error_count"], 121)
        self.assertEqual(result["fallback_count"], 121)
        self.assertEqual(result["success_count"], 121)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["unavailable_count"], 0)
        self.assertTrue(result["is_complete"])
        self.assertFalse(result["is_complete_live"])
        self.assertEqual(result["areas"][0]["area_code"], "POI001")
        self.assertEqual(result["areas"][0]["data_source"], "database_fallback")
        self.assertIn("live_error", result["areas"][0])

    def test_current_population_all_route_exists(self):
        self.assertIn("/population/current/all", {route.path for route in app.routes})

    def test_current_population_all_returns_status_body_when_live_fetches_fail(self):
        with patch(
            "app.main.fetch_current_all_areas",
            return_value={
                "requested_area_count": 121,
                "returned_area_count": 121,
                "success_count": 0,
                "error_count": 121,
                "unavailable_count": 121,
                "is_complete": False,
                "areas": [],
                "errors": [{"area": "POI001", "error": "failed"}],
            },
        ):
            result = current_population_all()

        self.assertEqual(result["error_count"], 121)
        self.assertFalse(result["is_complete"])


if __name__ == "__main__":
    unittest.main()
