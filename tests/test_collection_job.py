import unittest
from time import sleep
from types import SimpleNamespace
from unittest.mock import patch

from app.areas import all_area_codes, all_area_names
from app.collection_job import CollectionJobRunner
from app.main import area_names


class CollectionJobTest(unittest.TestCase):
    def test_all_area_codes_has_121_poi_codes(self):
        codes = all_area_codes()

        self.assertEqual(len(codes), 121)
        self.assertEqual(codes[0], "POI001")
        self.assertEqual(codes[-1], "POI121")

    def test_all_area_names_returns_names_only(self):
        names = all_area_names()

        self.assertEqual(len(names), 111)
        self.assertEqual(names[0], "강남 MICE 관광특구")
        self.assertEqual(names[-1], "송리단길·호수단길")
        self.assertIn("광화문·덕수궁", names)
        self.assertFalse(any(name.startswith("POI") for name in names))

    def test_area_names_api_returns_names_only(self):
        names = area_names()

        self.assertEqual(names, all_area_names())
        self.assertFalse(any(name.startswith("POI") for name in names))

    def test_collection_job_records_status(self):
        runner = CollectionJobRunner()

        with patch(
            "app.collection_job.collect_all_areas",
            return_value={
                "requested_area_count": 121,
                "collected": [{"id": 1}],
                "errors": [],
            },
        ):
            runner.start(SimpleNamespace(), SimpleNamespace(), interval_seconds=60)
            while runner.status()["rounds_completed"] < 1:
                sleep(0.01)
            runner.stop()

        status = runner.status()
        self.assertGreaterEqual(status["rounds_completed"], 1)
        self.assertEqual(status["requested_area_count"], 121)
        self.assertGreaterEqual(status["collected_total"], 1)
        self.assertTrue(any(event["message"] == "collection round finished" for event in status["recent_events"]))

    def test_start_rejects_second_running_job(self):
        runner = CollectionJobRunner()
        with patch(
            "app.collection_job.collect_all_areas",
            return_value={"requested_area_count": 121, "collected": [], "errors": []},
        ):
            runner.start(SimpleNamespace(), SimpleNamespace(), interval_seconds=60)
            with self.assertRaises(ValueError):
                runner.start(SimpleNamespace(), SimpleNamespace(), interval_seconds=60)
            runner.stop()

        self.assertTrue(
            any(event["message"] == "collection job stop requested" for event in runner.status()["recent_events"])
        )


if __name__ == "__main__":
    unittest.main()
