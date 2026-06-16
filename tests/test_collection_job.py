import unittest
from time import sleep
from types import SimpleNamespace
from unittest.mock import patch

from app.areas import all_area_codes
from app.collection_job import CollectionJobRunner


class CollectionJobTest(unittest.TestCase):
    def test_all_area_codes_has_121_poi_codes(self):
        codes = all_area_codes()

        self.assertEqual(len(codes), 121)
        self.assertEqual(codes[0], "POI001")
        self.assertEqual(codes[-1], "POI121")

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
