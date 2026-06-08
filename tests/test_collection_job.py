import unittest
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
        runner.start(SimpleNamespace(), SimpleNamespace(), duration_seconds=1)

        with patch(
            "app.collection_job.collect_all_areas",
            return_value={
                "requested_area_count": 121,
                "collected": [{"id": 1}],
                "errors": [],
            },
        ):
            runner.run(SimpleNamespace(), SimpleNamespace())

        status = runner.status()
        self.assertFalse(status["running"])
        self.assertGreaterEqual(status["rounds_completed"], 1)
        self.assertEqual(status["requested_area_count"], 121)
        self.assertGreaterEqual(status["collected_total"], 1)


if __name__ == "__main__":
    unittest.main()
