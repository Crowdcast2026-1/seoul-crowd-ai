import unittest
from datetime import date, datetime, time, timedelta
from tempfile import TemporaryDirectory
from pathlib import Path
from zoneinfo import ZoneInfo

from app.ml import CONGESTION_LABELS, predict_crowd, split_observations, train_crowd_model
from app.models import PopulationObservation, TrainingConfig


SEOUL_TZ = ZoneInfo("Asia/Seoul")


class PyTorchModelTest(unittest.TestCase):
    def test_train_splits_data_and_saves_metrics(self):
        observations = _sample_observations(24)

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "crowd_model.pt"
            result = train_crowd_model(
                observations,
                model_path=model_path,
                config=TrainingConfig(epochs=5, learning_rate=0.01, seed=7),
            )

            self.assertTrue(model_path.exists())
            self.assertEqual(sum(result["split_counts"].values()), 24)
            self.assertIn("loss", result["metrics"]["train"])
            self.assertIn("accuracy", result["metrics"]["validation"])
            self.assertIn("accuracy", result["metrics"]["test"])
            self.assertIn("training_summary", result)
            self.assertLessEqual(result["training_summary"]["best_epoch"], result["training_summary"]["epochs_ran"])

            prediction = predict_crowd(
                model_path=model_path,
                area_name="광화문·덕수궁",
                target_date=date(2030, 1, 1),
                target_time=time(18, 30),
            )
            self.assertIn(prediction["predicted_congestion_level"], CONGESTION_LABELS)
            self.assertEqual(set(prediction["probabilities"]), set(CONGESTION_LABELS))

    def test_split_requires_valid_ratios(self):
        with self.assertRaises(ValueError):
            split_observations(_sample_observations(5), train_ratio=0.9, validation_ratio=0.2, seed=1)


def _sample_observations(count: int) -> list[PopulationObservation]:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=SEOUL_TZ)
    rows = []
    for index in range(count):
        observed_at = start + timedelta(hours=index * 3)
        label = CONGESTION_LABELS[index % len(CONGESTION_LABELS)]
        rows.append(
            PopulationObservation(
                area_name="광화문·덕수궁",
                area_code="POI009",
                observed_at=observed_at,
                source_updated_at=observed_at,
                congestion_level=label,
                population_min=1000 + index * 100,
                population_max=1500 + index * 100,
            )
        )
    return rows


if __name__ == "__main__":
    unittest.main()
