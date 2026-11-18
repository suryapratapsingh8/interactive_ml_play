import io
import unittest
from fastapi.testclient import TestClient
from main import app
from pipeline import run_experiment, run_manual_playground
from schemas import ExperimentConfig, ManualPlaygroundConfig


def _model_to_dict(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


class TestMLPlaygroundV21(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_proactive_nan_strategy_recommendation_endpoint(self):
        res = self.client.get("/api/datasets/housing_synthetic/recommend-nan-strategy")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn(data["recommended_nan_strategy"], ("drop", "median_mode", "mean_mode", "knn_impute"))
        self.assertGreater(data["nan_row_pct"], 0)
        self.assertGreaterEqual(len(data["available_strategies"]), 6)

    def test_percentage_sampling_20_to_80_and_all_nan_strategies(self):
        # Verify 25% vs 75% sampling
        res_25 = run_experiment(ExperimentConfig(dataset_id="iris", sample_percentage=25.0, nan_strategy="drop"))
        res_75 = run_experiment(ExperimentConfig(dataset_id="iris", sample_percentage=75.0, nan_strategy="drop"))
        self.assertLess(res_25.sampling_summary.sampled_rows, res_75.sampling_summary.sampled_rows)
        self.assertEqual(res_25.sampling_summary.sample_percentage_used, 25.0)
        self.assertEqual(res_75.sampling_summary.sample_percentage_used, 75.0)

        # Verify all NaN strategies run cleanly in Manual Playground
        for strat in ("auto", "drop", "median_mode", "mean_mode", "knn_impute", "ffill_bfill", "constant_zero"):
            man_res = run_manual_playground(
                ManualPlaygroundConfig(
                    dataset_id="housing_synthetic",
                    algorithm_id="ridge_reg",
                    sample_percentage=50.0,
                    nan_strategy=strat,
                )
            )
            if strat == "drop":
                self.assertGreater(man_res.sampling_summary.nan_rows_removed, 0)
            elif strat != "auto":
                # Imputers preserve all rows whose target is non-NaN
                self.assertEqual(man_res.sampling_summary.nan_rows_removed, 0)
                self.assertEqual(man_res.sampling_summary.nan_strategy_applied, strat)

    def test_custom_csv_upload_with_nan_recommendation_and_percentage_preview(self):
        rows = ["age,income,credit_score,approved_status"]
        for i in range(200):
            if i % 8 == 0:
                rows.append(f"{25 + (i % 40)},,700,{'Yes' if i % 2 == 0 else 'No'}")
            else:
                rows.append(f"{25 + (i % 40)},{45000 + i * 180},{620 + (i % 150)},{'Yes' if i % 2 == 0 else 'No'}")
        csv_bytes = "\n".join(rows).encode("utf-8")

        up_res = self.client.post(
            "/api/datasets/upload",
            files={"file": ("loan_applications.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"task_type": "auto"},
        )
        self.assertEqual(up_res.status_code, 200)
        up_data = up_res.json()
        self.assertIn("recommended_nan_strategy", up_data)
        self.assertEqual(up_data["detected_task_type"], "classification")

        prev_res = self.client.post(
            f"/api/datasets/{up_data['id']}/sample-preview?sample_percentage=40&nan_strategy=knn_impute"
        )
        self.assertEqual(prev_res.status_code, 200)
        prev_data = prev_res.json()
        self.assertEqual(prev_data["nan_strategy_applied"], "knn_impute")
        self.assertEqual(prev_data["sampled_rows_count"], 80)  # 40% of 200 imputed rows


if __name__ == "__main__":
    unittest.main()