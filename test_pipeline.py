import time
import unittest
from fastapi.testclient import TestClient
from main import app


class TestMLPlaygroundV22(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_prepare_apply_nan_then_sample_percentage_and_download_csv(self):
        # Step 1: Apply KNN Imputer first, then sample 50% of post-NaN rows
        prep_res = self.client.post(
            "/api/datasets/housing_synthetic/prepare?nan_strategy=knn_impute&sample_percentage=50"
        )
        self.assertEqual(prep_res.status_code, 200)
        data = prep_res.json()
        self.assertEqual(data["nan_strategy_applied"], "knn_impute")
        self.assertEqual(data["remaining_nans_in_processed"], 0)
        self.assertEqual(data["sample_percentage_used"], 50.0)
        self.assertEqual(data["sampled_rows_count"], int(round(data["rows_after_nan_handling"] * 0.50)))

        # Step 2: Download the processed CSV and verify header & row count
        dl_res = self.client.get("/api/datasets/housing_synthetic/download-processed")
        self.assertEqual(dl_res.status_code, 200)
        self.assertIn("text/csv", dl_res.headers.get("content-type", ""))
        csv_lines = [ln for ln in dl_res.text.strip().splitlines() if ln]
        self.assertEqual(len(csv_lines) - 1, data["sampled_rows_count"])

    def test_async_benchmark_progress_bar_and_eta_tracking(self):
        start_res = self.client.post(
            "/api/auto-selector/start",
            json={"dataset_id": "iris", "sample_percentage": 40.0, "nan_strategy": "auto"},
        )
        self.assertEqual(start_res.status_code, 200)
        job_id = start_res.json()["job_id"]

        final_job = None
        for _ in range(80):
            st = self.client.get(f"/api/auto-selector/status/{job_id}").json()
            self.assertIn("progress_pct", st)
            self.assertIn("eta_seconds", st)
            if st["status"] in ("completed", "failed"):
                final_job = st
                break
            time.sleep(0.1)

        self.assertIsNotNone(final_job)
        self.assertEqual(final_job["status"], "completed")
        self.assertEqual(final_job["progress_pct"], 100.0)
        self.assertEqual(final_job["eta_seconds"], 0.0)
        self.assertIsNotNone(final_job["result"]["best_algorithm"])

    def test_public_openapi_schema_hides_internal_endpoints(self):
        schema = self.client.get("/openapi.json").json()
        public_paths = set(schema["paths"].keys())
        self.assertIn("/api/datasets/{dataset_id}/prepare", public_paths)
        self.assertIn("/api/datasets/{dataset_id}/download-processed", public_paths)
        self.assertNotIn("/api/health", public_paths)
        self.assertNotIn("/api/modes", public_paths)


if __name__ == "__main__":
    unittest.main()