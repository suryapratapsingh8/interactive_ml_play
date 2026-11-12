import io
import unittest
from fastapi.testclient import TestClient
from main import app
from pipeline import run_experiment, run_manual_playground
from schemas import ExperimentConfig, ManualPlaygroundConfig


def _model_to_dict(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


class TestMLPlaygroundProductionV2(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_modes_endpoints(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        modes = self.client.get("/api/modes").json()["modes"]
        self.assertEqual(len(modes), 4)

    def test_dataset_inspect_and_intelligent_target_configuration(self):
        insp = self.client.get("/api/datasets/housing_synthetic/inspect").json()
        self.assertIn("price_k", insp["columns"])
        self.assertIn("neighborhood", insp["columns"])
        self.assertEqual(insp["detected_task_type"], "regression")

        cfg_res = self.client.post(
            "/api/datasets/housing_synthetic/configure-target",
            json={"target_column": "neighborhood", "task_type": "auto"},
        )
        self.assertEqual(cfg_res.status_code, 200)
        self.assertEqual(cfg_res.json()["resolved_task_type"], "classification")

        cfg_res2 = self.client.post(
            "/api/datasets/housing_synthetic/configure-target",
            json={"target_column": "price_k", "task_type": "auto"},
        )
        self.assertEqual(cfg_res2.status_code, 200)
        self.assertEqual(cfg_res2.json()["resolved_task_type"], "regression")

    def test_auto_selector_250_clean_sampling_and_metrics(self):
        cfg = ExperimentConfig(dataset_id="iris", sample_size=250, random_state=42)
        result = run_experiment(cfg)
        self.assertGreater(result.sampling_summary.nan_rows_removed, 0)
        self.assertEqual(result.sampling_summary.sampled_rows, 250)
        self.assertEqual(result.sampling_summary.detected_task_type, "classification")
        top_metrics = result.leaderboard[0].metrics
        for key in ("Accuracy", "Precision", "Recall", "F1 Score", "MAE (Class Error)", "MSE (Class Error)"):
            self.assertIn(key, top_metrics)

    def test_manual_learning_playground_and_hyperparameter_sweep(self):
        man_cfg = ManualPlaygroundConfig(
            dataset_id="wine",
            target_column="target",
            task_type="auto",
            algorithm_id="knn_clf",
            sample_size=250,
            scaling="standard",
            n_neighbors=7,
            knn_weights="distance",
        )
        res = run_manual_playground(man_cfg)
        self.assertEqual(res.algorithm_id, "knn_clf")
        self.assertEqual(res.sampling_summary.sampled_rows, 250)
        self.assertIsNotNone(res.confusion_matrix)

        sweep_res = self.client.post(
            "/api/playground/hyperparameter-sweep",
            json={"playground_config": _model_to_dict(man_cfg), "param_name": "n_neighbors", "param_values": [1, 5, 11]},
        )
        self.assertEqual(sweep_res.status_code, 200)
        sweep_data = sweep_res.json()
        self.assertEqual(len(sweep_data["curve"]), 3)

    def test_custom_csv_upload_dropdown_and_auto_detection(self):
        rows = ["age,income,credit_score,approved_status"]
        for i in range(300):
            if i % 15 == 0:
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
        self.assertEqual(up_data["columns"], ["age", "income", "credit_score", "approved_status"])
        self.assertEqual(up_data["detected_task_type"], "classification")

        custom_id = up_data["id"]
        retarget = self.client.post(
            f"/api/datasets/{custom_id}/configure-target",
            json={"target_column": "income", "task_type": "auto"},
        )
        self.assertEqual(retarget.status_code, 200)
        self.assertEqual(retarget.json()["resolved_task_type"], "regression")


if __name__ == "__main__":
    unittest.main()
