import io
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from concepts import ALGORITHMS_CATALOG, ML_CONCEPTS_LIBRARY, WORKFLOW_MODES
from datasets import (
    configure_dataset_target,
    delete_custom_dataset,
    export_processed_csv_bytes,
    inspect_dataset_columns,
    list_available_datasets,
    load_dataset,
    recommend_nan_strategy,
    register_uploaded_csv,
)
from pipeline import (
    prepare_and_store_dataset,
    run_experiment,
    run_hyperparameter_sweep,
    run_manual_playground,
)
from schemas import (
    DatasetInspectionResponse,
    ExperimentConfig,
    ExperimentResponse,
    HyperparameterSweepRequest,
    ManualPlaygroundConfig,
    ManualPlaygroundResponse,
    TargetConfigurationRequest,
    TargetConfigurationResponse,
)

app = FastAPI(
    title="ML Studio API",
    description="Machine Learning Algorithm Selector, Dataset Preprocessor & Manual Tuning Playground.",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_JOBS: Dict[str, Dict[str, Any]] = {}


def _to_dict(obj) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


# --- Core User-Facing Endpoints ---
@app.get("/api/datasets", tags=["Datasets"])
def get_datasets() -> dict:
    return {"datasets": list_available_datasets()}


@app.post("/api/datasets/upload", tags=["Datasets"])
async def upload_custom_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(default=None),
    task_type: str = Form(default="auto"),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid .csv file.")
    content = await file.read()
    try:
        return register_uploaded_csv(
            csv_bytes=content,
            filename=file.filename,
            target_column=target_column,
            task_type=task_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/prepare", tags=["Datasets"])
def prepare_dataset_for_review_and_download(
    dataset_id: str,
    target_column: Optional[str] = Query(default=None),
    task_type: str = Query(default="auto"),
    nan_strategy: str = Query(default="auto"),
    sample_percentage: float = Query(default=60.0, ge=20.0, le=80.0),
    random_state: int = Query(default=42),
) -> dict:
    """
    Applies the selected NaN handling strategy first, then samples 20%-80% of the clean rows,
    and saves the processed dataset so the user can review and download it as CSV.
    """
    try:
        return prepare_and_store_dataset(
            dataset_id=dataset_id,
            target_column=target_column,
            task_type=task_type,
            nan_strategy=nan_strategy,
            sample_percentage=sample_percentage,
            random_state=random_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/download-processed", tags=["Datasets"])
def download_processed_dataset_csv(dataset_id: str) -> StreamingResponse:
    """Downloads the processed (NaN-handled + 20%-80% sampled) dataset as a clean CSV file."""
    try:
        csv_bytes, filename = export_processed_csv_bytes(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/auto-selector/run", response_model=ExperimentResponse, tags=["ML Benchmark & Playground"])
def execute_auto_selector(config: ExperimentConfig) -> ExperimentResponse:
    try:
        return run_experiment(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto-selector execution failed: {exc}") from exc


@app.post("/api/playground/train", response_model=ManualPlaygroundResponse, tags=["ML Benchmark & Playground"])
def train_manual_playground_model(config: ManualPlaygroundConfig) -> ManualPlaygroundResponse:
    try:
        return run_manual_playground(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Manual playground run failed: {exc}") from exc


# --- Internal UI Helper Endpoints (Hidden from Public Schema) ---
@app.get("/api/health", include_in_schema=False)
def health_check() -> dict:
    return {"status": "ok", "version": "2.2.0"}


@app.get("/api/modes", include_in_schema=False)
def get_workflow_modes() -> dict:
    return {"modes": WORKFLOW_MODES}


@app.get("/api/datasets/{dataset_id}/inspect", response_model=DatasetInspectionResponse, include_in_schema=False)
def inspect_dataset(
    dataset_id: str,
    target_column: Optional[str] = Query(default=None),
    task_type: str = Query(default="auto"),
) -> DatasetInspectionResponse:
    try:
        return inspect_dataset_columns(dataset_id, target_column, task_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/recommend-nan-strategy", include_in_schema=False)
def get_nan_strategy_recommendation(
    dataset_id: str,
    target_column: Optional[str] = Query(default=None),
) -> dict:
    try:
        df, default_target, _ = load_dataset(dataset_id)
        t_col = target_column if (target_column and target_column in df.columns) else default_target
        strategy, reason, nan_pct = recommend_nan_strategy(df, t_col)
        return {
            "dataset_id": dataset_id,
            "target_column": t_col,
            "nan_row_pct": nan_pct,
            "recommended_nan_strategy": strategy,
            "reason": reason,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/configure-target", response_model=TargetConfigurationResponse, include_in_schema=False)
def set_dataset_target_and_task(
    dataset_id: str,
    payload: TargetConfigurationRequest,
) -> TargetConfigurationResponse:
    try:
        return configure_dataset_target(dataset_id, payload.target_column, payload.task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/datasets/{dataset_id}", include_in_schema=False)
def remove_custom_dataset(dataset_id: str) -> dict:
    deleted = delete_custom_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Custom dataset ID not found.")
    return {"status": "deleted", "dataset_id": dataset_id}


@app.get("/api/algorithms", include_in_schema=False)
def list_algorithms(task_type: Optional[str] = Query(default=None)) -> dict:
    if task_type in ("classification", "regression"):
        return {"algorithms": [a for a in ALGORITHMS_CATALOG if a["task_type"] == task_type]}
    return {"algorithms": ALGORITHMS_CATALOG}


@app.post("/api/auto-selector/start", include_in_schema=False)
def start_auto_selector_job(config: ExperimentConfig) -> dict:
    """Starts the progressive benchmark in a background thread so the UI can display live progress & ETA."""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    BENCHMARK_JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "completed_steps": 0,
        "total_steps": 7,
        "progress_pct": 0.0,
        "current_message": "Initializing dataset & NaN preprocessing...",
        "elapsed_seconds": 0.0,
        "eta_seconds": 2.5,
        "result": None,
        "error": None,
    }

    def _on_progress(done: int, total: int, msg: str, elapsed: float, eta: float):
        BENCHMARK_JOBS[job_id].update(
            {
                "completed_steps": done,
                "total_steps": total,
                "progress_pct": round((done / max(1, total)) * 100.0, 1),
                "current_message": msg,
                "elapsed_seconds": round(elapsed, 2),
                "eta_seconds": round(eta, 2),
            }
        )

    def _worker():
        try:
            res = run_experiment(config, progress_callback=_on_progress)
            BENCHMARK_JOBS[job_id].update(
                {
                    "status": "completed",
                    "progress_pct": 100.0,
                    "eta_seconds": 0.0,
                    "current_message": f"Completed! Best algorithm: {res.best_algorithm}",
                    "result": _to_dict(res),
                }
            )
        except Exception as exc:
            BENCHMARK_JOBS[job_id].update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "eta_seconds": 0.0,
                }
            )

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/auto-selector/status/{job_id}", include_in_schema=False)
def get_auto_selector_job_status(job_id: str) -> dict:
    job = BENCHMARK_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Benchmark job not found.")
    return job


@app.post("/api/playground/hyperparameter-sweep", include_in_schema=False)
def execute_hyperparameter_sweep(req: HyperparameterSweepRequest) -> dict:
    try:
        return run_hyperparameter_sweep(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/concepts", include_in_schema=False)
def get_ml_concepts() -> dict:
    return {"concepts": ML_CONCEPTS_LIBRARY}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/auto-selector", response_class=HTMLResponse, include_in_schema=False)
@app.get("/manual-playground", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dataset-studio", response_class=HTMLResponse, include_in_schema=False)
@app.get("/concepts-lab", response_class=HTMLResponse, include_in_schema=False)
def serve_playground_app() -> HTMLResponse:
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>ML Studio is running.</h1>")