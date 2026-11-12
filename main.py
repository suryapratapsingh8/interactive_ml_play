from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from concepts import ALGORITHMS_CATALOG, ML_CONCEPTS_LIBRARY, WORKFLOW_MODES
from datasets import (
    configure_dataset_target,
    delete_custom_dataset,
    inspect_dataset_columns,
    list_available_datasets,
    load_dataset,
    register_uploaded_csv,
)
from pipeline import (
    run_experiment,
    run_hyperparameter_sweep,
    run_manual_playground,
    sample_clean_250_rows,
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
    title="ML Algorithm Selector & Interactive Manual Learning Playground API",
    description=(
        "Production-grade FastAPI platform featuring: (1) Workflow Mode Selector, "
        "(2) Dataset Studio with Target Column Dropdown & Intelligent Task-Type Detection, "
        "(3) Auto Algorithm Selector (~250 clean non-NaN rows, Tier 0 → Tier 6), and "
        "(4) Manual Hyperparameter Tuning & Feature Tweaking Playground."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


# --- 1. System & Mode Discovery Endpoints ---
@app.get("/api/health", tags=["System & Navigation"])
def health_check() -> dict:
    return {
        "status": "ok",
        "version": "2.0.0",
        "service": "ML Algorithm Selector & Manual Learning Playground",
        "total_endpoints": 16,
    }


@app.get("/api/modes", tags=["System & Navigation"])
def get_workflow_modes() -> dict:
    """Returns the interactive workflow modes shown on the initial 'What do you want to do?' selector."""
    return {"modes": WORKFLOW_MODES}


# --- 2. Dataset Studio, Upload, Target Column Dropdown & Intelligent Task Detection Endpoints ---
@app.get("/api/datasets", tags=["Dataset Studio & Target Detection"])
def get_datasets() -> dict:
    return {"datasets": list_available_datasets()}


@app.post("/api/datasets/upload", tags=["Dataset Studio & Target Detection"])
async def upload_custom_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(default=None),
    task_type: str = Form(default="auto"),
) -> dict:
    """
    Uploads a custom CSV dataset, extracts all column names for the Target Column dropdown,
    and runs intelligent task detection (Classification vs. Regression) if task_type is 'auto'.
    """
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


@app.get(
    "/api/datasets/{dataset_id}/inspect",
    response_model=DatasetInspectionResponse,
    tags=["Dataset Studio & Target Detection"],
)
def inspect_dataset(
    dataset_id: str,
    target_column: Optional[str] = Query(default=None, description="Optional target column to inspect"),
    task_type: str = Query(default="auto", description="'auto', 'classification', or 'regression'"),
) -> DatasetInspectionResponse:
    """Returns full column dropdown list, missing value stats, and intelligent task-type recommendation."""
    try:
        return inspect_dataset_columns(dataset_id, target_column, task_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/columns", tags=["Dataset Studio & Target Detection"])
def get_dataset_columns(dataset_id: str) -> dict:
    try:
        info = inspect_dataset_columns(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "dataset_id": dataset_id,
        "total_rows": info.total_rows,
        "clean_rows": info.clean_non_nan_rows,
        "nan_rows": info.nan_rows_count,
        "columns": info.columns,
        "default_target": info.default_target,
        "default_task_type": info.detected_task_type,
        "task_detection_reason": info.task_detection_reason,
        "column_profiles": [(p.model_dump() if hasattr(p, "model_dump") else p.dict()) for p in info.column_profiles],
    }


@app.post(
    "/api/datasets/{dataset_id}/configure-target",
    response_model=TargetConfigurationResponse,
    tags=["Dataset Studio & Target Detection"],
)
def set_dataset_target_and_task(
    dataset_id: str,
    payload: TargetConfigurationRequest,
) -> TargetConfigurationResponse:
    """
    Sets the target column from the user's dropdown selection and either applies the user's explicit
    model type ('classification'/'regression') or intelligently detects the task type ('auto').
    """
    try:
        return configure_dataset_target(dataset_id, payload.target_column, payload.task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/sample-preview", tags=["Dataset Studio & Target Detection"])
def preview_clean_250_sample(
    dataset_id: str,
    target_column: Optional[str] = Query(default=None),
    sample_size: int = Query(default=250, ge=20, le=2000),
    nan_strategy: str = Query(default="drop"),
    random_state: int = Query(default=42),
) -> dict:
    """Runs the NaN handler and ~250-row random/stratified sampler and returns a preview of the sampled rows."""
    try:
        df, default_target, _ = load_dataset(dataset_id)
        t_col = target_column if (target_column and target_column in df.columns) else default_target
        info = configure_dataset_target(dataset_id, t_col, "auto")
        sampled_df, orig, clean, nan_rem = sample_clean_250_rows(
            df, t_col, sample_size, random_state, info.resolved_task_type, nan_strategy
        )
        return {
            "dataset_id": dataset_id,
            "target_column": t_col,
            "detected_task_type": info.resolved_task_type,
            "detection_reason": info.intelligent_detection_reason,
            "original_rows": orig,
            "clean_rows_available": clean,
            "nan_rows_handled": nan_rem,
            "sampled_rows_count": len(sampled_df),
            "sample_rows": sampled_df.head(12).to_dict(orient="records"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/datasets/{dataset_id}", tags=["Dataset Studio & Target Detection"])
def remove_custom_dataset(dataset_id: str) -> dict:
    deleted = delete_custom_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Custom dataset ID not found or is a protected built-in dataset.")
    return {"status": "deleted", "dataset_id": dataset_id}


# --- 3. Algorithm Catalog & Hyperparameter Schema Endpoints ---
@app.get("/api/algorithms", tags=["Algorithms & Hyperparameter Schemas"])
def list_algorithms(task_type: Optional[str] = Query(default=None)) -> dict:
    """Lists all algorithms and their tunable hyperparameter controls for the Manual Learning Playground."""
    if task_type in ("classification", "regression"):
        filtered = [a for a in ALGORITHMS_CATALOG if a["task_type"] == task_type]
        return {"algorithms": filtered}
    return {"algorithms": ALGORITHMS_CATALOG}


@app.get("/api/algorithms/{algorithm_id}", tags=["Algorithms & Hyperparameter Schemas"])
def get_algorithm_details(algorithm_id: str) -> dict:
    for alg in ALGORITHMS_CATALOG:
        if alg["id"] == algorithm_id:
            return alg
    raise HTTPException(status_code=404, detail=f"Algorithm '{algorithm_id}' not found.")


# --- 4. Auto Algorithm Selector Endpoints ---
@app.post(
    "/api/auto-selector/run",
    response_model=ExperimentResponse,
    tags=["Auto Algorithm Selector"],
)
@app.post(
    "/api/experiment",
    response_model=ExperimentResponse,
    tags=["Auto Algorithm Selector"],
)
def execute_auto_selector(config: ExperimentConfig) -> ExperimentResponse:
    """
    Drops NaN rows, randomly samples ~250 clean rows, and benchmarks algorithms progressively
    from Tier 0 (Naive Baseline) to Tier 6 (Gradient Boosting) to determine the best algorithm.
    """
    try:
        return run_experiment(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto-selector execution failed: {exc}") from exc


# --- 5. Manual Learning & Hyperparameter Tuning Playground Endpoints ---
@app.post(
    "/api/playground/train",
    response_model=ManualPlaygroundResponse,
    tags=["Manual Learning Playground"],
)
def train_manual_playground_model(config: ManualPlaygroundConfig) -> ManualPlaygroundResponse:
    """
    Trains a single user-selected algorithm with custom hyperparameters, feature exclusions,
    scaling, polynomial degree, and noise injection for hands-on learning.
    """
    try:
        return run_manual_playground(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Manual playground run failed: {exc}") from exc


@app.post("/api/playground/hyperparameter-sweep", tags=["Manual Learning Playground"])
def execute_hyperparameter_sweep(req: HyperparameterSweepRequest) -> dict:
    """
    Runs a 1D validation curve sweep across a chosen hyperparameter (e.g., max_depth, n_neighbors,
    regularization, n_estimators, learning_rate) to visualize underfitting vs. overfitting.
    """
    try:
        return run_hyperparameter_sweep(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- 6. ML Concepts & Guided Presets Endpoints ---
@app.get("/api/concepts", tags=["ML Concepts Academy"])
def get_ml_concepts() -> dict:
    return {"concepts": ML_CONCEPTS_LIBRARY}


@app.get("/api/concepts/{concept_id}", tags=["ML Concepts Academy"])
def get_single_concept(concept_id: str) -> dict:
    for c in ML_CONCEPTS_LIBRARY:
        if c["id"] == concept_id:
            return c
    raise HTTPException(status_code=404, detail=f"Concept '{concept_id}' not found.")


# --- 7. Interactive Multi-View Frontend Routes ---
@app.get("/", response_class=HTMLResponse, tags=["UI Pages"])
@app.get("/auto-selector", response_class=HTMLResponse, tags=["UI Pages"])
@app.get("/manual-playground", response_class=HTMLResponse, tags=["UI Pages"])
@app.get("/dataset-studio", response_class=HTMLResponse, tags=["UI Pages"])
@app.get("/concepts-lab", response_class=HTMLResponse, tags=["UI Pages"])
def serve_playground_app() -> HTMLResponse:
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>ML Playground API is running. Visit /docs for OpenAPI UI.</h1>")
