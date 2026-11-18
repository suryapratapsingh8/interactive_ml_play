from pathlib import Path
from typing import Optional
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
    recommend_nan_strategy,
    register_uploaded_csv,
)
from pipeline import (
    run_experiment,
    run_hyperparameter_sweep,
    run_manual_playground,
    sample_percentage_rows,
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
        "(2) Dataset Studio with Target Column Dropdown, Intelligent Task-Type Detection & Proactive NaN Strategy Suggestion, "
        "(3) Auto Algorithm Selector (20%–80% Sampling, Tier 0 → Tier 6), and "
        "(4) Manual Hyperparameter Tuning & Feature Tweaking Playground."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


def _to_dict(obj) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


# --- 1. System & Mode Discovery Endpoints ---
@app.get("/api/health", tags=["System & Navigation"])
def health_check() -> dict:
    return {
        "status": "ok",
        "version": "2.1.0",
        "service": "ML Algorithm Selector & Manual Learning Playground",
        "total_endpoints": 17,
    }


@app.get("/api/modes", tags=["System & Navigation"])
def get_workflow_modes() -> dict:
    return {"modes": WORKFLOW_MODES}


# --- 2. Dataset Studio, Upload, Target Dropdown & Proactive NaN Recommendation Endpoints ---
@app.get("/api/datasets", tags=["Dataset Studio & Target Detection"])
def get_datasets() -> dict:
    return {"datasets": list_available_datasets()}


@app.post("/api/datasets/upload", tags=["Dataset Studio & Target Detection"])
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


@app.get(
    "/api/datasets/{dataset_id}/inspect",
    response_model=DatasetInspectionResponse,
    tags=["Dataset Studio & Target Detection"],
)
def inspect_dataset(
    dataset_id: str,
    target_column: Optional[str] = Query(default=None),
    task_type: str = Query(default="auto"),
) -> DatasetInspectionResponse:
    try:
        return inspect_dataset_columns(dataset_id, target_column, task_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/recommend-nan-strategy", tags=["Dataset Studio & Target Detection"])
def get_nan_strategy_recommendation(
    dataset_id: str,
    target_column: Optional[str] = Query(default=None),
) -> dict:
    """Proactively analyzes dataset missingness & skewness to suggest the optimal NaN strategy before user selection."""
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
            "available_strategies": [
                {"id": "auto", "label": f"Auto (Use Suggested: {strategy})"},
                {"id": "drop", "label": "Remove / Drop NaN Rows"},
                {"id": "median_mode", "label": "Median (Numeric) / Mode (Categorical) Imputer"},
                {"id": "mean_mode", "label": "Mean (Numeric) / Mode (Categorical) Imputer"},
                {"id": "knn_impute", "label": "KNN Imputer (Distance-Weighted Neighbors)"},
                {"id": "ffill_bfill", "label": "Forward Fill / Backward Fill (Sequential)"},
                {"id": "constant_zero", "label": "Constant Fill (0 / 'Missing')"},
            ],
        }
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
        "nan_row_pct": info.nan_row_pct,
        "recommended_nan_strategy": info.recommended_nan_strategy,
        "nan_recommendation_reason": info.nan_recommendation_reason,
        "columns": info.columns,
        "default_target": info.default_target,
        "default_task_type": info.detected_task_type,
        "task_detection_reason": info.task_detection_reason,
        "column_profiles": [_to_dict(p) for p in info.column_profiles],
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
    try:
        return configure_dataset_target(dataset_id, payload.target_column, payload.task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/sample-preview", tags=["Dataset Studio & Target Detection"])
def preview_percentage_sample(
    dataset_id: str,
    target_column: Optional[str] = Query(default=None),
    sample_percentage: float = Query(default=60.0, ge=20.0, le=80.0),
    nan_strategy: str = Query(default="auto"),
    random_state: int = Query(default=42),
) -> dict:
    try:
        df, default_target, _ = load_dataset(dataset_id)
        t_col = target_column if (target_column and target_column in df.columns) else default_target
        info = configure_dataset_target(dataset_id, t_col, "auto")
        (
            sampled_df,
            orig,
            post_nan,
            raw_nans,
            nan_rem,
            applied_strat,
            rec_reason,
        ) = sample_percentage_rows(
            df, t_col, sample_percentage, random_state, info.resolved_task_type, nan_strategy
        )
        return {
            "dataset_id": dataset_id,
            "target_column": t_col,
            "detected_task_type": info.resolved_task_type,
            "detection_reason": info.intelligent_detection_reason,
            "original_rows": orig,
            "rows_after_nan_handling": post_nan,
            "raw_nan_rows": raw_nans,
            "nan_rows_removed": nan_rem,
            "nan_strategy_applied": applied_strat,
            "nan_recommendation_reason": rec_reason,
            "sample_percentage_used": sample_percentage,
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
    try:
        return run_manual_playground(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Manual playground run failed: {exc}") from exc


@app.post("/api/playground/hyperparameter-sweep", tags=["Manual Learning Playground"])
def execute_hyperparameter_sweep(req: HyperparameterSweepRequest) -> dict:
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