from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: List[Any]
    recommended_task_if_target: str
    task_detection_reason: str


class DatasetInspectionResponse(BaseModel):
    dataset_id: str
    name: str
    total_rows: int
    clean_non_nan_rows: int
    nan_rows_count: int
    columns: List[str]
    default_target: str
    detected_task_type: str
    task_detection_reason: str
    column_profiles: List[ColumnProfile]


class TargetConfigurationRequest(BaseModel):
    target_column: str = Field(..., description="Target column selected from the dataset column dropdown")
    task_type: Optional[str] = Field(
        default="auto",
        description="Optional user override: 'auto' (intelligent detection), 'classification', or 'regression'",
    )


class TargetConfigurationResponse(BaseModel):
    dataset_id: str
    target_column: str
    user_selected_task: str
    resolved_task_type: str
    intelligent_detection_reason: str
    available_feature_columns: List[str]
    target_summary: Dict[str, Any]


class ExperimentConfig(BaseModel):
    dataset_id: str = Field(default="iris")
    target_column: Optional[str] = Field(default=None)
    task_type: Optional[str] = Field(default="auto")
    sample_size: int = Field(default=250, ge=20, le=2000)
    random_state: int = Field(default=42)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    scaling: str = Field(default="standard")
    polynomial_degree: int = Field(default=1, ge=1, le=2)
    feature_selection_k: Optional[int] = Field(default=None)
    regularization_c_or_alpha: float = Field(default=1.0, gt=0.0)
    max_depth: Optional[int] = Field(default=5, ge=1, le=20)
    n_neighbors: int = Field(default=5, ge=1, le=30)


class ManualPlaygroundConfig(BaseModel):
    dataset_id: str = Field(default="iris")
    target_column: Optional[str] = Field(default=None)
    task_type: Optional[str] = Field(default="auto")
    algorithm_id: str = Field(
        default="rf_clf",
        description="Specific algorithm ID chosen by user in the Manual Playground",
    )
    sample_size: int = Field(default=250, ge=20, le=2000)
    nan_strategy: str = Field(
        default="drop",
        description="'drop' (strictly drop NaN rows) or 'median_mode' (impute missing values)",
    )
    random_state: int = Field(default=42)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    scaling: str = Field(default="standard", description="'none', 'standard', 'minmax', or 'robust'")
    polynomial_degree: int = Field(default=1, ge=1, le=2)
    excluded_features: List[str] = Field(default_factory=list, description="Features manually toggled off by user")
    feature_selection_k: Optional[int] = Field(default=None)
    # Hyperparameters for manual tweaking
    regularization: float = Field(default=1.0, gt=0.001, le=100.0)
    max_depth: int = Field(default=5, ge=1, le=25)
    min_samples_split: int = Field(default=2, ge=2, le=50)
    n_estimators: int = Field(default=30, ge=5, le=150)
    learning_rate: float = Field(default=0.15, gt=0.01, le=1.0)
    n_neighbors: int = Field(default=5, ge=1, le=35)
    knn_weights: str = Field(default="uniform", description="'uniform' or 'distance'")
    kernel_gamma: float = Field(default=0.2, gt=0.001, le=10.0)
    add_target_noise: float = Field(
        default=0.0, ge=0.0, le=0.5, description="Simulated noise level to test algorithm robustness"
    )


class HyperparameterSweepRequest(BaseModel):
    playground_config: ManualPlaygroundConfig
    param_name: str = Field(
        default="max_depth",
        description="Hyperparameter to sweep: 'max_depth', 'n_neighbors', 'regularization', 'n_estimators', 'learning_rate'",
    )
    param_values: Optional[List[float]] = Field(default=None)


class ModelResult(BaseModel):
    algorithm: str
    complexity_tier: str
    tier_order: int
    why_this_tier: str
    train_score: float
    test_score: float
    cv_mean: float
    cv_std: float
    overfitting_gap: float
    bias_variance_diagnosis: str
    metrics: Dict[str, float]
    fit_time_ms: float


class SamplingSummary(BaseModel):
    original_rows: int
    rows_after_nan_drop: int
    nan_rows_removed: int
    sampled_rows: int
    target_sample_size: int
    train_rows: int
    test_rows: int
    numeric_features: List[str]
    categorical_features: List[str]
    target_column: str
    detected_task_type: str
    task_detection_reason: str = ""


class ExperimentResponse(BaseModel):
    sampling_summary: SamplingSummary
    primary_metric: str
    best_algorithm: str
    best_score: float
    selection_rationale: str
    progression_narrative: List[str]
    leaderboard: List[ModelResult]
    feature_importances: Dict[str, float]
    concept_takeaways: Dict[str, str]
    sample_preview: List[Dict[str, Any]]


class ManualPlaygroundResponse(BaseModel):
    run_id: str
    algorithm_id: str
    algorithm_name: str
    complexity_tier: str
    sampling_summary: SamplingSummary
    applied_hyperparameters: Dict[str, Any]
    primary_metric: str
    train_score: float
    test_score: float
    cv_mean: float
    cv_std: float
    overfitting_gap: float
    bias_variance_diagnosis: str
    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    confusion_matrix: Optional[Dict[str, Any]] = None
    prediction_diagnostics: List[Dict[str, Any]]
    coaching_tips: List[str]
    fit_time_ms: float
