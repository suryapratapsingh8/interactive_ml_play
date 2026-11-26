import io
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from schemas import ColumnProfile, DatasetInspectionResponse, TargetConfigurationResponse

try:
    from sklearn import datasets as sk_datasets
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

CUSTOM_DATASETS: Dict[str, Dict[str, Any]] = {}
PROCESSED_DATASETS: Dict[str, Dict[str, Any]] = {}


def _to_dict(model_obj: Any) -> dict:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


BUILTIN_CATALOG = {
    "iris": {
        "id": "iris",
        "name": "Iris Flower Species (Botanical Classification)",
        "task_type": "classification",
        "target_column": "target",
        "description": "3-class botanical classification (Setosa, Versicolor, Virginica) with 4 sepal/petal features and injected NaN rows.",
    },
    "wine": {
        "id": "wine",
        "name": "Wine Chemical Cultivars (Multi-Class Classification)",
        "task_type": "classification",
        "target_column": "target",
        "description": "Chemical constituents of wines across 3 cultivars with missing values for testing NaN strategies.",
    },
    "breast_cancer": {
        "id": "breast_cancer",
        "name": "Wisconsin Breast Cancer Diagnostic (Binary Classification)",
        "task_type": "classification",
        "target_column": "target",
        "description": "Biopsy dataset classifying benign vs. malignant tumors across cell-nuclei geometry features.",
    },
    "diabetes": {
        "id": "diabetes",
        "name": "Diabetes Disease Progression (Clinical Regression)",
        "task_type": "regression",
        "target_column": "target",
        "description": "Baseline physiological variables predicting quantitative disease progression 1 year later.",
    },
    "housing_synthetic": {
        "id": "housing_synthetic",
        "name": "Residential Property Valuation (Non-Linear Regression)",
        "task_type": "regression",
        "target_column": "price_k",
        "description": "450 residential properties with square footage, bedrooms, age, transit distance, school score, neighborhood category, and missing records.",
    },
}


def save_processed_dataset(dataset_id: str, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Stores the processed (NaN-handled + percentage-sampled) dataset for review and CSV download."""
    PROCESSED_DATASETS[dataset_id] = {
        "df": df.copy(),
        "metadata": metadata,
    }
    return metadata


def get_processed_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    return PROCESSED_DATASETS.get(dataset_id)


def export_processed_csv_bytes(dataset_id: str) -> Tuple[bytes, str]:
    entry = PROCESSED_DATASETS.get(dataset_id)
    if not entry:
        raise ValueError(f"No processed dataset found for '{dataset_id}'. Apply NaN handling and sampling first.")
    df: pd.DataFrame = entry["df"]
    meta = entry["metadata"]
    csv_text = df.to_csv(index=False)
    strat = meta.get("nan_strategy_applied", "clean")
    pct = int(round(meta.get("sample_percentage_used", 60)))
    filename = f"{dataset_id}_{strat}_{pct}pct.csv"
    return csv_text.encode("utf-8"), filename


def recommend_nan_strategy(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[str, str, float]:
    """
    Analyzes missingness rate, dataset size, and feature skewness to proactively suggest
    the best NaN handling strategy before the user selects one.
    """
    total_rows = len(df)
    if total_rows == 0:
        return "drop", "Dataset is empty.", 0.0

    clean_rows = int(df.dropna(axis=0, how="any").shape[0])
    nan_rows = total_rows - clean_rows
    nan_pct = round((nan_rows / total_rows) * 100.0, 2)

    if nan_rows == 0:
        return (
            "drop",
            "Suggested: Remove / Drop NaN Rows — Your dataset has 0 missing rows (0.0%), so no imputation is needed.",
            0.0,
        )

    feature_cols = [c for c in df.columns if c != target_col]
    num_cols_with_nan = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any()
    ]
    cat_cols_with_nan = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any()
    ]

    max_skew = 0.0
    for c in num_cols_with_nan:
        s = df[c].dropna()
        if len(s) >= 5 and s.std() > 1e-9:
            max_skew = max(max_skew, abs(float(s.skew())))

    if nan_pct <= 5.0 and clean_rows >= 150:
        return (
            "drop",
            f"Suggested: Remove / Drop NaN Rows — Only {nan_pct}% of rows ({nan_rows}/{total_rows}) contain NaNs and {clean_rows} complete rows remain, so dropping avoids artificial imputation bias.",
            nan_pct,
        )

    if nan_pct > 20.0:
        if len(num_cols_with_nan) >= 2:
            return (
                "knn_impute",
                f"Suggested: KNN Imputer — High missingness ({nan_pct}% of rows contain NaNs). Dropping would lose too much signal; KNN Imputer reconstructs missing values from similar neighbors.",
                nan_pct,
            )
        return (
            "median_mode",
            f"Suggested: Median / Mode Imputer — High missingness ({nan_pct}% of rows). Median/Mode imputation preserves sample size cleanly.",
            nan_pct,
        )

    if max_skew > 1.0 or len(cat_cols_with_nan) > 0:
        return (
            "median_mode",
            f"Suggested: Median / Mode Imputer — {nan_pct}% of rows ({nan_rows}/{total_rows}) have NaNs with skewed or mixed categorical features (max skewness {max_skew:.2f}); Median/Mode resists outliers while retaining all rows.",
            nan_pct,
        )

    if len(num_cols_with_nan) >= 2:
        return (
            "knn_impute",
            f"Suggested: KNN Imputer — {nan_pct}% of rows ({nan_rows}/{total_rows}) have NaNs across numeric features; KNN Imputer leverages feature correlations better than simple averages.",
            nan_pct,
        )

    return (
        "mean_mode",
        f"Suggested: Mean / Mode Imputer — {nan_pct}% of rows have NaNs in symmetric numeric features (max skewness {max_skew:.2f}); Mean/Mode imputation preserves the distribution center.",
        nan_pct,
    )


def intelligent_detect_task_type(series: pd.Series, requested: Optional[str] = "auto") -> Tuple[str, str]:
    if requested and requested.lower() in ("classification", "regression"):
        mode = requested.lower()
        return mode, f"Explicitly set to '{mode}' by user selection."

    clean_s = series.dropna()
    n_total = len(clean_s)
    if n_total == 0:
        return "classification", "Column is empty after dropping NaNs; defaulting to classification."

    n_unique = int(clean_s.nunique())
    unique_ratio = n_unique / max(1, n_total)

    if (
        pd.api.types.is_bool_dtype(clean_s)
        or pd.api.types.is_object_dtype(clean_s)
        or isinstance(clean_s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(clean_s)
    ):
        return (
            "classification",
            f"Auto-detected Classification: Target '{series.name}' has non-numeric/categorical dtype ({clean_s.dtype}) with {n_unique} distinct classes.",
        )

    numeric_vals = pd.to_numeric(clean_s, errors="coerce").dropna().to_numpy()
    all_integers = bool(np.all(np.equal(np.mod(numeric_vals, 1), 0)))

    if n_unique == 2:
        return (
            "classification",
            f"Auto-detected Binary Classification: Target '{series.name}' contains exactly 2 distinct values ({list(clean_s.unique()[:2])}).",
        )

    if all_integers and n_unique <= 12:
        return (
            "classification",
            f"Auto-detected Multi-Class Classification: Target '{series.name}' contains {n_unique} discrete integer class labels (cardinality ratio {unique_ratio:.1%}).",
        )

    if all_integers and unique_ratio < 0.05 and n_unique <= 20:
        return (
            "classification",
            f"Auto-detected Classification: Low cardinality integer target ({n_unique} unique values across {n_total} rows, ratio {unique_ratio:.1%}).",
        )

    return (
        "regression",
        f"Auto-detected Regression: Target '{series.name}' is continuous/high-cardinality numeric ({n_unique} unique values across {n_total} clean rows, range [{numeric_vals.min():.2f}, {numeric_vals.max():.2f}]).",
    )


def _inject_nans_and_ensure_size(
    df: pd.DataFrame, target_col: str, min_clean_rows: int = 320, nan_rate: float = 0.08
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_df = df.copy()
    if len(base_df) < min_clean_rows:
        needed = min_clean_rows - len(base_df) + 40
        sampled = base_df.sample(n=needed, replace=True, random_state=42).copy()
        num_cols = [c for c in sampled.columns if c != target_col and pd.api.types.is_numeric_dtype(sampled[c])]
        for col in num_cols:
            std = float(base_df[col].std() or 1.0)
            sampled[col] = sampled[col] + rng.normal(0, 0.03 * std, size=len(sampled))
        base_df = pd.concat([base_df, sampled], ignore_index=True)

    n_nan_rows = max(15, int(len(base_df) * nan_rate))
    nan_indices = rng.choice(len(base_df), size=n_nan_rows, replace=False)
    feature_cols = [c for c in base_df.columns if c != target_col]
    for idx in nan_indices:
        col = feature_cols[int(rng.integers(0, len(feature_cols)))]
        base_df.loc[int(idx), col] = np.nan

    return base_df


def _generate_fallback_dataset(dataset_id: str) -> Tuple[pd.DataFrame, str, str]:
    rng = np.random.default_rng(42)
    if dataset_id == "iris":
        n_per_class = 115
        rows = []
        specs = [
            (0, [5.0, 3.4, 1.5, 0.25], [0.35, 0.38, 0.17, 0.10]),
            (1, [5.9, 2.77, 4.26, 1.33], [0.51, 0.31, 0.47, 0.20]),
            (2, [6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
        ]
        for label, means, stds in specs:
            feats = rng.normal(means, stds, size=(n_per_class, 4))
            for r in feats:
                rows.append(
                    {
                        "sepal length (cm)": round(float(r[0]), 2),
                        "sepal width (cm)": round(float(r[1]), 2),
                        "petal length (cm)": round(float(r[2]), 2),
                        "petal width (cm)": round(float(r[3]), 2),
                        "target": label,
                    }
                )
        df = pd.DataFrame(rows)
        return _inject_nans_and_ensure_size(df, "target", 320, nan_rate=0.045), "target", "classification"

    if dataset_id == "wine":
        n = 360
        cultivar = rng.choice([0, 1, 2], size=n, p=[0.33, 0.40, 0.27])
        alcohol = 12.2 + 0.85 * (cultivar == 0) - 0.65 * (cultivar == 1) + 0.55 * (cultivar == 2) + rng.normal(0, 0.45, n)
        flavanoids = 2.0 + 0.95 * (cultivar == 0) + 0.05 * (cultivar == 1) - 1.2 * (cultivar == 2) + rng.normal(0, 0.42, n)
        color_intensity = 4.2 + 1.3 * (cultivar == 0) - 1.1 * (cultivar == 1) + 3.1 * (cultivar == 2) + rng.normal(0, 0.9, n)
        proline = 750 + 360 * (cultivar == 0) - 230 * (cultivar == 1) - 120 * (cultivar == 2) + rng.normal(0, 140, n)
        malic_acid = 2.3 - 0.3 * (cultivar == 0) - 0.4 * (cultivar == 1) + 1.0 * (cultivar == 2) + rng.normal(0, 0.55, n)
        od280 = 2.6 + 0.55 * (cultivar == 0) + 0.2 * (cultivar == 1) - 0.9 * (cultivar == 2) + rng.normal(0, 0.35, n)
        df = pd.DataFrame(
            {
                "alcohol": np.round(alcohol, 2),
                "malic_acid": np.round(malic_acid, 2),
                "flavanoids": np.round(flavanoids, 2),
                "color_intensity": np.round(color_intensity, 2),
                "od280_od315": np.round(od280, 2),
                "proline": np.round(proline, 1),
                "target": cultivar,
            }
        )
        return _inject_nans_and_ensure_size(df, "target", 320, nan_rate=0.09), "target", "classification"

    if dataset_id == "breast_cancer":
        n = 420
        malignant = rng.choice([0, 1], size=n, p=[0.38, 0.62])
        worst_radius = 16.5 - 4.2 * malignant + rng.normal(0, 2.3, n)
        worst_perimeter = 110.0 - 28.0 * malignant + rng.normal(0, 15.0, n)
        worst_concave = 0.145 - 0.075 * malignant + rng.normal(0, 0.035, n).clip(-0.05, 0.2)
        mean_texture = 21.5 - 3.6 * malignant + rng.normal(0, 3.8, n)
        mean_area = 680.0 - 240.0 * malignant + rng.normal(0, 130.0, n)
        df = pd.DataFrame(
            {
                "worst_radius": np.round(worst_radius, 2),
                "worst_perimeter": np.round(worst_perimeter, 2),
                "worst_concave_points": np.round(worst_concave, 4),
                "mean_texture": np.round(mean_texture, 2),
                "mean_area": np.round(mean_area, 1),
                "target": malignant,
            }
        )
        return _inject_nans_and_ensure_size(df, "target", 360, nan_rate=0.08), "target", "classification"

    if dataset_id == "diabetes":
        n = 420
        bmi = rng.normal(26.4, 4.4, n)
        bp = rng.normal(94.6, 13.8, n)
        s5_ltg = rng.normal(4.64, 0.52, n)
        s3_hdl = rng.normal(49.8, 12.9, n)
        age = rng.uniform(22, 75, n)
        target = (
            -180.0
            + 5.8 * bmi
            + 1.15 * bp
            + 42.0 * s5_ltg
            - 0.85 * s3_hdl
            + 0.35 * age
            + 0.12 * ((bmi - 26.0) ** 2)
            + rng.normal(0, 24.0, n)
        )
        df = pd.DataFrame(
            {
                "age": np.round(age, 1),
                "bmi": np.round(bmi, 2),
                "bp": np.round(bp, 2),
                "s3_hdl": np.round(s3_hdl, 2),
                "s5_ltg": np.round(s5_ltg, 3),
                "target": np.round(target, 1),
            }
        )
        return _inject_nans_and_ensure_size(df, "target", 360, nan_rate=0.12), "target", "regression"

    raise ValueError(f"Unknown dataset_id: {dataset_id}")


def load_dataset(dataset_id: str) -> Tuple[pd.DataFrame, str, str]:
    if dataset_id in CUSTOM_DATASETS:
        entry = CUSTOM_DATASETS[dataset_id]
        return entry["df"].copy(), entry["target_column"], entry["task_type"]

    if dataset_id == "housing_synthetic":
        rng = np.random.default_rng(101)
        n = 450
        sqft = rng.normal(1850, 480, size=n).clip(600, 4200)
        bedrooms = rng.integers(1, 6, size=n)
        age_years = rng.uniform(0, 55, size=n)
        transit_km = rng.exponential(3.2, size=n).clip(0.2, 20.0)
        school_score = rng.uniform(4.0, 10.0, size=n)
        neighborhood = rng.choice(["Urban Core", "Suburban", "Waterfront", "Historic"], size=n)
        tier_bonus = (
            pd.Series(neighborhood)
            .map({"Waterfront": 95.0, "Urban Core": 55.0, "Historic": 40.0, "Suburban": 15.0})
            .to_numpy()
        )
        price_k = (
            80.0
            + 0.16 * sqft
            + 14.5 * bedrooms
            - 1.35 * age_years
            + 0.008 * (age_years ** 2)
            - 7.2 * np.log1p(transit_km)
            + 18.0 * school_score
            + tier_bonus
            + rng.normal(0, 18.0, size=n)
        )
        df = pd.DataFrame(
            {
                "sqft": np.round(sqft, 1),
                "bedrooms": bedrooms,
                "age_years": np.round(age_years, 1),
                "transit_km": np.round(transit_km, 2),
                "school_score": np.round(school_score, 2),
                "neighborhood": neighborhood,
                "price_k": np.round(price_k, 2),
            }
        )
        df = _inject_nans_and_ensure_size(df, "price_k", min_clean_rows=400, nan_rate=0.14)
        return df, "price_k", "regression"

    if HAS_SKLEARN and dataset_id in ("iris", "wine", "breast_cancer", "diabetes"):
        loaders = {
            "iris": (sk_datasets.load_iris, "classification", 0.045),
            "wine": (sk_datasets.load_wine, "classification", 0.09),
            "breast_cancer": (sk_datasets.load_breast_cancer, "classification", 0.08),
            "diabetes": (sk_datasets.load_diabetes, "regression", 0.12),
        }
        loader_fn, task_type, nrate = loaders[dataset_id]
        raw = loader_fn(as_frame=True)
        df = _inject_nans_and_ensure_size(raw.frame.copy(), "target", min_clean_rows=320, nan_rate=nrate)
        return df, "target", task_type

    return _generate_fallback_dataset(dataset_id)


def inspect_dataset_columns(
    dataset_id: str,
    target_column: Optional[str] = None,
    task_override: Optional[str] = "auto",
) -> DatasetInspectionResponse:
    df, default_target, _ = load_dataset(dataset_id)
    resolved_target = target_column if (target_column and target_column in df.columns) else default_target

    profiles: List[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        rec_task, rec_reason = intelligent_detect_task_type(s, "auto")
        miss = int(s.isna().sum())
        samples = [
            v.item() if hasattr(v, "item") else v
            for v in s.dropna().head(4).tolist()
        ]
        profiles.append(
            ColumnProfile(
                name=str(col),
                dtype=str(s.dtype),
                missing_count=miss,
                missing_pct=round((miss / max(1, len(df))) * 100.0, 2),
                unique_count=int(s.nunique(dropna=True)),
                sample_values=samples,
                recommended_task_if_target=rec_task,
                task_detection_reason=rec_reason,
            )
        )

    detected_task, reason = intelligent_detect_task_type(df[resolved_target], task_override)
    rec_nan_strategy, nan_rec_reason, nan_row_pct = recommend_nan_strategy(df, resolved_target)

    ds_name = (
        CUSTOM_DATASETS[dataset_id]["name"]
        if dataset_id in CUSTOM_DATASETS
        else BUILTIN_CATALOG.get(dataset_id, {}).get("name", dataset_id)
    )
    clean_rows = int(df.dropna().shape[0])

    return DatasetInspectionResponse(
        dataset_id=dataset_id,
        name=ds_name,
        total_rows=int(len(df)),
        clean_non_nan_rows=clean_rows,
        nan_rows_count=int(len(df) - clean_rows),
        nan_row_pct=nan_row_pct,
        recommended_nan_strategy=rec_nan_strategy,
        nan_recommendation_reason=nan_rec_reason,
        columns=[str(c) for c in df.columns],
        default_target=resolved_target,
        detected_task_type=detected_task,
        task_detection_reason=reason,
        column_profiles=profiles,
    )


def configure_dataset_target(
    dataset_id: str,
    target_column: str,
    task_type: Optional[str] = "auto",
) -> TargetConfigurationResponse:
    df, _, _ = load_dataset(dataset_id)
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset columns: {list(df.columns)}")

    resolved_task, reason = intelligent_detect_task_type(df[target_column], task_type)
    if dataset_id in CUSTOM_DATASETS:
        CUSTOM_DATASETS[dataset_id]["target_column"] = target_column
        CUSTOM_DATASETS[dataset_id]["task_type"] = resolved_task

    s_clean = df[target_column].dropna()
    if resolved_task == "classification":
        vc = s_clean.astype(str).value_counts().head(10).to_dict()
        summary = {"unique_classes": int(s_clean.nunique()), "class_distribution": vc}
    else:
        num_s = pd.to_numeric(s_clean, errors="coerce").dropna()
        summary = {
            "min": round(float(num_s.min()), 4),
            "max": round(float(num_s.max()), 4),
            "mean": round(float(num_s.mean()), 4),
            "std": round(float(num_s.std()), 4),
        }

    return TargetConfigurationResponse(
        dataset_id=dataset_id,
        target_column=target_column,
        user_selected_task=task_type or "auto",
        resolved_task_type=resolved_task,
        intelligent_detection_reason=reason,
        available_feature_columns=[str(c) for c in df.columns if c != target_column],
        target_summary=summary,
    )


def register_uploaded_csv(
    csv_bytes: bytes,
    filename: str,
    target_column: Optional[str] = None,
    task_type: str = "auto",
) -> dict:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if df.empty or df.shape[1] < 2:
        raise ValueError("Uploaded CSV must contain at least 2 columns and non-empty rows.")

    resolved_target = target_column if (target_column and target_column in df.columns) else str(df.columns[-1])
    detected_task, detection_reason = intelligent_detect_task_type(df[resolved_target], task_type)
    custom_id = f"custom_{uuid.uuid4().hex[:8]}"

    CUSTOM_DATASETS[custom_id] = {
        "id": custom_id,
        "name": f"Custom Upload: {filename}",
        "df": df,
        "target_column": resolved_target,
        "task_type": detected_task,
    }
    inspection = inspect_dataset_columns(custom_id, resolved_target, task_type)
    return {
        "id": custom_id,
        "name": f"Custom Upload: {filename}",
        "total_rows": inspection.total_rows,
        "clean_non_nan_rows": inspection.clean_non_nan_rows,
        "nan_rows_count": inspection.nan_rows_count,
        "nan_row_pct": inspection.nan_row_pct,
        "recommended_nan_strategy": inspection.recommended_nan_strategy,
        "nan_recommendation_reason": inspection.nan_recommendation_reason,
        "columns": inspection.columns,
        "default_target": resolved_target,
        "detected_task_type": detected_task,
        "task_detection_reason": detection_reason,
        "column_profiles": [_to_dict(p) for p in inspection.column_profiles],
    }


def delete_custom_dataset(dataset_id: str) -> bool:
    PROCESSED_DATASETS.pop(dataset_id, None)
    if dataset_id in CUSTOM_DATASETS:
        del CUSTOM_DATASETS[dataset_id]
        return True
    return False


def list_available_datasets() -> List[dict]:
    items = list(BUILTIN_CATALOG.values())
    for cid, entry in CUSTOM_DATASETS.items():
        df = entry["df"]
        items.append(
            {
                "id": cid,
                "name": entry["name"],
                "task_type": entry["task_type"],
                "target_column": entry["target_column"],
                "description": f"User-uploaded CSV with {len(df)} total rows ({len(df.dropna())} clean non-NaN rows).",
            }
        )
    return items