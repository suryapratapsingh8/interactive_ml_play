import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from concepts import ALGORITHMS_CATALOG
from datasets import (
    intelligent_detect_task_type,
    load_dataset,
    recommend_nan_strategy,
    save_processed_dataset,
)
from schemas import (
    ExperimentConfig,
    ExperimentResponse,
    HyperparameterSweepRequest,
    ManualPlaygroundConfig,
    ManualPlaygroundResponse,
    ModelResult,
    SamplingSummary,
)


def detect_task_type(y: pd.Series, requested: Optional[str] = "auto") -> str:
    task, _ = intelligent_detect_task_type(y, requested)
    return task


def _knn_impute_numeric(df_in: pd.DataFrame, numeric_cols: List[str], k: int = 5) -> pd.DataFrame:
    df = df_in.copy()
    if not numeric_cols:
        return df

    mat = df[numeric_cols].to_numpy(dtype=float)
    col_medians = np.nanmedian(mat, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    col_stds = np.nanstd(mat, axis=0)
    col_stds = np.where((np.isnan(col_stds)) | (col_stds < 1e-9), 1.0, col_stds)

    mat_filled = np.where(np.isnan(mat), col_medians, mat)
    mat_norm = (mat_filled - col_medians) / col_stds

    nan_row_indices = np.where(np.isnan(mat).any(axis=1))[0]
    for r_idx in nan_row_indices:
        missing_cols = np.where(np.isnan(mat[r_idx]))[0]
        obs_cols = np.where(~np.isnan(mat[r_idx]))[0]
        for c_idx in missing_cols:
            valid_donors = np.where(~np.isnan(mat[:, c_idx]))[0]
            if len(valid_donors) == 0 or len(obs_cols) == 0:
                mat[r_idx, c_idx] = col_medians[c_idx]
                continue
            diff = mat_norm[valid_donors][:, obs_cols] - mat_norm[r_idx, obs_cols]
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            k_eff = min(k, len(valid_donors))
            nn_pos = np.argsort(dists)[:k_eff]
            nn_rows = valid_donors[nn_pos]
            weights = 1.0 / (dists[nn_pos] + 1e-5)
            mat[r_idx, c_idx] = float(np.sum(weights * mat[nn_rows, c_idx]) / np.sum(weights))

    df[numeric_cols] = mat
    return df


def sample_percentage_rows(
    df: pd.DataFrame,
    target_col: str,
    sample_percentage: float = 60.0,
    random_state: int = 42,
    task_type: str = "classification",
    nan_strategy: str = "auto",
) -> Tuple[pd.DataFrame, int, int, int, int, str, str]:
    """
    Stage 1: Applies NaN handling first ('auto' -> recommended, 'drop', 'median_mode',
             'mean_mode', 'knn_impute', 'ffill_bfill', or 'constant_zero').
    Stage 2: Randomly samples `sample_percentage` (20% to 80%) of the post-NaN rows.
    """
    original_rows = len(df)
    raw_nan_rows = int(len(df) - len(df.dropna(axis=0, how="any")))
    rec_strategy, rec_reason, _ = recommend_nan_strategy(df, target_col)

    applied_strategy = rec_strategy if nan_strategy == "auto" else nan_strategy

    if applied_strategy == "drop":
        clean_df = df.dropna(axis=0, how="any").reset_index(drop=True)
    else:
        clean_df = df.dropna(subset=[target_col]).copy().reset_index(drop=True)
        feature_cols = [c for c in clean_df.columns if c != target_col]
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(clean_df[c])]
        cat_cols = [c for c in feature_cols if c not in num_cols]

        if applied_strategy == "median_mode":
            for col in num_cols:
                med = clean_df[col].median()
                clean_df[col] = clean_df[col].fillna(0.0 if pd.isna(med) else med)
            for col in cat_cols:
                mode_vals = clean_df[col].mode(dropna=True)
                clean_df[col] = clean_df[col].fillna(mode_vals.iloc[0] if len(mode_vals) > 0 else "Missing")
        elif applied_strategy == "mean_mode":
            for col in num_cols:
                mn = clean_df[col].mean()
                clean_df[col] = clean_df[col].fillna(0.0 if pd.isna(mn) else mn)
            for col in cat_cols:
                mode_vals = clean_df[col].mode(dropna=True)
                clean_df[col] = clean_df[col].fillna(mode_vals.iloc[0] if len(mode_vals) > 0 else "Missing")
        elif applied_strategy == "knn_impute":
            clean_df = _knn_impute_numeric(clean_df, num_cols, k=5)
            for col in cat_cols:
                mode_vals = clean_df[col].mode(dropna=True)
                clean_df[col] = clean_df[col].fillna(mode_vals.iloc[0] if len(mode_vals) > 0 else "Missing")
        elif applied_strategy == "ffill_bfill":
            clean_df[feature_cols] = clean_df[feature_cols].ffill().bfill()
            for col in num_cols:
                clean_df[col] = clean_df[col].fillna(0.0)
            for col in cat_cols:
                clean_df[col] = clean_df[col].fillna("Missing")
        elif applied_strategy == "constant_zero":
            for col in num_cols:
                clean_df[col] = clean_df[col].fillna(0.0)
            for col in cat_cols:
                clean_df[col] = clean_df[col].fillna("Missing")
        else:
            applied_strategy = "drop"
            clean_df = df.dropna(axis=0, how="any").reset_index(drop=True)

        clean_df = clean_df.dropna(axis=0, how="any").reset_index(drop=True)

    rows_after_nan = len(clean_df)
    nan_rows_removed = original_rows - rows_after_nan

    if rows_after_nan == 0:
        raise ValueError("No rows remain after handling NaN values. Please try an Imputer strategy or check your dataset.")

    clamped_pct = max(20.0, min(80.0, float(sample_percentage)))
    target_sample_count = max(20, int(round(rows_after_nan * (clamped_pct / 100.0))))
    actual_sample = min(target_sample_count, rows_after_nan)

    if actual_sample < rows_after_nan:
        if task_type == "classification" and clean_df[target_col].nunique() > 1:
            parts = []
            counts = clean_df[target_col].value_counts(normalize=True)
            for cls_val, prop in counts.items():
                cls_rows = clean_df[clean_df[target_col] == cls_val]
                n_take = max(1, int(round(prop * actual_sample)))
                parts.append(cls_rows.sample(n=min(len(cls_rows), n_take), random_state=random_state))
            sampled_df = pd.concat(parts, ignore_index=True)
            if len(sampled_df) > actual_sample:
                sampled_df = sampled_df.sample(n=actual_sample, random_state=random_state)
            elif len(sampled_df) < actual_sample:
                rem = clean_df.drop(sampled_df.index, errors="ignore")
                extra = rem.sample(n=min(len(rem), actual_sample - len(sampled_df)), random_state=random_state)
                sampled_df = pd.concat([sampled_df, extra], ignore_index=True)
            sampled_df = sampled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        else:
            sampled_df = clean_df.sample(n=actual_sample, random_state=random_state).reset_index(drop=True)
    else:
        sampled_df = clean_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return (
        sampled_df,
        original_rows,
        rows_after_nan,
        raw_nan_rows,
        nan_rows_removed,
        applied_strategy,
        rec_reason,
    )


def prepare_and_store_dataset(
    dataset_id: str,
    target_column: Optional[str] = None,
    task_type: str = "auto",
    nan_strategy: str = "auto",
    sample_percentage: float = 60.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Applies the user-chosen NaN strategy first, then samples 20%-80% of the post-NaN dataset,
    stores the resulting clean dataset so the user can review and download it as a CSV.
    """
    df, default_target, _ = load_dataset(dataset_id)
    t_col = target_column if (target_column and target_column in df.columns) else default_target
    resolved_task, task_reason = intelligent_detect_task_type(df[t_col], task_type)

    (
        sampled_df,
        orig_rows,
        rows_after_nan,
        raw_nan_rows,
        nan_removed,
        applied_strat,
        rec_reason,
    ) = sample_percentage_rows(
        df=df,
        target_col=t_col,
        sample_percentage=sample_percentage,
        random_state=random_state,
        task_type=resolved_task,
        nan_strategy=nan_strategy,
    )

    meta = {
        "dataset_id": dataset_id,
        "target_column": t_col,
        "detected_task_type": resolved_task,
        "task_detection_reason": task_reason,
        "original_rows": orig_rows,
        "raw_nan_rows": raw_nan_rows,
        "nan_strategy_requested": nan_strategy,
        "nan_strategy_applied": applied_strat,
        "nan_recommendation_reason": rec_reason,
        "rows_after_nan_handling": rows_after_nan,
        "nan_rows_removed": nan_removed,
        "remaining_nans_in_processed": int(sampled_df.isna().sum().sum()),
        "sample_percentage_used": sample_percentage,
        "sampled_rows_count": len(sampled_df),
        "columns": [str(c) for c in sampled_df.columns],
        "preview_rows": sampled_df.head(25).to_dict(orient="records"),
    }
    save_processed_dataset(dataset_id, sampled_df, meta)
    return meta


def preprocess_features(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaling: str,
    polynomial_degree: int,
    feature_selection_k: Optional[int],
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_names: List[str] = []
    train_blocks = []
    test_blocks = []

    if numeric_cols:
        X_tr_num = X_train_df[numeric_cols].to_numpy(dtype=float)
        X_te_num = X_test_df[numeric_cols].to_numpy(dtype=float)

        if scaling == "standard":
            mean = X_tr_num.mean(axis=0)
            std = np.where(X_tr_num.std(axis=0) < 1e-9, 1.0, X_tr_num.std(axis=0))
            X_tr_num = (X_tr_num - mean) / std
            X_te_num = (X_te_num - mean) / std
        elif scaling == "minmax":
            mn = X_tr_num.min(axis=0)
            mx = X_tr_num.max(axis=0)
            span = np.where((mx - mn) < 1e-9, 1.0, mx - mn)
            X_tr_num = (X_tr_num - mn) / span
            X_te_num = (X_te_num - mn) / span
        elif scaling == "robust":
            med = np.median(X_tr_num, axis=0)
            q75 = np.percentile(X_tr_num, 75, axis=0)
            q25 = np.percentile(X_tr_num, 25, axis=0)
            iqr = np.where((q75 - q25) < 1e-9, 1.0, q75 - q25)
            X_tr_num = (X_tr_num - med) / iqr
            X_te_num = (X_te_num - med) / iqr

        train_blocks.append(X_tr_num)
        test_blocks.append(X_te_num)
        feature_names.extend(numeric_cols)

        if polynomial_degree == 2:
            n_f = X_tr_num.shape[1]
            poly_tr = []
            poly_te = []
            for i in range(n_f):
                for j in range(i, n_f):
                    poly_tr.append(X_tr_num[:, i] * X_tr_num[:, j])
                    poly_te.append(X_te_num[:, i] * X_te_num[:, j])
                    feature_names.append(f"{numeric_cols[i]}*{numeric_cols[j]}")
            if poly_tr:
                train_blocks.append(np.column_stack(poly_tr))
                test_blocks.append(np.column_stack(poly_te))

    if categorical_cols:
        combined = pd.concat([X_train_df[categorical_cols], X_test_df[categorical_cols]], axis=0)
        dummies = pd.get_dummies(combined, prefix=categorical_cols, dtype=float)
        tr_dum = dummies.iloc[: len(X_train_df)].to_numpy(dtype=float)
        te_dum = dummies.iloc[len(X_train_df) :].to_numpy(dtype=float)
        train_blocks.append(tr_dum)
        test_blocks.append(te_dum)
        feature_names.extend([str(c) for c in dummies.columns])

    X_tr = np.hstack(train_blocks) if train_blocks else np.zeros((len(X_train_df), 1))
    X_te = np.hstack(test_blocks) if test_blocks else np.zeros((len(X_test_df), 1))

    if feature_selection_k and 0 < feature_selection_k < X_tr.shape[1]:
        corrs = np.array(
            [
                abs(np.corrcoef(X_tr[:, idx], y_train)[0, 1]) if np.std(X_tr[:, idx]) > 1e-9 else 0.0
                for idx in range(X_tr.shape[1])
            ]
        )
        corrs = np.nan_to_num(corrs, nan=0.0)
        top_idx = np.argsort(corrs)[::-1][:feature_selection_k]
        X_tr = X_tr[:, top_idx]
        X_te = X_te[:, top_idx]
        feature_names = [feature_names[i] for i in top_idx]

    return X_tr, X_te, feature_names


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(np.mean(y_true == y_pred))
    classes = np.unique(y_true)
    prec_sum, rec_sum, f1_sum = 0.0, 0.0, 0.0
    n_total = len(y_true)
    for c in classes:
        n_c = int(np.sum(y_true == c))
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        weight = n_c / max(1, n_total)
        prec_sum += p * weight
        rec_sum += r * weight
        f1_sum += f * weight

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec_sum, 4),
        "Recall": round(rec_sum, 4),
        "F1 Score": round(f1_sum, 4),
        "MAE (Class Error)": round(mae, 4),
        "MSE (Class Error)": round(mse, 4),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4),
    }


class SimpleCARTNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class CARTTree:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        task: str = "classification",
        max_features: Optional[int] = None,
        rng=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.task = task
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(42)
        self.root = None
        self.feature_importances_ = None

    def _leaf_value(self, y: np.ndarray) -> float:
        if self.task == "classification":
            vals, counts = np.unique(y, return_counts=True)
            return float(vals[np.argmax(counts)])
        return float(np.mean(y))

    def _impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return float(1.0 - np.sum(probs ** 2))
        return float(np.var(y))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features, dtype=float)
        self.root = self._grow(X, y, depth=0)
        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> SimpleCARTNode:
        if depth >= self.max_depth or len(y) < max(4, self.min_samples_split) or np.unique(y).size == 1:
            return SimpleCARTNode(value=self._leaf_value(y))

        n_samples, n_features = X.shape
        feat_indices = np.arange(n_features)
        if self.max_features and self.max_features < n_features:
            feat_indices = self.rng.choice(n_features, size=self.max_features, replace=False)

        parent_imp = self._impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for f in feat_indices:
            col = X[:, f]
            thresholds = np.unique(np.percentile(col, [15, 30, 50, 70, 85]))
            for t in thresholds:
                left_mask = col <= t
                n_l = int(np.sum(left_mask))
                n_r = n_samples - n_l
                if n_l < 2 or n_r < 2:
                    continue
                gain = parent_imp - (
                    (n_l / n_samples) * self._impurity(y[left_mask])
                    + (n_r / n_samples) * self._impurity(y[~left_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(t)

        if best_feat is None or best_gain <= 1e-7:
            return SimpleCARTNode(value=self._leaf_value(y))

        self.feature_importances_[best_feat] += best_gain * len(y)
        mask = X[:, best_feat] <= best_thresh
        left_node = self._grow(X[mask], y[mask], depth + 1)
        right_node = self._grow(X[~mask], y[~mask], depth + 1)
        return SimpleCARTNode(feature=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    def _pred_one(self, row: np.ndarray, node: SimpleCARTNode) -> float:
        if node.value is not None:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._pred_one(row, node.left)
        return self._pred_one(row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([self._pred_one(r, self.root) for r in X])
        return preds.astype(int) if self.task == "classification" else preds


def _fit_predict_classifier(
    alg_key: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    regularization: float = 1.0,
    max_depth: int = 5,
    min_samples_split: int = 2,
    n_neighbors: int = 5,
    knn_weights: str = "uniform",
    n_estimators: int = 25,
    learning_rate: float = 0.2,
    kernel_gamma: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    n_classes = int(np.max(y_tr)) + 1

    if alg_key in ("dummy", "dummy_clf"):
        vals, counts = np.unique(y_tr, return_counts=True)
        maj = vals[np.argmax(counts)]
        return np.full(len(X_tr), maj), np.full(len(X_te), maj)

    if alg_key in ("logistic", "logistic_clf"):
        n_samples, n_features = X_tr.shape
        Y_onehot = np.eye(n_classes)[y_tr.astype(int)]
        reg_lambda = 1.0 / max(0.02, regularization)

        def loss_and_grad(w_flat):
            W = w_flat.reshape(n_features + 1, n_classes)
            logits = X_tr @ W[:-1] + W[-1]
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            loss = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-12), axis=1)) + 0.5 * reg_lambda * np.sum(W[:-1] ** 2) / n_samples
            diff = (probs - Y_onehot) / n_samples
            grad_w = X_tr.T @ diff + (reg_lambda / n_samples) * W[:-1]
            grad_b = np.sum(diff, axis=0, keepdims=True)
            return loss, np.vstack([grad_w, grad_b]).ravel()

        res = minimize(loss_and_grad, np.zeros((n_features + 1) * n_classes), jac=True, method="L-BFGS-B", options={"maxiter": 120})
        W = res.x.reshape(n_features + 1, n_classes)
        return np.argmax(X_tr @ W[:-1] + W[-1], axis=1), np.argmax(X_te @ W[:-1] + W[-1], axis=1)

    if alg_key in ("knn", "knn_clf"):
        k = max(1, min(n_neighbors, len(X_tr) - 1))
        d_tr = cdist(X_tr, X_tr)
        d_te = cdist(X_te, X_tr)

        def vote(dist_mat):
            idx = np.argsort(dist_mat, axis=1)[:, :k]
            out = []
            for row_i, row_idx in enumerate(idx):
                if knn_weights == "distance":
                    dists = dist_mat[row_i, row_idx]
                    w = 1.0 / (dists + 1e-6)
                    scores = [np.sum(w[y_tr[row_idx] == c]) for c in range(n_classes)]
                    out.append(int(np.argmax(scores)))
                else:
                    vals, counts = np.unique(y_tr[row_idx], return_counts=True)
                    out.append(int(vals[np.argmax(counts)]))
            return np.array(out, dtype=int)

        return vote(d_tr), vote(d_te)

    if alg_key in ("tree", "tree_clf"):
        tree = CARTTree(max_depth=max_depth, min_samples_split=min_samples_split, task="classification", rng=np.random.default_rng(random_state))
        tree.fit(X_tr, y_tr)
        return tree.predict(X_tr), tree.predict(X_te)

    if alg_key in ("svm_rbf", "svm_rbf_clf"):
        gamma = kernel_gamma if kernel_gamma is not None else (1.0 / max(1, X_tr.shape[1] * max(float(np.var(X_tr)), 1e-4)))
        K_tr = np.exp(-gamma * cdist(X_tr, X_tr, "sqeuclidean"))
        K_te = np.exp(-gamma * cdist(X_te, X_tr, "sqeuclidean"))
        Y_onehot = np.eye(n_classes)[y_tr.astype(int)]
        reg = 1.0 / max(0.02, regularization)
        alpha = np.linalg.solve(K_tr + reg * np.eye(len(X_tr)), Y_onehot)
        return np.argmax(K_tr @ alpha, axis=1), np.argmax(K_te @ alpha, axis=1)

    if alg_key in ("rf", "rf_clf"):
        rng = np.random.default_rng(random_state)
        max_f = max(1, int(np.sqrt(X_tr.shape[1])))
        tr_votes = np.zeros((len(X_tr), n_classes))
        te_votes = np.zeros((len(X_te), n_classes))
        for _ in range(n_estimators):
            boot = rng.choice(len(X_tr), size=len(X_tr), replace=True)
            t = CARTTree(max_depth=max_depth, min_samples_split=min_samples_split, task="classification", max_features=max_f, rng=rng)
            t.fit(X_tr[boot], y_tr[boot])
            p_tr = t.predict(X_tr)
            p_te = t.predict(X_te)
            for c in range(n_classes):
                tr_votes[:, c] += (p_tr == c)
                te_votes[:, c] += (p_te == c)
        return np.argmax(tr_votes, axis=1), np.argmax(te_votes, axis=1)

    if alg_key in ("gb", "gb_clf"):
        rng = np.random.default_rng(random_state)
        Y_onehot = np.eye(n_classes)[y_tr.astype(int)]
        F_tr = np.zeros((len(X_tr), n_classes))
        F_te = np.zeros((len(X_te), n_classes))
        depth = min(max_depth or 3, 6)
        for _ in range(n_estimators):
            logits = F_tr - np.max(F_tr, axis=1, keepdims=True)
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            residuals = Y_onehot - probs
            for c in range(n_classes):
                tree = CARTTree(max_depth=depth, min_samples_split=min_samples_split, task="regression", rng=rng)
                tree.fit(X_tr, residuals[:, c])
                F_tr[:, c] += learning_rate * tree.predict(X_tr)
                F_te[:, c] += learning_rate * tree.predict(X_te)
        return np.argmax(F_tr, axis=1), np.argmax(F_te, axis=1)

    raise ValueError(f"Unknown classification algorithm: {alg_key}")


def _fit_predict_regressor(
    alg_key: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    regularization: float = 1.0,
    max_depth: int = 5,
    min_samples_split: int = 2,
    n_neighbors: int = 5,
    knn_weights: str = "uniform",
    n_estimators: int = 25,
    learning_rate: float = 0.2,
    kernel_gamma: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if alg_key in ("dummy", "dummy_reg"):
        m = float(np.mean(y_tr))
        return np.full(len(X_tr), m), np.full(len(X_te), m)

    X_tr_b = np.column_stack([X_tr, np.ones(len(X_tr))])
    X_te_b = np.column_stack([X_te, np.ones(len(X_te))])

    if alg_key in ("ols", "ols_reg"):
        w, _, _, _ = np.linalg.lstsq(X_tr_b, y_tr, rcond=None)
        return X_tr_b @ w, X_te_b @ w

    if alg_key in ("ridge", "ridge_reg"):
        reg = regularization * np.eye(X_tr_b.shape[1])
        reg[-1, -1] = 0.0
        w = np.linalg.solve(X_tr_b.T @ X_tr_b + reg, X_tr_b.T @ y_tr)
        return X_tr_b @ w, X_te_b @ w

    if alg_key in ("lasso", "lasso_reg"):
        alpha = regularization
        w = np.linalg.lstsq(X_tr_b, y_tr, rcond=None)[0]
        for _ in range(60):
            for j in range(X_tr.shape[1]):
                col = X_tr_b[:, j]
                denom = float(np.dot(col, col))
                if denom < 1e-9:
                    continue
                res_j = y_tr - (X_tr_b @ w - col * w[j])
                rho = float(np.dot(col, res_j))
                w[j] = np.sign(rho) * max(0.0, abs(rho) - alpha * len(X_tr) * 0.5) / denom
            w[-1] = float(np.mean(y_tr - X_tr @ w[:-1]))
        return X_tr_b @ w, X_te_b @ w

    if alg_key in ("knn", "knn_reg"):
        k = max(1, min(n_neighbors, len(X_tr) - 1))
        d_tr = cdist(X_tr, X_tr)
        d_te = cdist(X_te, X_tr)

        def avg_k(dist_mat):
            idx = np.argsort(dist_mat, axis=1)[:, :k]
            if knn_weights == "distance":
                out = []
                for row_i, row_idx in enumerate(idx):
                    w = 1.0 / (dist_mat[row_i, row_idx] + 1e-6)
                    out.append(float(np.sum(w * y_tr[row_idx]) / np.sum(w)))
                return np.array(out, dtype=float)
            return np.mean(y_tr[idx], axis=1)

        return avg_k(d_tr), avg_k(d_te)

    if alg_key in ("tree", "tree_reg"):
        tree = CARTTree(max_depth=max_depth, min_samples_split=min_samples_split, task="regression", rng=np.random.default_rng(random_state))
        tree.fit(X_tr, y_tr)
        return tree.predict(X_tr), tree.predict(X_te)

    if alg_key in ("svr_rbf", "svr_rbf_reg"):
        gamma = kernel_gamma if kernel_gamma is not None else (1.0 / max(1, X_tr.shape[1] * max(float(np.var(X_tr)), 1e-4)))
        K_tr = np.exp(-gamma * cdist(X_tr, X_tr, "sqeuclidean"))
        K_te = np.exp(-gamma * cdist(X_te, X_tr, "sqeuclidean"))
        reg = 1.0 / max(0.02, regularization)
        y_mean = float(np.mean(y_tr))
        alpha = np.linalg.solve(K_tr + reg * np.eye(len(X_tr)), y_tr - y_mean)
        return K_tr @ alpha + y_mean, K_te @ alpha + y_mean

    if alg_key in ("rf", "rf_reg"):
        rng = np.random.default_rng(random_state)
        max_f = max(1, int(np.ceil(X_tr.shape[1] * 0.7)))
        tr_preds = np.zeros(len(X_tr))
        te_preds = np.zeros(len(X_te))
        for _ in range(n_estimators):
            boot = rng.choice(len(X_tr), size=len(X_tr), replace=True)
            t = CARTTree(max_depth=max_depth, min_samples_split=min_samples_split, task="regression", max_features=max_f, rng=rng)
            t.fit(X_tr[boot], y_tr[boot])
            tr_preds += t.predict(X_tr) / n_estimators
            te_preds += t.predict(X_te) / n_estimators
        return tr_preds, te_preds

    if alg_key in ("gb", "gb_reg"):
        rng = np.random.default_rng(random_state)
        base_val = float(np.mean(y_tr))
        f_tr = np.full(len(X_tr), base_val)
        f_te = np.full(len(X_te), base_val)
        depth = min(max_depth or 3, 6)
        for _ in range(n_estimators):
            residuals = y_tr - f_tr
            t = CARTTree(max_depth=depth, min_samples_split=min_samples_split, task="regression", rng=rng)
            t.fit(X_tr, residuals)
            f_tr += learning_rate * t.predict(X_tr)
            f_te += learning_rate * t.predict(X_te)
        return f_tr, f_te

    raise ValueError(f"Unknown regression algorithm: {alg_key}")


def _diagnose_bias_variance(train_score: float, test_score: float, baseline_score: float) -> str:
    gap = train_score - test_score
    if test_score <= baseline_score + 0.03 and train_score < 0.65:
        return "High Bias (Underfitting): Model is too simple to capture signal."
    if gap > 0.12:
        return f"High Variance (Overfitting): Train score exceeds test score by {gap:.3f}."
    if gap > 0.06:
        return f"Mild Variance: Slight gap ({gap:.3f}) between train and test performance."
    return "Well-Balanced Generalization: Strong test performance with low train-test gap."


def run_experiment(
    config: ExperimentConfig,
    progress_callback: Optional[Callable[[int, int, str, float, float], None]] = None,
) -> ExperimentResponse:
    """
    Runs progressive Tier 0 -> Tier 6 benchmark with optional real-time progress & ETA callbacks.
    """
    start_ts = time.perf_counter()
    df, default_target, _ = load_dataset(config.dataset_id)
    target_col = config.target_column if (config.target_column and config.target_column in df.columns) else default_target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    inferred_task, detection_reason = intelligent_detect_task_type(df[target_col], config.task_type)

    (
        sampled_df,
        original_rows,
        rows_after_nan,
        raw_nan_rows,
        nan_rows_removed,
        applied_nan_strategy,
        nan_rec_reason,
    ) = sample_percentage_rows(
        df=df,
        target_col=target_col,
        sample_percentage=config.sample_percentage,
        random_state=config.random_state,
        task_type=inferred_task,
        nan_strategy=config.nan_strategy,
    )

    # Also store processed dataset so user can download it after running benchmark
    save_processed_dataset(
        config.dataset_id,
        sampled_df,
        {
            "nan_strategy_applied": applied_nan_strategy,
            "sample_percentage_used": config.sample_percentage,
            "target_column": target_col,
        },
    )

    X_df = sampled_df.drop(columns=[target_col])
    y_raw = sampled_df[target_col]

    numeric_cols = [str(c) for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical_cols = [str(c) for c in X_df.columns if c not in numeric_cols]

    if inferred_task == "classification":
        _, y_encoded = np.unique(y_raw.astype(str), return_inverse=True)
        y_all = y_encoded.astype(int)
    else:
        y_all = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)

    n_total = len(sampled_df)
    n_test = max(6, int(round(n_total * config.test_size)))
    n_train = n_total - n_test

    X_train_df = X_df.iloc[:n_train].reset_index(drop=True)
    X_test_df = X_df.iloc[n_train:].reset_index(drop=True)
    y_train = y_all[:n_train]
    y_test = y_all[n_train:]

    X_tr, X_te, feature_names = preprocess_features(
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scaling=config.scaling,
        polynomial_degree=config.polynomial_degree,
        feature_selection_k=config.feature_selection_k,
        y_train=y_train,
    )

    if inferred_task == "classification":
        primary_metric = "F1 Score (Weighted)"
        candidate_specs = [
            ("Zero-Rule Baseline (Dummy Classifier)", "Tier 0: Naive Baseline", 0, "Predicts the most frequent class to establish the minimum floor.", "dummy"),
            ("Logistic Regression", "Tier 1: Linear Model", 1, "Fits linear log-odds decision boundaries with L2 regularization.", "logistic"),
            ("K-Nearest Neighbors (KNN)", "Tier 2: Distance / Instance-Based", 2, "Classifies points by local k-neighborhood vote; sensitive to scaling.", "knn"),
            ("Decision Tree Classifier", "Tier 3: Non-Linear Single Tree", 3, "Recursively splits features by Gini impurity; can overfit when deep.", "tree"),
            ("Support Vector / Kernel Classifier (RBF)", "Tier 4: Kernel Classifier", 4, "Projects features into smooth non-linear RBF kernel space.", "svm_rbf"),
            ("Random Forest Classifier", "Tier 5: Bagging Ensemble", 5, "Bagged ensemble of bootstrapped CART trees that reduces variance.", "rf"),
            ("Gradient Boosting Classifier", "Tier 6: Boosting Ensemble", 6, "Sequentially fits shallow trees to softmax residuals.", "gb"),
        ]
    else:
        primary_metric = "R2 Score (with MAE & RMSE)"
        candidate_specs = [
            ("Mean Baseline (Dummy Regressor)", "Tier 0: Naive Baseline", 0, "Always predicts the training target mean (R2 = 0.0).", "dummy"),
            ("Ordinary Least Squares (Linear Regression)", "Tier 1: Unregularized Linear", 1, "Fits a linear hyperplane minimizing squared error (MSE).", "ols"),
            ("Ridge Regression (L2 Regularized)", "Tier 1B: L2 Linear Model", 2, "Shrinks correlated feature weights smoothly via L2 penalty.", "ridge"),
            ("Lasso Regression (L1 Sparse Linear)", "Tier 1C: L1 Linear Model", 3, "Drives redundant feature weights to zero via L1 soft-thresholding.", "lasso"),
            ("K-Nearest Neighbors Regressor", "Tier 2: Distance / Local Averaging", 4, "Predicts the local target average of the k nearest neighbors.", "knn"),
            ("Decision Tree Regressor", "Tier 3: Non-Linear Piecewise Tree", 5, "Partitions feature space into rectangular constant regions.", "tree"),
            ("Kernel Support Vector Regressor (RBF)", "Tier 4: Kernel Regression", 6, "Fits a smooth non-linear function in RBF kernel space.", "svr_rbf"),
            ("Random Forest Regressor", "Tier 5: Bagging Ensemble", 7, "Averages predictions across bootstrapped regression trees.", "rf"),
            ("Gradient Boosting Regressor", "Tier 6: Boosting Ensemble", 8, "Sequentially fits shallow trees to negative loss gradients.", "gb"),
        ]

    total_steps = len(candidate_specs)
    results: List[ModelResult] = []
    baseline_test_score = 0.0
    k_folds = min(config.cv_folds, max(2, len(X_tr) // 4))
    fold_indices = np.array_split(np.arange(len(X_tr)), k_folds)

    if progress_callback:
        progress_callback(0, total_steps, f"Preparing {len(sampled_df)} clean rows & starting Tier 0...", 0.0, 2.5)

    for step_idx, (name, tier_label, tier_order, why_tier, alg_key) in enumerate(candidate_specs, start=1):
        if progress_callback:
            elapsed_now = time.perf_counter() - start_ts
            avg_per_step = (elapsed_now / max(1, step_idx - 1)) if step_idx > 1 else 0.35
            eta_now = avg_per_step * (total_steps - step_idx + 1)
            progress_callback(step_idx - 1, total_steps, f"Evaluating {name} ({tier_label})...", elapsed_now, eta_now)

        t0 = time.perf_counter()
        if inferred_task == "classification":
            y_pred_tr, y_pred_te = _fit_predict_classifier(
                alg_key, X_tr, y_train, X_te,
                regularization=config.regularization_c_or_alpha,
                max_depth=config.max_depth or 5,
                n_neighbors=config.n_neighbors,
                random_state=config.random_state,
            )
            tr_m = classification_metrics(y_train, y_pred_tr)
            te_m = classification_metrics(y_test, y_pred_te)
            train_primary = tr_m["F1 Score"]
            test_primary = te_m["F1 Score"]
        else:
            y_pred_tr, y_pred_te = _fit_predict_regressor(
                alg_key, X_tr, y_train, X_te,
                regularization=config.regularization_c_or_alpha,
                max_depth=config.max_depth or 5,
                n_neighbors=config.n_neighbors,
                random_state=config.random_state,
            )
            tr_m = regression_metrics(y_train, y_pred_tr)
            te_m = regression_metrics(y_test, y_pred_te)
            train_primary = tr_m["R2 Score"]
            test_primary = te_m["R2 Score"]
        fit_ms = (time.perf_counter() - t0) * 1000.0

        cv_scores = []
        for f_idx in range(k_folds):
            val_idx = fold_indices[f_idx]
            tr_idx = np.setdiff1d(np.arange(len(X_tr)), val_idx)
            if inferred_task == "classification":
                _, val_pred = _fit_predict_classifier(
                    alg_key, X_tr[tr_idx], y_train[tr_idx], X_tr[val_idx],
                    regularization=config.regularization_c_or_alpha,
                    max_depth=config.max_depth or 5,
                    n_neighbors=config.n_neighbors,
                    random_state=config.random_state,
                )
                cv_scores.append(classification_metrics(y_train[val_idx], val_pred)["F1 Score"])
            else:
                _, val_pred = _fit_predict_regressor(
                    alg_key, X_tr[tr_idx], y_train[tr_idx], X_tr[val_idx],
                    regularization=config.regularization_c_or_alpha,
                    max_depth=config.max_depth or 5,
                    n_neighbors=config.n_neighbors,
                    random_state=config.random_state,
                )
                cv_scores.append(regression_metrics(y_train[val_idx], val_pred)["R2 Score"])

        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        if tier_order == 0:
            baseline_test_score = test_primary

        gap = float(train_primary - test_primary)
        diagnosis = _diagnose_bias_variance(train_primary, test_primary, baseline_test_score)

        results.append(
            ModelResult(
                algorithm=name,
                complexity_tier=tier_label,
                tier_order=tier_order,
                why_this_tier=why_tier,
                train_score=round(train_primary, 4),
                test_score=round(test_primary, 4),
                cv_mean=round(cv_mean, 4),
                cv_std=round(cv_std, 4),
                overfitting_gap=round(gap, 4),
                bias_variance_diagnosis=diagnosis,
                metrics=te_m,
                fit_time_ms=round(fit_ms, 2),
            )
        )

        if progress_callback:
            elapsed_after = time.perf_counter() - start_ts
            avg_step = elapsed_after / step_idx
            eta_after = max(0.0, avg_step * (total_steps - step_idx))
            progress_callback(step_idx, total_steps, f"Completed {name} ({step_idx}/{total_steps})", elapsed_after, eta_after)

    ordered_by_tier = sorted(results, key=lambda r: r.tier_order)
    best_so_far = ordered_by_tier[0]
    progression_narrative = [
        f"Step 1 ({best_so_far.complexity_tier}): Started with {best_so_far.algorithm}, achieving a baseline test score of {best_so_far.test_score:.4f} (CV: {best_so_far.cv_mean:.4f})."
    ]
    for idx, res in enumerate(ordered_by_tier[1:], start=2):
        delta = res.test_score - best_so_far.test_score
        if delta > 0.005:
            progression_narrative.append(
                f"Step {idx} ({res.complexity_tier}): {res.algorithm} improved test score by +{delta:.4f} to {res.test_score:.4f} (CV: {res.cv_mean:.4f} ± {res.cv_std:.4f})."
            )
            best_so_far = res
        else:
            progression_narrative.append(
                f"Step {idx} ({res.complexity_tier}): {res.algorithm} scored {res.test_score:.4f} (did not surpass {best_so_far.algorithm}'s {best_so_far.test_score:.4f})."
            )

    leaderboard = sorted(
        results,
        key=lambda r: (0.6 * r.cv_mean + 0.4 * r.test_score, -abs(r.overfitting_gap)),
        reverse=True,
    )
    winner = leaderboard[0]
    runner_up = leaderboard[1] if len(leaderboard) > 1 else winner

    probe_tree = CARTTree(max_depth=5, task=inferred_task, rng=np.random.default_rng(config.random_state))
    probe_tree.fit(X_tr, y_train)
    imp_vec = probe_tree.feature_importances_
    if np.sum(imp_vec) <= 1e-9:
        imp_vec = np.array([abs(np.corrcoef(X_tr[:, i], y_train)[0, 1]) for i in range(X_tr.shape[1])])
        imp_vec = np.nan_to_num(imp_vec, nan=0.01)
        imp_vec = imp_vec / max(np.sum(imp_vec), 1e-9)

    pairs = sorted(zip(feature_names, imp_vec), key=lambda x: x[1], reverse=True)[:10]
    feature_importances = {str(k): round(float(v), 4) for k, v in pairs}

    selection_rationale = (
        f"{winner.algorithm} ({winner.complexity_tier}) ranked #1 out of {len(leaderboard)} algorithms on the "
        f"{config.sample_percentage:.0f}% sample ({len(sampled_df)} rows after '{applied_nan_strategy}' NaN handling). "
        f"It achieved a test score of {winner.test_score:.4f} and {k_folds}-fold CV mean of {winner.cv_mean:.4f} (±{winner.cv_std:.4f}), "
        f"outperforming the naive baseline ({baseline_test_score:.4f}) and runner-up {runner_up.algorithm} ({runner_up.test_score:.4f})."
    )

    concept_takeaways = {
        "Percentage Sampling (20%–80%)": (
            f"Sampled {config.sample_percentage:.0f}% ({len(sampled_df)} of {rows_after_nan} post-NaN rows) using seed={config.random_state}."
        ),
        "NaN Handling & Suggestion": (
            f"Applied '{applied_nan_strategy}' (requested: '{config.nan_strategy}'). {nan_rec_reason}"
        ),
        "Intelligent Target & Task Detection": detection_reason,
        "Feature Scaling Impact": (
            f"Current scaling is '{config.scaling}'. Distance-based (KNN) and kernel/regularized models (SVM, Logistic, Ridge/Lasso) respond strongly to scaling, while tree models remain scale-invariant."
        ),
    }

    return ExperimentResponse(
        sampling_summary=SamplingSummary(
            original_rows=original_rows,
            rows_after_nan_handling=rows_after_nan,
            nan_rows_in_raw=raw_nan_rows,
            nan_rows_removed=nan_rows_removed,
            nan_strategy_requested=config.nan_strategy,
            nan_strategy_applied=applied_nan_strategy,
            nan_recommendation_reason=nan_rec_reason,
            sampled_rows=len(sampled_df),
            sample_percentage_used=config.sample_percentage,
            train_rows=len(X_train_df),
            test_rows=len(X_test_df),
            numeric_features=numeric_cols,
            categorical_features=categorical_cols,
            target_column=target_col,
            detected_task_type=inferred_task,
            task_detection_reason=detection_reason,
        ),
        primary_metric=primary_metric,
        best_algorithm=winner.algorithm,
        best_score=winner.test_score,
        selection_rationale=selection_rationale,
        progression_narrative=progression_narrative,
        leaderboard=leaderboard,
        feature_importances=feature_importances,
        concept_takeaways=concept_takeaways,
        sample_preview=sampled_df.head(8).to_dict(orient="records"),
    )


def run_manual_playground(config: ManualPlaygroundConfig) -> ManualPlaygroundResponse:
    df, default_target, _ = load_dataset(config.dataset_id)
    target_col = config.target_column if (config.target_column and config.target_column in df.columns) else default_target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    inferred_task, detection_reason = intelligent_detect_task_type(df[target_col], config.task_type)

    alg_map = {a["id"]: a for a in ALGORITHMS_CATALOG}
    alg_id = config.algorithm_id
    if alg_id not in alg_map or alg_map[alg_id]["task_type"] != inferred_task:
        prefix = alg_id.replace("_clf", "").replace("_reg", "")
        candidate_id = f"{prefix}_{'clf' if inferred_task == 'classification' else 'reg'}"
        alg_id = candidate_id if candidate_id in alg_map else ("rf_clf" if inferred_task == "classification" else "rf_reg")

    alg_meta = alg_map[alg_id]

    (
        sampled_df,
        original_rows,
        rows_after_nan,
        raw_nan_rows,
        nan_rows_removed,
        applied_nan_strategy,
        nan_rec_reason,
    ) = sample_percentage_rows(
        df=df,
        target_col=target_col,
        sample_percentage=config.sample_percentage,
        random_state=config.random_state,
        task_type=inferred_task,
        nan_strategy=config.nan_strategy,
    )

    drop_cols = [target_col] + [c for c in config.excluded_features if c in sampled_df.columns and c != target_col]
    X_df = sampled_df.drop(columns=drop_cols)
    if X_df.shape[1] == 0:
        raise ValueError("At least 1 feature column must remain active (cannot exclude all features).")

    y_raw = sampled_df[target_col]
    numeric_cols = [str(c) for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical_cols = [str(c) for c in X_df.columns if c not in numeric_cols]

    rng = np.random.default_rng(config.random_state)
    if inferred_task == "classification":
        class_labels, y_encoded = np.unique(y_raw.astype(str), return_inverse=True)
        y_all = y_encoded.astype(int)
        if config.add_target_noise > 0:
            n_flip = int(round(len(y_all) * config.add_target_noise))
            if n_flip > 0 and len(class_labels) > 1:
                flip_idx = rng.choice(len(y_all), size=n_flip, replace=False)
                y_all[flip_idx] = rng.integers(0, len(class_labels), size=n_flip)
    else:
        class_labels = np.array([])
        y_all = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        if config.add_target_noise > 0:
            std_y = float(np.std(y_all) or 1.0)
            y_all = y_all + rng.normal(0, config.add_target_noise * std_y, size=len(y_all))

    n_total = len(sampled_df)
    n_test = max(6, int(round(n_total * config.test_size)))
    n_train = n_total - n_test

    X_train_df = X_df.iloc[:n_train].reset_index(drop=True)
    X_test_df = X_df.iloc[n_train:].reset_index(drop=True)
    y_train = y_all[:n_train]
    y_test = y_all[n_train:]

    X_tr, X_te, feature_names = preprocess_features(
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scaling=config.scaling,
        polynomial_degree=config.polynomial_degree,
        feature_selection_k=config.feature_selection_k,
        y_train=y_train,
    )

    t0 = time.perf_counter()
    if inferred_task == "classification":
        primary_metric = "F1 Score (Weighted)"
        y_pred_tr, y_pred_te = _fit_predict_classifier(
            alg_id, X_tr, y_train, X_te,
            regularization=config.regularization, max_depth=config.max_depth,
            min_samples_split=config.min_samples_split, n_neighbors=config.n_neighbors,
            knn_weights=config.knn_weights, n_estimators=config.n_estimators,
            learning_rate=config.learning_rate, kernel_gamma=config.kernel_gamma,
            random_state=config.random_state,
        )
        tr_m = classification_metrics(y_train, y_pred_tr)
        te_m = classification_metrics(y_test, y_pred_te)
        train_primary = tr_m["F1 Score"]
        test_primary = te_m["F1 Score"]
        _, base_te = _fit_predict_classifier("dummy_clf", X_tr, y_train, X_te)
        baseline_score = classification_metrics(y_test, base_te)["F1 Score"]
    else:
        primary_metric = "R2 Score (with MAE & RMSE)"
        y_pred_tr, y_pred_te = _fit_predict_regressor(
            alg_id, X_tr, y_train, X_te,
            regularization=config.regularization, max_depth=config.max_depth,
            min_samples_split=config.min_samples_split, n_neighbors=config.n_neighbors,
            knn_weights=config.knn_weights, n_estimators=config.n_estimators,
            learning_rate=config.learning_rate, kernel_gamma=config.kernel_gamma,
            random_state=config.random_state,
        )
        tr_m = regression_metrics(y_train, y_pred_tr)
        te_m = regression_metrics(y_test, y_pred_te)
        train_primary = tr_m["R2 Score"]
        test_primary = te_m["R2 Score"]
        baseline_score = 0.0
    fit_ms = (time.perf_counter() - t0) * 1000.0

    k_folds = min(config.cv_folds, max(2, len(X_tr) // 4))
    fold_indices = np.array_split(np.arange(len(X_tr)), k_folds)
    cv_scores = []
    for f_idx in range(k_folds):
        val_idx = fold_indices[f_idx]
        tr_idx = np.setdiff1d(np.arange(len(X_tr)), val_idx)
        if inferred_task == "classification":
            _, val_pred = _fit_predict_classifier(
                alg_id, X_tr[tr_idx], y_train[tr_idx], X_tr[val_idx],
                regularization=config.regularization, max_depth=config.max_depth,
                min_samples_split=config.min_samples_split, n_neighbors=config.n_neighbors,
                knn_weights=config.knn_weights, n_estimators=config.n_estimators,
                learning_rate=config.learning_rate, kernel_gamma=config.kernel_gamma,
                random_state=config.random_state,
            )
            cv_scores.append(classification_metrics(y_train[val_idx], val_pred)["F1 Score"])
        else:
            _, val_pred = _fit_predict_regressor(
                alg_id, X_tr[tr_idx], y_train[tr_idx], X_tr[val_idx],
                regularization=config.regularization, max_depth=config.max_depth,
                min_samples_split=config.min_samples_split, n_neighbors=config.n_neighbors,
                knn_weights=config.knn_weights, n_estimators=config.n_estimators,
                learning_rate=config.learning_rate, kernel_gamma=config.kernel_gamma,
                random_state=config.random_state,
            )
            cv_scores.append(regression_metrics(y_train[val_idx], val_pred)["R2 Score"])

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    gap = float(train_primary - test_primary)
    diagnosis = _diagnose_bias_variance(train_primary, test_primary, baseline_score)

    confusion_data = None
    pred_diagnostics = []
    if inferred_task == "classification":
        n_cls = len(class_labels)
        mat = np.zeros((n_cls, n_cls), dtype=int)
        for yt, yp in zip(y_test, y_pred_te):
            if 0 <= yt < n_cls and 0 <= yp < n_cls:
                mat[int(yt), int(yp)] += 1
        confusion_data = {"labels": [str(lbl) for lbl in class_labels], "matrix": mat.tolist()}
        for i in range(min(10, len(y_test))):
            pred_diagnostics.append(
                {
                    "row": i + 1,
                    "actual": str(class_labels[int(y_test[i])]),
                    "predicted": str(class_labels[int(y_pred_te[i])]),
                    "correct": bool(y_test[i] == y_pred_te[i]),
                }
            )
    else:
        for i in range(min(10, len(y_test))):
            err = float(y_pred_te[i] - y_test[i])
            pred_diagnostics.append(
                {
                    "row": i + 1,
                    "actual": round(float(y_test[i]), 3),
                    "predicted": round(float(y_pred_te[i]), 3),
                    "residual": round(err, 3),
                }
            )

    probe_tree = CARTTree(max_depth=5, task=inferred_task, rng=np.random.default_rng(config.random_state))
    probe_tree.fit(X_tr, y_train)
    imp_vec = probe_tree.feature_importances_
    if np.sum(imp_vec) <= 1e-9:
        imp_vec = np.array([abs(np.corrcoef(X_tr[:, i], y_train)[0, 1]) for i in range(X_tr.shape[1])])
        imp_vec = np.nan_to_num(imp_vec, nan=0.01)
        imp_vec = imp_vec / max(np.sum(imp_vec), 1e-9)
    pairs = sorted(zip(feature_names, imp_vec), key=lambda x: x[1], reverse=True)[:10]
    feature_importances = {str(k): round(float(v), 4) for k, v in pairs}

    coaching_tips = [
        f"NaN Strategy Applied: '{applied_nan_strategy}' (Requested: '{config.nan_strategy}'). {nan_rec_reason}",
        f"Sample Size: Used {config.sample_percentage:.0f}% of post-NaN dataset ({len(sampled_df)} of {rows_after_nan} rows).",
        f"Generalization Check: Train = {train_primary:.4f} vs. Test = {test_primary:.4f} (Gap = {gap:.4f}, {k_folds}-Fold CV = {cv_mean:.4f} ± {cv_std:.4f}).",
    ]
    if gap > 0.10:
        coaching_tips.append(
            "Overfitting Detected: Try increasing Sample Percentage toward 80%, lowering Max Depth, or increasing Regularization."
        )
    if config.scaling == "none" and any(k in alg_id for k in ("knn", "svm", "svr", "logistic", "ridge", "lasso")):
        coaching_tips.append(
            "Scaling Warning: Switch Scaling to 'standard' for distance/kernel/regularized models."
        )

    applied_hparams = {
        "sample_percentage": config.sample_percentage,
        "nan_strategy_applied": applied_nan_strategy,
        "scaling": config.scaling,
        "polynomial_degree": config.polynomial_degree,
        "regularization": config.regularization,
        "max_depth": config.max_depth,
        "min_samples_split": config.min_samples_split,
        "n_neighbors": config.n_neighbors,
        "knn_weights": config.knn_weights,
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "kernel_gamma": config.kernel_gamma,
        "add_target_noise": config.add_target_noise,
    }

    return ManualPlaygroundResponse(
        run_id=f"run_{uuid.uuid4().hex[:6]}",
        algorithm_id=alg_id,
        algorithm_name=alg_meta["name"],
        complexity_tier=alg_meta["complexity_tier"],
        sampling_summary=SamplingSummary(
            original_rows=original_rows,
            rows_after_nan_handling=rows_after_nan,
            nan_rows_in_raw=raw_nan_rows,
            nan_rows_removed=nan_rows_removed,
            nan_strategy_requested=config.nan_strategy,
            nan_strategy_applied=applied_nan_strategy,
            nan_recommendation_reason=nan_rec_reason,
            sampled_rows=len(sampled_df),
            sample_percentage_used=config.sample_percentage,
            train_rows=len(X_train_df),
            test_rows=len(X_test_df),
            numeric_features=numeric_cols,
            categorical_features=categorical_cols,
            target_column=target_col,
            detected_task_type=inferred_task,
            task_detection_reason=detection_reason,
        ),
        applied_hyperparameters=applied_hparams,
        primary_metric=primary_metric,
        train_score=round(train_primary, 4),
        test_score=round(test_primary, 4),
        cv_mean=round(cv_mean, 4),
        cv_std=round(cv_std, 4),
        overfitting_gap=round(gap, 4),
        bias_variance_diagnosis=diagnosis,
        metrics=te_m,
        feature_importances=feature_importances,
        confusion_matrix=confusion_data,
        prediction_diagnostics=pred_diagnostics,
        coaching_tips=coaching_tips,
        fit_time_ms=round(fit_ms, 2),
    )


def run_hyperparameter_sweep(req: HyperparameterSweepRequest) -> Dict[str, Any]:
    default_grids = {
        "max_depth": [1, 2, 3, 5, 8, 12, 16],
        "n_neighbors": [1, 3, 5, 9, 15, 21, 29],
        "regularization": [0.05, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0],
        "n_estimators": [5, 10, 20, 35, 50, 75],
        "learning_rate": [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7],
    }
    grid = req.param_values if req.param_values else default_grids.get(req.param_name, [1, 3, 5, 10])
    curve = []
    res = None
    for val in grid:
        cfg_copy = (
            req.playground_config.model_copy(deep=True)
            if hasattr(req.playground_config, "model_copy")
            else req.playground_config.copy(deep=True)
        )
        if hasattr(cfg_copy, req.param_name):
            if req.param_name in ("max_depth", "n_neighbors", "n_estimators", "min_samples_split"):
                setattr(cfg_copy, req.param_name, int(round(val)))
            else:
                setattr(cfg_copy, req.param_name, float(val))
        res = run_manual_playground(cfg_copy)
        curve.append(
            {
                "param_value": val,
                "train_score": res.train_score,
                "test_score": res.test_score,
                "cv_mean": res.cv_mean,
                "overfitting_gap": res.overfitting_gap,
            }
        )
    best_pt = max(curve, key=lambda x: (0.6 * x["cv_mean"] + 0.4 * x["test_score"]))
    return {
        "param_name": req.param_name,
        "algorithm_name": res.algorithm_name if res else "",
        "primary_metric": res.primary_metric if res else "",
        "best_param_value": best_pt["param_value"],
        "best_cv_mean": best_pt["cv_mean"],
        "best_test_score": best_pt["test_score"],
        "curve": curve,
    }