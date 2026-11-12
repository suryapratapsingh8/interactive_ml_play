# ML Studio v2.0 — Auto Algorithm Selector & Manual Hyperparameter Playground

A production-grade Machine Learning platform built with **FastAPI**, **Python**, and an interactive 5-view Web Application.

## What's New in v2.0
1. **"What Would You Like To Do?" Workflow Hub**: Starts with an interactive mode selector so you can pick between Dataset & Target Setup, the Auto Algorithm Selector, the Manual Tuning Playground, or the ML Concepts & API Lab.
2. **Dataset Upload with Target Column Dropdown & Intelligent Task Detection**:
   - Upload any `.csv` dataset or choose from 5 built-in datasets.
   - Automatically populates a **Target Column Dropdown** with all columns from the dataset.
   - Optionally choose `Classification` or `Regression`, or leave on `Auto-Detect` to let the intelligent ML heuristic inspect dtype, unique cardinality ratio, and value distribution to determine the task type automatically.
3. **100% Manual Learning & Hyperparameter Tuning Playground**:
   - Pick **one specific algorithm** (`Logistic Regression`, `OLS`, `Ridge`, `Lasso`, `KNN`, `Decision Tree`, `RBF SVM/SVR`, `Random Forest`, `Gradient Boosting`).
   - Tune hyperparameters (`C`/`alpha`, `max_depth`, `min_samples_split`, `n_neighbors`, `knn_weights`, `n_estimators`, `learning_rate`, `kernel_gamma`).
   - Toggle individual feature columns on/off, switch NaN handling (`Drop NaNs` vs `Impute Median/Mode`), inject target noise, and run **1D Hyperparameter Validation Curve Sweeps**.
4. **16 Production FastAPI Endpoints**: Inspectable at `http://127.0.0.1:8000/docs`.

## Run Instructions

```bash
cd ml-playground-v2-production
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://127.0.0.1:8000/** in your browser (note: if you had an older server running on port 8000, stop it first or hard-refresh with `Ctrl+Shift+R` / `Cmd+Shift+R`).
