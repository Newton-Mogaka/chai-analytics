# ChaiAnalytics — Model-Native Regression Benchmark

**Created:** 2026-11-11  
**Branch:** main  


---

## Overview

Implemented **Option 2 (Model-Native Evaluation)** with **both a single script and an interactive notebook** for benchmarking the three ML models in their native paradigms using RMSE, MSE, R², MAE, and MAPE.

---

## Three Models Evaluated

| Model | Type | Training Script | Native Paradigm |
|-------|------|-----------------|-----------------|
| **SARIMA** | Seasonal ARIMA (time series) | `scripts/train_sarima.py` | Per-farm 6-month yield forecasts |
| **XGBoost** | Gradient Boosted Trees (supervised) | `scripts/train_xgboost.py` | Monthly yield prediction on feature matrix |
| **GNN/GraphSAGE** | Graph Neural Network (unsupervised) | `scripts/train_gnn.py` | Farm clustering (6 clusters) + anomaly detection |

---

## Files Created

### 1. Single Script — `scripts/benchmark_regression.py`
Reproducible, CI-ready benchmark runner.

**Usage:**
```bash
# Full benchmark (saves to results/benchmarks/YYYYMMDD_HHMMSS/)
python3 scripts/benchmark_regression.py

# Custom test seasons and horizons
python3 scripts/benchmark_regression.py --test-seasons 2023 2024 --horizons 1 3 6

# Skip specific models
python3 scripts/benchmark_regression.py --skip-gnn
```

**Outputs (in timestamped directory):**
- `benchmark_summary.json` — Aggregate metrics for all models
- `sarima_detailed.csv` — Per-farm, per-horizon results
- `xgb_cv_detailed.csv` — TimeSeriesSplit CV fold results
- `xgb_per_farm.csv` — Per-farm holdout errors
- `xgb_predictions.csv` — Actual vs predicted (holdout)
- `gnn_per_cluster.csv` — Cluster quality + anomaly stats

---

### 2. Interactive Notebook — `notebooks/benchmark_regression.ipynb`
8 cells for deep-dive exploration:

1. **Load existing results** — Auto-detects latest benchmark run
2. **Run benchmark inline** — Via subprocess
3. **SARIMA analysis** — Boxplots by horizon, per-farm RMSE/R² distributions, best/worst farms
4. **XGBoost CV** — Fold-by-fold metrics, summary statistics
5. **XGBoost Holdout** — Actual vs predicted scatter, residuals, error distributions, MAPE by yield level
6. **GNN Analysis** — Anomaly score distributions, cluster quality (Silhouette/DB/CH), top anomalous farms
7. **Cross-model comparison** — With explicit caveats
8. **Correlation analysis** — SARIMA vs XGBoost per-farm errors, disagreement cases

---

## Evaluation Methodology (Model-Native)

### SARIMA — Expanding Window Time-Series CV
- **Train:** Seasons 2010 → 2022 (expanding)
- **Test:** Seasons 2023, 2024
- **Horizons:** 1, 3, 6 months ahead
- **Per farm:** Re-fits `auto_arima` on each expanding window
- **Metrics:** RMSE, MSE, R², MAE, MAPE per horizon

### XGBoost — TimeSeriesSplit CV + Holdout
- **CV:** 5-fold `TimeSeriesSplit` (no shuffle → no leakage)
- **Holdout:** Seasons 2023–2024 (row-level)
- **Fixes:** Original `KFold(shuffle=True)` leakage
- **Adds:** R² to existing MAE/RMSE/MAPE reporting
- **Metrics:** Per-fold + aggregate + per-farm breakdown

### GNN — Unsupervised Cluster Quality + Anomaly Analysis
- **Cluster Quality:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz
- **Anomaly Scores:** Distribution stats, per-cluster breakdown, top-15 anomalous farms
- **Correlation:** Anomaly score vs. yield variability (CV, std) where farm data available
- **Note:** No regression target — evaluates representation quality, not prediction

---

## Dependencies

```bash
# Core (already in Docker/requirements.txt)
pip install matplotlib seaborn jupyter pyarrow pmdarima xgboost torch torch-geometric

# Already satisfied
# pandas, numpy, scikit-learn, pymongo, joblib
```

**Recommended:** Run inside the `ml_engine` Docker container which has all dependencies pre-installed.

---

## Requirements

- **MongoDB** running with `chaimterics` database containing:
  - `farms` collection (historical seasons for SARIMA evaluation)
  - `gnn_outputs` collection (embeddings, cluster_id, anomaly_score)
- **Pre-trained models** in `models/` (already present from training runs):
  - `models/sarima/{factory}/{member_no}.pkl` + `_meta.json`
  - `models/xgb_yield.pkl`, `models/preprocessor.pkl`
  - `models/gnn_norm_params.pkl`

---

## Cross-Model Comparison Caveats

> ⚠️ **Direct comparison is limited by paradigm differences:**

| Aspect | SARIMA | XGBoost | GNN |
|--------|--------|---------|-----|
| **Test Set** | Per-farm time series (2023-24) | Row-level holdout (2023-24) | N/A (unsupervised) |
| **Prediction Unit** | Monthly kg (per farm) | Monthly kg (per row) | Cluster ID + anomaly score |
| **Paradigm** | Time-series forecasting | Supervised regression | Unsupervised representation |
| **Comparable** | 1-month horizon vs holdout | ✓ (with caveats) | Not directly |

**Best practice:** Use cross-model correlation analysis (Cell 8 in notebook) to find farms where models agree/disagree, rather than aggregate metric comparison.

---

## Output Structure

```
results/benchmarks/
└── 20261111_143022/           # Timestamped run directory
    ├── benchmark_summary.json  # Aggregate metrics (all models)
    ├── sarima_detailed.csv     # Per-farm, per-horizon
    ├── xgb_cv_detailed.csv     # 5-fold CV results
    ├── xgb_per_farm.csv        # Per-farm holdout errors
    ├── xgb_predictions.csv     # Actual vs predicted (240 farms × 24 months)
    └── gnn_per_cluster.csv     # Cluster quality + anomaly stats
```

---

## Next Steps

1. **Install dependencies** (if not using Docker):
   ```bash
   pip install matplotlib seaborn jupyter pmdarima xgboost torch torch-geometric
   ```

2. **Start MongoDB** with `chaimterics` database:
   ```bash
   docker compose up -d mongodb
   ```

3. **Run benchmark:**
   ```bash
   python scripts/benchmark_regression.py
   ```

4. **Explore interactively:**
   ```bash
   jupyter notebook notebooks/benchmark_regression.ipynb
   ```

---

## Related Files

| Category | Files |
|----------|-------|
| **Model Training** | `scripts/train_sarima.py`, `scripts/train_xgboost.py`, `scripts/train_gnn.py` |
| **Data Pipeline** | `scripts/preprocess.py`, `scripts/build_graph.py`, `data/data_generator.py` |
| **Model Artifacts** | `models/xgb_yield.pkl`, `models/preprocessor.pkl`, `models/sarima/`, `models/gnn_norm_params.pkl` |
| **API/Inference** | `api/blueprints/insights.py`, `api/blueprints/farms.py` |

---
