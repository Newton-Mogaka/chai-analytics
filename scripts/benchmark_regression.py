#!/usr/bin/env python3
"""
ChaiAnalytics — Model-Native Regression Benchmark
==================================================

Evaluates each model in its native paradigm:
- SARIMA: Expanding-window time-series CV (forecast horizons 1, 3, 6 months)
- XGBoost: TimeSeriesSplit CV + holdout test set (fixes KFold leakage)
- GNN: Cluster quality metrics + anomaly score analysis

Metrics: RMSE, MSE, R², MAE, MAPE

Run:
    python scripts/benchmark_regression.py
    python scripts/benchmark_regression.py --output-dir results/benchmark_20241110
    python scripts/benchmark_regression.py --notebook  # Generate notebook too
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from pmdarima import auto_arima

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
loaddotenv()

# ─── Paths ───
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SARIMA_DIR = MODELS_DIR / "sarima"
RESULTS_DIR = BASE_DIR / "results" / "benchmarks"

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://maggie:tDircFKbxj5KA6DN@diggy.rug2w.mongodb.net/chaimterics?retryWrites=true&w=majority&appName=Diggy")


# ─── Utility Functions ───
def load_mongo_db():
    """Connect to MongoDB and return database."""
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client.get_default_database()


def load_preprocess_artifacts(models_dir: Path):
    """Load X, y, row_ids, preprocessor from preprocess.py artifacts."""
    X = pd.read_parquet(models_dir / "X_features.parquet")
    y = pd.read_parquet(models_dir / "y_target.parquet")["monthly_kg"]
    row_ids = pd.read_parquet(models_dir / "row_ids.parquet")
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    return X, y, row_ids, preprocessor


def load_xgboost_model(models_dir: Path):
    """Load trained XGBoost model."""
    return joblib.load(models_dir / "xgb_yield.pkl")


def load_sarima_manifest(models_dir: Path):
    """Load SARIMA farm manifest."""
    manifest_path = models_dir / "sarima" / "farm_manifest.parquet"
    if manifest_path.exists():
        return pd.read_parquet(manifest_path)
    return None


def load_sarima_model(factory: str, member_no: str, models_dir: Path):
    """Load a specific farm's SARIMA model."""
    model_path = models_dir / "sarima" / factory / f"{member_no}.pkl"
    meta_path = models_dir / "sarima" / factory / f"{member_no}_meta.json"
    if not model_path.exists():
        return None, None
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def load_gnn_outputs(db):
    """Load GNN outputs from MongoDB."""
    cursor = db.gnn_outputs.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    return df


def load_gnn_norm_params(models_dir: Path):
    """Load GNN normalization parameters."""
    path = models_dir / "gnn_norm_params.pkl"
    if path.exists():
        return joblib.load(path)
    return None


# ─── SARIMA Evaluation ───
def evaluate_sarima(models_dir: Path, test_seasons: list = None, horizons: list = None):
    """
    Evaluate SARIMA using expanding-window time-series CV.
    For each farm with sufficient data, train on expanding windows and forecast horizons.
    """
    if test_seasons is None:
        test_seasons = [2023, 2024]
    if horizons is None:
        horizons = [1, 3, 6]

    print("\n" + "="*70)
    print("SARIMA EVALUATION — Expanding Window Time-Series CV")
    print("="*70)

    manifest = load_sarima_manifest(models_dir)
    if manifest is None:
        print("  [WARN] No SARIMA manifest found. Skipping.")
        return None

    ok_farms = manifest[manifest["status"] == "ok"]
    print(f"  {len(ok_farms)} farms with successfully trained SARIMA models")

    # Connect to MongoDB to get farm historical data
    db = load_mongo_db()
    farms = list(db.farms.find(
        {"ktda_member_no": {"$in": ok_farms["member_no"].tolist()}},
        {"ktda_member_no": 1, "factory_code": 1, "historical_seasons": 1, "_id": 0}
    ))

    results = []

    for farm in farms:
        member_no = farm["ktda_member_no"]
        factory = farm["factory_code"]

        # Build the full monthly series
        series = build_farm_series(farm)
        if len(series) < 24:
            continue

        # Expanding window evaluation
        # Train on data up to season S, test on season S+1
        seasons = sorted(set(s["season_year"] for s in farm.get("historical_seasons", [])))
        test_seasons_set = set(test_seasons)

        for test_season in test_seasons_set:
            if test_season not in seasons:
                continue

            train_seasons = [s for s in seasons if s < test_season]
            if len(train_seasons) < 2:
                continue

            train_series = build_farm_series_from_seasons(farm, train_seasons)
            if len(train_series) < 24:
                continue

            try:
                model = auto_arima(
                    train_series,
                    seasonal=True, m=12,
                    stepwise=True, suppress_warnings=True, error_action="ignore",
                    max_p=3, max_q=3, max_P=2, max_Q=2,
                    information_criterion="aic", n_jobs=1
                )

                # Get test season actuals
                test_season_data = next(s for s in farm["historical_seasons"] if s["season_year"] == test_season)
                actuals = test_season_data.get("monthly_kg", [None]*12)
                actuals = [a for a in actuals if a is not None]

                if not actuals:
                    continue

                # Forecast and evaluate at different horizons
                for h in horizons:
                    if h > len(actuals):
                        continue
                    forecast, conf_int = model.predict(n_periods=h, return_conf_int=True, alpha=0.2)
                    preds = [max(0.0, float(v)) for v in forecast[:h]]
                    y_true = actuals[:h]

                    rmse = np.sqrt(mean_squared_error(y_true, preds))
                    mse = mean_squared_error(y_true, preds)
                    r2 = r2_score(y_true, preds) if len(y_true) > 1 else np.nan
                    mae = mean_absolute_error(y_true, preds)
                    mape = np.mean(np.abs((np.array(y_true) - np.array(preds)) / np.array(y_true))) * 100

                    results.append({
                        "model": "SARIMA",
                        "member_no": member_no,
                        "factory": factory,
                        "test_season": test_season,
                        "horizon": h,
                        "rmse": rmse, "mse": mse, "r2": r2, "mae": mae, "mape": mape,
                        "n_obs": len(train_series)
                    })

            except Exception as e:
                print(f"    [FAIL] {member_no} season {test_season}: {e}")
                continue

    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("  No SARIMA evaluation results generated.")
        return None

    # Aggregate results
    print(f"\n  Generated {len(df_results)} evaluation points")
    agg = df_results.groupby("horizon")[["rmse", "mse", "r2", "mae", "mape"]].agg(["mean", "std", "count"])
    print(f"\n  Aggregate by horizon:")
    print(agg.to_string())

    return df_results


def build_farm_series(farm: dict) -> pd.Series:
    """Build full monthly yield series from farm historical seasons."""
    rows = []
    for season in sorted(farm.get("historical_seasons", []), key=lambda s: s["season_year"]):
        monthly_kg = season.get("monthly_kg", [])
        for kg in monthly_kg:
            rows.append(float(kg) if kg is not None else np.nan)
    series = pd.Series(rows, dtype=float)
    return series.ffill().bfill()


def build_farm_series_from_seasons(farm: dict, season_years: list) -> pd.Series:
    """Build monthly yield series from specific seasons only."""
    rows = []
    for season in sorted(farm.get("historical_seasons", []), key=lambda s: s["season_year"]):
        if season["season_year"] not in season_years:
            continue
        monthly_kg = season.get("monthly_kg", [])
        for kg in monthly_kg:
            rows.append(float(kg) if kg is not None else np.nan)
    series = pd.Series(rows, dtype=float)
    return series.ffill().bfill()


# ─── XGBoost Evaluation ───
def evaluate_xgboost(models_dir: Path, test_seasons: list = None):
    """
    Evaluate XGBoost using TimeSeriesSplit CV (no leakage) + holdout test set.
    """
    if test_seasons is None:
        test_seasons = [2023, 2024]

    print("\n" + "="*70)
    print("XGBOOST EVALUATION — TimeSeriesSplit CV + Holdout")
    print("="*70)

    X, y, row_ids, preprocessor = load_preprocess_artifacts(models_dir)
    model = load_xgboost_model(models_dir)

    print(f"  Full dataset: {X.shape[0]} rows × {X.shape[1]} features")

    # Identify test rows (2023-2024 seasons)
    test_mask = row_ids["season_year"].isin(test_seasons)
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    row_ids_test = row_ids[test_mask]

    print(f"  Train: {X_train.shape[0]} rows (seasons {sorted(row_ids[train_mask]['season_year'].unique())})")
    print(f"  Test:  {X_test.shape[0]} rows (seasons {sorted(row_ids[test_mask]['season_year'].unique())})")

    # Transform features
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # ─── TimeSeriesSplit CV on Training Data ───
    print("\n  Running TimeSeriesSplit CV (5 splits)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = xgb.XGBRegressor(**model.get_params())
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = m.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mse = mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        mask = y_val > 10
        mape = np.mean(np.abs((y_val[mask] - preds[mask]) / y_val[mask])) * 100 if mask.any() else np.nan

        cv_results.append({
            "fold": fold + 1,
            "train_size": len(tr_idx),
            "val_size": len(val_idx),
            "rmse": rmse, "mse": mse, "r2": r2, "mae": mae, "mape": mape
        })
        print(f"    Fold {fold+1}: RMSE={rmse:.1f}  R²={r2:.3f}  MAE={mae:.1f}  MAPE={mape:.1f}%")

    cv_df = pd.DataFrame(cv_results)
    cv_summary = cv_df[["rmse", "mse", "r2", "mae", "mape"]].agg(["mean", "std"])
    print(f"\n  CV Summary:")
    print(cv_summary.to_string())

    # ─── Holdout Test Set Evaluation ───
    print("\n  Evaluating on holdout test set (2023-2024)...")
    preds_test = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    mse = mean_squared_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)
    mae = mean_absolute_error(y_test, preds_test)
    mask = y_test > 10
    mape = np.mean(np.abs((y_test[mask] - preds_test[mask]) / y_test[mask])) * 100 if mask.any() else np.nan

    test_result = {
        "model": "XGBoost",
        "eval_type": "holdout",
        "test_seasons": test_seasons,
        "n_test": len(y_test),
        "rmse": rmse, "mse": mse, "r2": r2, "mae": mae, "mape": mape
    }

    print(f"\n  Holdout Test Results:")
    print(f"    RMSE: {rmse:.2f} kg")
    print(f"    MSE:  {mse:.2f} kg²")
    print(f"    R²:   {r2:.4f}")
    print(f"    MAE:  {mae:.2f} kg")
    print(f"    MAPE: {mape:.2f}%")

    # Per-farm breakdown on test set
    test_df = row_ids_test.copy()
    test_df["actual"] = y_test.values
    test_df["predicted"] = preds_test
    test_df["error"] = test_df["predicted"] - test_df["actual"]
    test_df["abs_error"] = test_df["error"].abs()
    test_df["ape"] = np.where(test_df["actual"] > 10,
                               test_df["abs_error"] / test_df["actual"] * 100, np.nan)

    farm_errors = test_df.groupby("ktda_member_no").agg(
        n_obs=("actual", "count"),
        mae=("abs_error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
        mape=("ape", "mean")
    ).reset_index()

    return {
        "cv_results": cv_df,
        "cv_summary": cv_summary,
        "holdout": test_result,
        "per_farm": farm_errors,
        "predictions": test_df
    }


# ─── GNN Evaluation ───
def evaluate_gnn(models_dir: Path):
    """
    Evaluate GNN using cluster quality metrics and anomaly score analysis.
    Since GNN is unsupervised, we evaluate:
    - Cluster quality (silhouette, Davies-Bouldin, Calinski-Harabasz)
    - Anomaly score distribution
    - Correlation of anomaly scores with yield variability
    """
    print("\n" + "="*70)
    print("GNN EVALUATION — Cluster Quality + Anomaly Analysis")
    print("="*70)

    db = load_mongo_db()
    gnn_outputs = load_gnn_outputs(db)
    norm_params = load_gnn_norm_params(models_dir)

    if gnn_outputs.empty:
        print("  [WARN] No GNN outputs found in MongoDB. Run train_gnn.py first.")
        return None

    print(f"  Loaded {len(gnn_outputs)} farm GNN outputs")

    # ─── Cluster Quality Metrics ───
    embeddings = np.stack(gnn_outputs["embedding"].values)
    cluster_ids = gnn_outputs["cluster_id"].values
    anomaly_scores = gnn_outputs["anomaly_score"].values

    n_clusters = len(np.unique(cluster_ids))
    print(f"  Clusters: {n_clusters}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    # Silhouette score (higher = better separation)
    if n_clusters > 1 and n_clusters < len(embeddings):
        sil = silhouette_score(embeddings, cluster_ids)
        db_idx = davies_bouldin_score(embeddings, cluster_ids)
        ch_idx = calinski_harabasz_score(embeddings, cluster_ids)
    else:
        sil = db_idx = ch_idx = np.nan

    print(f"\n  Cluster Quality Metrics:")
    print(f"    Silhouette Score:      {sil:.4f}  (higher = better, range [-1, 1])")
    print(f"    Davies-Bouldin Index:  {db_idx:.4f}  (lower = better)")
    print(f"    Calinski-Harabasz:     {ch_idx:.4f}  (higher = better)")

    # ─── Anomaly Score Distribution ───
    print(f"\n  Anomaly Score Distribution:")
    print(f"    Mean:  {anomaly_scores.mean():.4f}")
    print(f"    Std:   {anomaly_scores.std():.4f}")
    print(f"    Min:   {anomaly_scores.min():.4f}")
    print(f"    Max:   {anomaly_scores.max():.4f}")
    print(f"    >0.7:  {(anomaly_scores > 0.7).sum()} farms ({(anomaly_scores > 0.7).mean()*100:.1f}%)")
    print(f"    >0.5:  {(anomaly_scores > 0.5).sum()} farms ({(anomaly_scores > 0.5).mean()*100:.1f}%)")

    # Per-cluster anomaly stats
    print(f"\n  Anomaly Scores by Cluster:")
    for cid in sorted(np.unique(cluster_ids)):
        mask = cluster_ids == cid
        scores = anomaly_scores[mask]
        print(f"    Cluster {cid}: {mask.sum()} farms, mean={scores.mean():.3f}, max={scores.max():.3f}")

    # ─── Correlation with Yield Variability (if farm data available) ───
    # Load farm yield stats to correlate with anomaly scores
    farms = list(db.farms.find(
        {"ktda_member_no": {"$in": gnn_outputs["ktda_member_no"].tolist()}},
        {"ktda_member_no": 1, "historical_seasons": 1, "_id": 0}
    ))

    yield_stats = []
    for farm in farms:
        member_no = farm["ktda_member_no"]
        all_yields = []
        for season in farm.get("historical_seasons", []):
            monthly = season.get("monthly_kg", [])
            all_yields.extend([y for y in monthly if y is not None])
        if len(all_yields) >= 12:
            yield_stats.append({
                "ktda_member_no": member_no,
                "yield_mean": np.mean(all_yields),
                "yield_std": np.std(all_yields),
                "yield_cv": np.std(all_yields) / np.mean(all_yields) if np.mean(all_yields) > 0 else np.nan,
                "n_months": len(all_yields)
            })

    if yield_stats:
        yield_df = pd.DataFrame(yield_stats)
        merged = gnn_outputs[["ktda_member_no", "cluster_id", "anomaly_score"]].merge(yield_df, on="ktda_member_no")

        if len(merged) > 10:
            corr_cv = merged["anomaly_score"].corr(merged["yield_cv"])
            corr_std = merged["anomaly_score"].corr(merged["yield_std"])
            print(f"\n  Anomaly Score vs Yield Variability Correlation:")
            print(f"    vs Yield CV:     {corr_cv:.4f}")
            print(f"    vs Yield Std:    {corr_std:.4f}")

    return {
        "cluster_quality": {
            "silhouette": float(sil) if not np.isnan(sil) else None,
            "davies_bouldin": float(db_idx) if not np.isnan(db_idx) else None,
            "calinski_harabasz": float(ch_idx) if not np.isnan(ch_idx) else None,
            "n_clusters": int(n_clusters),
            "n_farms": int(len(embeddings))
        },
        "anomaly_distribution": {
            "mean": float(anomaly_scores.mean()),
            "std": float(anomaly_scores.std()),
            "min": float(anomaly_scores.min()),
            "max": float(anomaly_scores.max()),
            "pct_above_07": float((anomaly_scores > 0.7).mean() * 100),
            "pct_above_05": float((anomaly_scores > 0.5).mean() * 100)
        },
        "per_cluster": gnn_outputs.groupby("cluster_id").agg(
            n_farms=("ktda_member_no", "count"),
            mean_anomaly=("anomaly_score", "mean"),
            max_anomaly=("anomaly_score", "max")
        ).reset_index().to_dict("records")
    }


# ─── Summary Report ───
def generate_summary_report(sarima_results, xgb_results, gnn_results, output_dir: Path):
    """Generate a comprehensive summary report."""

    summary = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "models_evaluated": [],
        "sarima": None,
        "xgboost": None,
        "gnn": None
    }

    if sarima_results is not None and len(sarima_results) > 0:
        summary["models_evaluated"].append("SARIMA")
        # Aggregate by horizon
        agg = sarima_results.groupby("horizon")[["rmse", "mse", "r2", "mae", "mape"]].mean()
        summary["sarima"] = {
            "n_farms_evaluated": int(sarima_results["member_no"].nunique()),
            "n_evaluation_points": int(len(sarima_results)),
            "by_horizon": agg.round(4).to_dict()
        }

    if xgb_results is not None:
        summary["models_evaluated"].append("XGBoost")
        holdout = xgb_results["holdout"]
        cv_mean = xgb_results["cv_summary"].loc["mean"]
        summary["xgboost"] = {
            "cv_metrics": cv_mean.round(4).to_dict(),
            "holdout_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                                for k, v in holdout.items() if k != "eval_type"}
        }

    if gnn_results is not None:
        summary["models_evaluated"].append("GNN")
        summary["gnn"] = {
            "cluster_quality": gnn_results["cluster_quality"],
            "anomaly_distribution": gnn_results["anomaly_distribution"]
        }

    # Save JSON summary
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")

    # Save detailed CSVs
    if sarima_results is not None:
        sarima_results.to_csv(output_dir / "sarima_detailed.csv", index=False)
    if xgb_results is not None:
        xgb_results["cv_results"].to_csv(output_dir / "xgb_cv_detailed.csv", index=False)
        xgb_results["per_farm"].to_csv(output_dir / "xgb_per_farm.csv", index=False)
        xgb_results["predictions"].to_csv(output_dir / "xgb_predictions.csv", index=False)
    if gnn_results is not None:
        pd.DataFrame(gnn_results["per_cluster"]).to_csv(output_dir / "gnn_per_cluster.csv", index=False)

    # Print final summary table
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    if sarima_results is not None and len(sarima_results) > 0:
        print("\nSARIMA (by forecast horizon):")
        agg = sarima_results.groupby("horizon")[["rmse", "r2", "mae"]].mean()
        print(agg.round(2).to_string())

    if xgb_results is not None:
        h = xgb_results["holdout"]
        print(f"\nXGBoost (holdout 2023-2024):")
        print(f"  RMSE: {h['rmse']:.2f}  MSE: {h['mse']:.2f}  R²: {h['r2']:.4f}  MAE: {h['mae']:.2f}  MAPE: {h['mape']:.2f}%")

    if gnn_results is not None:
        cq = gnn_results["cluster_quality"]
        print(f"\nGNN (cluster quality):")
        print(f"  Silhouette: {cq['silhouette']:.4f}  Davies-Bouldin: {cq['davies_bouldin']:.4f}  Calinski-Harabasz: {cq['calinski_harabasz']:.4f}")

    return summary


# ─── Main ───
def main():
    parser = argparse.ArgumentParser(description="Model-native regression benchmark")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/benchmarks/YYYYMMDD_HHMMSS)")
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2023, 2024],
                        help="Test seasons for holdout evaluation")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6],
                        help="Forecast horizons for SARIMA evaluation")
    parser.add_argument("--skip-sarima", action="store_true", help="Skip SARIMA evaluation")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost evaluation")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN evaluation")
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("="*70)
    print("CHAI ANALYTICS — MODEL-NATIVE REGRESSION BENCHMARK")
    print("="*70)
    print(f"Test seasons: {args.test_seasons}")
    print(f"SARIMA horizons: {args.horizons}")

    # Run evaluations
    sarima_results = None
    xgb_results = None
    gnn_results = None

    if not args.skip_sarima:
        sarima_results = evaluate_sarima(MODELS_DIR, args.test_seasons, args.horizons)

    if not args.skip_xgboost:
        xgb_results = evaluate_xgboost(MODELS_DIR, args.test_seasons)

    if not args.skip_gnn:
        gnn_results = evaluate_gnn(MODELS_DIR)

    # Generate summary
    summary = generate_summary_report(sarima_results, xgb_results, gnn_results, output_dir)

    print(f"\n✓ Benchmark complete. Results in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())