"""Part B — Delivery-Hours Regression."""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES

warnings.filterwarnings("ignore")


def run_part_b():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # 1. Split 70% / 15% / 15%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 2. Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
    test_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))

    checks = [
        ("train_rows", str(len(X_train))),
        ("val_rows", str(len(X_val))),
        ("test_rows", str(len(X_test))),
        ("train_scaled_max_abs_mean", str(round(train_scaled_max_abs_mean, 8))),
        ("test_scaled_max_abs_mean", str(round(test_scaled_max_abs_mean, 4))),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)

    # 3. Mean baseline
    y_base_pred = np.full_like(y_val, y_train.mean())
    base_mae = mean_absolute_error(y_val, y_base_pred)
    base_mse = mean_squared_error(y_val, y_base_pred)
    base_rmse = np.sqrt(base_mse)
    base_r2 = r2_score(y_val, y_base_pred)

    # 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    tree_csv_rows = []
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_pred = dt.predict(X_train)
        val_pred = dt.predict(X_val)
        tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
        v_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        d_label = "unlimited" if d is None else str(d)
        tree_csv_rows.append({
            "max_depth": d_label,
            "train_rmse": round(tr_rmse, 4),
            "val_rmse": round(v_rmse, 4),
        })
        tree_records.append((d_label, tr_rmse, v_rmse, d))

    pd.DataFrame(tree_csv_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)

    # Best depth with lowest val_rmse (ties -> smaller depth)
    best_tree_item = min(tree_records, key=lambda x: (x[2], x[3] if x[3] is not None else 999))
    best_depth = best_tree_item[3]

    # 5. Four candidate models
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = [
        {
            "model": "mean_baseline",
            "split": "val",
            "MAE": round(base_mae, 4),
            "MSE": round(base_mse, 4),
            "RMSE": round(base_rmse, 4),
            "R2": round(base_r2, 4),
        }
    ]

    candidate_results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        candidate_results[name] = {"model": model, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })

    # 6. Learning-rate experiment
    lr_rows = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        preds = gb.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        lr_rows.append({
            "learning_rate": lr,
            "val_rmse": round(rmse, 4),
            "val_r2": round(r2, 4),
        })
    pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    # 7. Cross-validated comparison and final model
    ranked_candidates = sorted(candidate_results.items(), key=lambda item: item[1]["RMSE"])
    top2_names = [ranked_candidates[0][0], ranked_candidates[1][0]]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rows = []
    for name in top2_names:
        mod = candidates[name]
        scores = cross_val_score(mod, X_train_val, y_train_val, cv=kf)
        cv_rows.append({
            "model": name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

    # Final model: lowest validation RMSE
    final_name = ranked_candidates[0][0]
    final_model = candidates[final_name]
    test_preds = final_model.predict(X_test)
    t_mae = mean_absolute_error(y_test, test_preds)
    t_mse = mean_squared_error(y_test, test_preds)
    t_rmse = np.sqrt(t_mse)
    t_r2 = r2_score(y_test, test_preds)

    metrics_rows.append({
        "model": final_name,
        "split": "test",
        "MAE": round(t_mae, 4),
        "MSE": round(t_mse, 4),
        "RMSE": round(t_rmse, 4),
        "R2": round(t_r2, 4),
    })

    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
    print(f"Part B complete. Best model: {final_name}. Test R2: {t_r2:.4f}")


if __name__ == "__main__":
    run_part_b()
