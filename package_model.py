"""Package the final classification model and artifacts for production serving."""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments

warnings.filterwarnings("ignore")


def package_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Read config from Part C
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth_val = config["max_depth"]
    max_depth = None if str(max_depth_val).lower() in ["unlimited", "none", "nan"] else int(max_depth_val)
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # 2. Rebuild Part C's stratified split
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 3. Fit final pipeline on training rows
    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": 42,
    }
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("model", RandomForestClassifier(**model_params)),
    ])
    final_pipeline.fit(X_train, y_train)

    # 4. Save artifacts in models/
    # (a) Model pickle
    joblib.dump(final_pipeline, "models/delay_model.pkl")

    # (b) Threshold
    pd.DataFrame([{"threshold": round(threshold, 4)}]).to_csv("models/delay_threshold.csv", index=False)

    # (c) Feature order
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

    # (d) Impute values from clean_shipments on historical data
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)
    impute_df = pd.DataFrame([
        {"column": k, "median": round(v, 4)} for k, v in medians.items()
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)

    # 5. Reload verification
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    in_mem_probs = final_pipeline.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]

    diff = float(np.max(np.abs(in_mem_probs - reloaded_probs)))
    test_rows_scored = len(X_test)

    reload_checks = [
        ("test_rows_scored", str(test_rows_scored)),
        ("reload_max_abs_probability_diff", str(round(diff, 8))),
        ("threshold", str(round(threshold, 4))),
    ]
    pd.DataFrame(reload_checks, columns=["metric", "value"]).to_csv(
        "outputs/part_d_reload_check.csv", index=False
    )

    print(f"Model packaged successfully. Reload max diff: {diff}. Threshold: {threshold}")


if __name__ == "__main__":
    package_model()

