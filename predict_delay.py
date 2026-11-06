"""Production scoring script for new parcels."""
import os
import warnings
import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

warnings.filterwarnings("ignore")


def predict_delay():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data and models
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments_df = pd.read_csv("outputs/hub_segments.csv")

    impute_df = pd.read_csv("models/impute_values.csv")
    stored_medians = dict(zip(impute_df["column"], impute_df["median"]))

    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    model = joblib.load("models/delay_model.pkl")

    # Preserve original shipment_id order
    original_order = new_raw["shipment_id"].tolist()

    # 2. Clean shipments with stored medians
    clean_new, _ = clean_shipments(new_raw, medians=stored_medians)

    # 3. Build features
    features_df = build_features(clean_new, hubs)

    # Reorder rows to match original input file order
    features_df["_sort_key"] = features_df["shipment_id"].map({sid: i for i, sid in enumerate(original_order)})
    features_df = features_df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

    # 4. Score with model
    X_new = features_df[feature_order]
    probs = model.predict_proba(X_new)[:, 1].round(4)
    features_df["delay_probability"] = probs
    features_df["flagged"] = (probs >= threshold).astype(int)

    # 5. Join hub segment
    hub_seg_map = dict(zip(hub_segments_df["hub_id"], hub_segments_df["segment"]))
    features_df["segment"] = features_df["hub_id"].map(hub_seg_map)

    # 6. Apply business rules for actions
    def determine_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            else:
                return "Proactive customer call"
        return "No action"

    features_df["action"] = features_df.apply(determine_action, axis=1)

    # 7. Write outputs/new_shipment_actions.csv
    out_cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    actions_df = features_df[out_cols]
    actions_df.to_csv("outputs/new_shipment_actions.csv", index=False)

    print(f"Scoring complete. Total parcels: {len(actions_df)}, Flagged: {actions_df['flagged'].sum()}")
    print(actions_df["action"].value_counts())


if __name__ == "__main__":
    predict_delay()

