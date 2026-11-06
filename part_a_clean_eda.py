"""Part A — Data Foundation and Delay Evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS


def run_part_a():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    raw_path = "data/raw_shipments.csv"
    hubs_path = "data/hubs.csv"

    raw = pd.read_csv(raw_path)
    hubs = pd.read_csv(hubs_path)

    # 1. Pre-cleaning metrics
    raw_rows = len(raw)
    dupes_mask = raw.duplicated()
    duplicate_rows_removed = int(dupes_mask.sum())
    dedup = raw.drop_duplicates().copy()
    rows_after_dedup = len(dedup)

    missing_distance_km_before = int(dedup["distance_km"].isna().sum())
    # missing_parcel_weight_before does not include sentinel rows (> 100)
    missing_parcel_weight_before = int(dedup["parcel_weight_kg"].isna().sum())
    sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())
    missing_declared_value_before = int(dedup["declared_value"].isna().sum())
    missing_payment_mode_before = int(dedup["payment_mode"].isna().sum())
    service_type_spellings_before = int(dedup["service_type"].nunique())

    # 2. Clean shipments
    clean_df, medians = clean_shipments(raw)
    service_type_spellings_after = int(clean_df["service_type"].nunique())

    # 3. Build features
    modelling_df = build_features(clean_df, hubs)
    rows_after_hub_merge = len(modelling_df)

    # 4. Add target is_delayed
    modelling_df["is_delayed"] = (modelling_df["delivery_hours"] > modelling_df["promised_hours"]).astype(int)

    # Save data/modelling_table.csv
    modelling_df.to_csv("data/modelling_table.csv", index=False)

    # 5. Check nulls in 17 features + delivery_hours + is_delayed
    check_cols = FEATURES + ["delivery_hours", "is_delayed"]
    nulls_in_features_and_target = int(modelling_df[check_cols].isna().sum().sum())

    delayed_count = int(modelling_df["is_delayed"].sum())
    delay_rate = float(modelling_df["is_delayed"].mean())

    # Delivery hours quartiles and outliers (1.5 x IQR)
    hours = modelling_df["delivery_hours"]
    q1 = float(hours.quantile(0.25))
    q3 = float(hours.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((hours < lower_bound) | (hours > upper_bound)).sum())

    # Declared value metrics on filled declared_value_inr and log_declared_value
    val_inr = modelling_df["declared_value_inr"]
    log_val = modelling_df["log_declared_value"]
    declared_value_mean_inr = float(val_inr.mean())
    declared_value_median_inr = float(val_inr.median())
    log_declared_value_mean = float(log_val.mean())
    log_declared_value_median = float(log_val.median())

    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", round(medians["distance_km"], 4)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", round(delay_rate, 4)),
        ("hours_q1", round(q1, 4)),
        ("hours_q3", round(q3, 4)),
        ("hours_lower_bound", round(lower_bound, 4)),
        ("hours_upper_bound", round(upper_bound, 4)),
        ("hours_outlier_count", outlier_count),
        ("declared_value_mean_inr", round(declared_value_mean_inr, 4)),
        ("declared_value_median_inr", round(declared_value_median_inr, 4)),
        ("log_declared_value_mean", round(log_declared_value_mean, 4)),
        ("log_declared_value_median", round(log_declared_value_median, 4)),
    ]

    formatted_checks = []
    for m, v in checks:
        if isinstance(v, (int, np.integer)):
            formatted_checks.append((m, str(int(v))))
        else:
            formatted_checks.append((m, f"{round(float(v), 4):.4f}".rstrip("0").rstrip(".") if "." in str(round(float(v), 4)) else str(v)))

    checks_df = pd.DataFrame(formatted_checks, columns=["metric", "value"])
    checks_df.to_csv("outputs/part_a_checks.csv", index=False)

    # 6. Delay evidence table
    modelling_df["distance_band"] = pd.cut(
        modelling_df["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
    )

    group_rows = []
    # service_type
    for st in ["Economy", "Express", "Standard"]:
        sub = modelling_df[modelling_df["service_type"] == st]
        group_rows.append({
            "dimension": "service_type",
            "group": st,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # hub_tier
    for ht in [1, 2, 3]:
        sub = modelling_df[modelling_df["hub_tier"] == ht]
        group_rows.append({
            "dimension": "hub_tier",
            "group": str(ht),
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # distance_band
    for db in ["0-200", "201-400", "401-600", "601+"]:
        sub = modelling_df[modelling_df["distance_band"] == db]
        group_rows.append({
            "dimension": "distance_band",
            "group": db,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    groups_df = pd.DataFrame(group_rows)
    groups_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)

    # 7. Correlations
    numeric_cols = modelling_df.select_dtypes(include=[np.number])
    correlations = numeric_cols.corr().round(4)
    correlations.to_csv("outputs/part_a_correlations.csv", index=True, index_label="feature")

    print("Part A complete. Written data/modelling_table.csv and outputs/part_a_*.csv")


if __name__ == "__main__":
    run_part_a()
