"""Shared preprocessing constants and functions for the Delhivery Delay Intelligence Pipeline."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    """Clean raw shipment DataFrame and impute missing numerical values with medians.
    
    Steps applied:
    1. Drop exact duplicate rows.
    2. Normalize service_type to title case without leading/trailing whitespace.
    3. Fill missing payment_mode with 'Unknown'.
    4. Clean declared_value into declared_value_inr (float) and drop declared_value.
    5. Convert dispatch_time to datetime.
    6. Sentinel replacement: parcel_weight_kg > 100 set to NaN.
    7. Fill missing distance_km, parcel_weight_kg, declared_value_inr with medians.
    
    Returns:
        tuple: (clean_df, medians_dict)
    """
    df = raw.drop_duplicates().copy()

    # 2. service_type cleaning
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # 3. payment_mode placeholder
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value text to float in declared_value_inr
    val = df["declared_value"].apply(
        lambda x: float(str(x).replace("₹", "").replace(",", "").strip()) if pd.notna(x) else np.nan
    )
    df["declared_value_inr"] = pd.to_numeric(val, errors="coerce")
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. sentinel weights treated as missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. median imputation
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }

    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return df, medians


def build_features(clean, hubs):
    """Attach hub information and engineer modeling features.
    
    Merges hub attributes (hub_city, region, hub_tier, hub_daily_capacity)
    and constructs promised_hours, is_cod, is_weekend, dispatch_hour,
    and log_declared_value. Does not use delivery_hours.
    
    Returns:
        pd.DataFrame: enriched feature DataFrame
    """
    df = clean.merge(hubs, on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df

