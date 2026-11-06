"""Part D — Hub Segments and Profiling."""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")


def run_part_d_segments():
    os.makedirs("outputs", exist_ok=True)

    modelling_df = pd.read_csv("data/modelling_table.csv")
    hubs_df = pd.read_csv("data/hubs.csv")

    # 1. Hub profile
    hub_profile = modelling_df.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).reset_index().sort_values("hub_id")

    # Round profile metrics to 4 decimals
    hub_profile_out = hub_profile.copy()
    hub_profile_out["mean_delivery_hours"] = hub_profile_out["mean_delivery_hours"].round(4)
    hub_profile_out["delay_rate"] = hub_profile_out["delay_rate"].round(4)
    hub_profile_out["mean_distance_km"] = hub_profile_out["mean_distance_km"].round(4)
    hub_profile_out["mean_n_stops"] = hub_profile_out["mean_n_stops"].round(4)
    hub_profile_out.to_csv("outputs/hub_profile.csv", index=False)

    # 2 & 3. Scaling and K-Means segmentation (order fixed: ascending hub_id)
    profile_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(hub_profile[profile_cols])

    dist_matrix = squareform(pdist(scaled_matrix, metric="euclidean"))
    n_hubs = len(hub_profile)

    separation_rows = []
    kmeans_models = {}

    for K in [2, 3, 4, 5]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        kmeans_models[K] = (km, labels)

        # Cluster sizes descending joined by hyphens
        counts = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
        cluster_sizes_str = "-".join(map(str, counts))

        # Compute across-to-within distance ratio
        within_dists = []
        across_dists = []
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = dist_matrix[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        ratio = np.mean(across_dists) / np.mean(within_dists)
        separation_rows.append({
            "K": K,
            "across_over_within": round(float(ratio), 4),
            "cluster_sizes": cluster_sizes_str,
        })

    pd.DataFrame(separation_rows).to_csv("outputs/part_d_separation.csv", index=False)

    # 4. K = 3 Segment naming and tier check
    _, k3_labels = kmeans_models[3]
    hub_profile["cluster"] = k3_labels

    cluster_mean_delay = hub_profile.groupby("cluster")["delay_rate"].mean().sort_values()
    # Lowest -> Standard, Middle -> Watch, Highest -> Escalate
    cluster_to_segment = {
        cluster_mean_delay.index[0]: "Standard",
        cluster_mean_delay.index[1]: "Watch",
        cluster_mean_delay.index[2]: "Escalate",
    }
    hub_profile["segment"] = hub_profile["cluster"].map(cluster_to_segment)

    hub_segments_df = hub_profile.merge(
        hubs_df[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left"
    )[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_segments_df.to_csv("outputs/hub_segments.csv", index=False)

    # Segment profile: means of hub-level values
    hub_merged_prof = hub_profile.merge(
        hubs_df[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left"
    )
    seg_profile_rows = []
    for seg_name in ["Standard", "Watch", "Escalate"]:
        sub = hub_merged_prof[hub_merged_prof["segment"] == seg_name]
        seg_profile_rows.append({
            "segment": seg_name,
            "hubs": len(sub),
            "mean_delivery_hours": round(float(sub["mean_delivery_hours"].mean()), 4),
            "delay_rate": round(float(sub["delay_rate"].mean()), 4),
            "mean_distance_km": round(float(sub["mean_distance_km"].mean()), 4),
            "mean_n_stops": round(float(sub["mean_n_stops"].mean()), 4),
        })
    pd.DataFrame(seg_profile_rows).to_csv("outputs/segment_profile.csv", index=False)

    # Tier vs Segment Mismatch
    tier_implied_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_segments_df["expected_segment"] = hub_segments_df["hub_tier"].map(tier_implied_segment)
    mismatches = hub_segments_df[hub_segments_df["segment"] != hub_segments_df["expected_segment"]][
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ]
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print("Part D segmentation complete. Written hub_profile.csv, part_d_separation.csv, hub_segments.csv, segment_profile.csv, tier_vs_segment_mismatch.csv")


if __name__ == "__main__":
    run_part_d_segments()

