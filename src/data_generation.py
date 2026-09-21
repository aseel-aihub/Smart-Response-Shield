"""
Synthetic dataset generation.

WHY THIS MODULE WAS REWRITTEN
-----------------------------
The first prototype produced its labels like this:

    target_sector = np.random.choice(sectors, num_samples, p=[.4,.3,.15,.15])

drawn independently of launch_lat, launch_lon, speed, angle and heading. The
label therefore carried no information about the features, so no model could
ever learn the mapping. Measured on the original data, the Random Forest
scored 40.0% while simply always answering "North Sector" scored 40.3% — the
classifier was doing slightly worse than a constant guess, which is the
signature of fitting pure noise.

The generator below instead derives every label from the physics:

    launch params -> trajectory integration -> impact coordinate
                  -> nearest facility -> that facility's sector

so the label is a genuine (non-linear, non-obvious) function of the inputs.
A deliberate fraction of labels is then corrupted (LABEL_NOISE_RATE) to
represent tracking error and evasive manoeuvring, which keeps the problem
realistic and stops the reported accuracy from being a meaningless 100%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ANGLE_MAX_DEG,
    ANGLE_MIN_DEG,
    DATASET_SIZE,
    HEADING_MAX_DEG,
    HEADING_MIN_DEG,
    LABEL_NOISE_RATE,
    LAUNCH_LAT_MAX,
    LAUNCH_LAT_MIN,
    LAUNCH_LON_MAX,
    LAUNCH_LON_MIN,
    RANDOM_SEED,
    SECTORS,
    SPEED_MAX_MS,
    SPEED_MIN_MS,
    THREAT_DATA_CSV,
)
from src.ballistics import simulate_trajectory
from src.facilities import load_facilities
from src.geo import destination_point, haversine_km

__all__ = ["FEATURE_COLUMNS", "generate_dataset", "build_and_save"]

FEATURE_COLUMNS = [
    "launch_lat",
    "launch_lon",
    "speed_m_s",
    "launch_angle_deg",
    "heading_deg",
]


def generate_dataset(n_samples: int = DATASET_SIZE, seed: int = RANDOM_SEED,
                     label_noise: float = LABEL_NOISE_RATE) -> pd.DataFrame:
    """
    Build a labelled dataset where the label is physically caused by the
    features, then add controlled label noise.
    """
    rng = np.random.default_rng(seed)
    facilities = load_facilities()

    fac_lat = facilities["lat"].to_numpy()
    fac_lon = facilities["lon"].to_numpy()

    launch_lat = rng.uniform(LAUNCH_LAT_MIN, LAUNCH_LAT_MAX, n_samples)
    launch_lon = rng.uniform(LAUNCH_LON_MIN, LAUNCH_LON_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN_MS, SPEED_MAX_MS, n_samples)
    angle = rng.uniform(ANGLE_MIN_DEG, ANGLE_MAX_DEG, n_samples)
    heading = rng.uniform(HEADING_MIN_DEG, HEADING_MAX_DEG, n_samples)

    rows = []
    for i in range(n_samples):
        traj = simulate_trajectory(float(speed[i]), float(angle[i]))
        imp_lat, imp_lon = destination_point(
            launch_lat[i], launch_lon[i], heading[i], traj.range_km
        )

        distances = haversine_km(imp_lat, imp_lon, fac_lat, fac_lon)
        nearest = int(distances.argmin())
        fac = facilities.iloc[nearest]

        rows.append(
            {
                "launch_lat": launch_lat[i],
                "launch_lon": launch_lon[i],
                "speed_m_s": speed[i],
                "launch_angle_deg": angle[i],
                "heading_deg": heading[i],
                # --- derived, kept for analysis and for the dashboard ---
                "range_km": traj.range_km,
                "time_of_flight_s": traj.time_of_flight_s,
                "impact_lat": float(imp_lat),
                "impact_lon": float(imp_lon),
                "nearest_facility_id": fac["facility_id"],
                "nearest_facility_km": float(distances[nearest]),
                # --- the supervised targets ---
                "target_facility": fac["name"],
                "target_sector": fac["sector"],
            }
        )

    df = pd.DataFrame(rows)

    # Keep an uncorrupted copy so the noise rate can be verified in tests.
    df["true_sector"] = df["target_sector"]

    if label_noise > 0:
        n_flip = int(round(label_noise * len(df)))
        flip_idx = rng.choice(len(df), size=n_flip, replace=False)
        for idx in flip_idx:
            alternatives = [s for s in SECTORS if s != df.at[idx, "target_sector"]]
            df.at[idx, "target_sector"] = rng.choice(alternatives)

    return df


def build_and_save(path=THREAT_DATA_CSV, **kwargs) -> pd.DataFrame:
    """Generate the dataset and write it to disk."""
    df = generate_dataset(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
