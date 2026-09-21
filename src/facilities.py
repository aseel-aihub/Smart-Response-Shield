"""
Registry of protected facilities, plus the nearest-facility matching that
turns a predicted impact coordinate into "which asset is actually at risk".

This module implements Objective 2 of the proposal, which had no
implementation at all in the first prototype.

ALL FACILITIES BELOW ARE FICTIONAL. They are invented assets placed inside
the notional coordinate frame defined in config.py so that the simulation has
something to protect. They deliberately do not correspond to any real
installation, and they carry generic designators rather than real names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from config import FACILITIES_CSV
from src.geo import haversine_km

__all__ = [
    "Facility",
    "FACILITIES",
    "facilities_dataframe",
    "write_facilities_csv",
    "load_facilities",
    "nearest_facility",
    "facilities_within",
]


@dataclass(frozen=True)
class Facility:
    facility_id: str
    name: str
    sector: str
    facility_type: str
    # 1 = lowest consequence if lost, 5 = highest
    criticality: int
    lat: float
    lon: float
    # Node this facility draws supply from, used by the self-healing network
    supply_node: str
    # Redundant feeder (N-1 design). Only high-criticality assets get one;
    # that asymmetry is deliberate, so the simulation can show which assets
    # survive a substation loss and which do not.
    backup_node: str = ""


# --------------------------------------------------------------------------
# The notional asset register (12 facilities, 3 per sector)
# --------------------------------------------------------------------------
FACILITIES: list[Facility] = [
    # ---- North Sector ----
    Facility("NTH-01", "Power Plant North Alpha", "North Sector", "Power Generation", 5, 17.58, 42.78, "Substation_North", "Substation_West"),
    Facility("NTH-02", "Desalination Plant North", "North Sector", "Water Supply", 5, 17.64, 43.02, "Substation_North", "Substation_East"),
    Facility("NTH-03", "Comms Relay North", "North Sector", "Communications", 3, 17.53, 43.18, "Substation_North", ""),
    # ---- South Sector ----
    Facility("STH-01", "Power Plant South Bravo", "South Sector", "Power Generation", 5, 17.12, 42.62, "Substation_South", "Substation_West"),
    Facility("STH-02", "Logistics Depot South", "South Sector", "Logistics", 3, 17.16, 42.95, "Substation_South", ""),
    Facility("STH-03", "Comms Relay South", "South Sector", "Communications", 3, 17.08, 43.08, "Substation_South", ""),
    # ---- East Sector ----
    Facility("EST-01", "Substation East", "East Sector", "Power Distribution", 4, 17.30, 43.16, "Substation_East", "Substation_North"),
    Facility("EST-02", "Water Treatment East", "East Sector", "Water Supply", 4, 17.42, 43.30, "Substation_East", "Substation_South"),
    Facility("EST-03", "Forward Operating Base East", "East Sector", "Military", 5, 17.24, 43.33, "Substation_East", "Substation_South"),
    # ---- West Sector ----
    Facility("WST-01", "Substation West", "West Sector", "Power Distribution", 4, 17.33, 42.55, "Substation_West", "Substation_South"),
    Facility("WST-02", "Fuel Storage West", "West Sector", "Fuel Storage", 4, 17.24, 42.45, "Substation_West", "Substation_North"),
    Facility("WST-03", "Radar Station West", "West Sector", "Sensor Site", 4, 17.46, 42.62, "Substation_West", "Substation_North"),
]


def facilities_dataframe() -> pd.DataFrame:
    """The in-code register as a DataFrame."""
    return pd.DataFrame([asdict(f) for f in FACILITIES])


def write_facilities_csv(path=FACILITIES_CSV) -> pd.DataFrame:
    """Persist the register so the dataset generator and app share one source."""
    df = facilities_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_facilities(path=FACILITIES_CSV) -> pd.DataFrame:
    """
    Load the register from CSV, falling back to the in-code definition if the
    file has not been generated yet. The fallback means the dashboard still
    runs on a fresh clone before any script has been executed.
    """
    if path.exists():
        return pd.read_csv(path)
    return facilities_dataframe()


def nearest_facility(lat: float, lon: float, facilities: pd.DataFrame | None = None):
    """
    Return (facility_row, distance_km) for the closest facility to a point.

    Distances to every facility are computed in a single vectorised haversine
    call rather than a Python loop, so this stays fast enough to run on every
    dashboard interaction.
    """
    df = load_facilities() if facilities is None else facilities

    distances = haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
    idx = int(distances.argmin())

    return df.iloc[idx], float(distances[idx])


def facilities_within(lat: float, lon: float, radius_km: float,
                      facilities: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    All facilities inside `radius_km` of a point, nearest first, with a
    `distance_km` column added. Used to show everything in the blast footprint
    rather than only the single closest asset.
    """
    df = (load_facilities() if facilities is None else facilities).copy()

    df["distance_km"] = haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())

    return df[df["distance_km"] <= radius_km].sort_values("distance_km").reset_index(drop=True)
