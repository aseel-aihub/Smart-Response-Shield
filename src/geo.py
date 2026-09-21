"""
Geodesic helpers.

All functions use the spherical-earth approximation, which is accurate to
roughly 0.5% over the short distances involved here (< 100 km). That is far
below the uncertainty of the threat kinematics themselves, so a full
ellipsoidal model (e.g. Vincenty) would add complexity without adding
meaningful accuracy.
"""

from __future__ import annotations

import numpy as np

from config import EARTH_RADIUS_KM

__all__ = [
    "haversine_km",
    "initial_bearing_deg",
    "destination_point",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in kilometres between two points.

    Accepts scalars or numpy arrays (broadcasting applies), which lets the
    nearest-facility search evaluate every facility in one vectorised call
    instead of looping.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # clip guards against floating-point values marginally above 1.0
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    return EARTH_RADIUS_KM * c


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """
    Initial compass bearing (0-360, 0 = due north) from point 1 to point 2.
    """
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def destination_point(lat, lon, bearing_deg, distance_km):
    """
    Point reached by travelling `distance_km` from (lat, lon) along a constant
    compass `bearing_deg`.

    This replaces the flat "divide metres by 111 000" approximation used in the
    original prototype, which distorts longitude badly away from the equator
    and does not account for the path curving as it travels.
    """
    ang = np.asarray(distance_km, dtype=float) / EARTH_RADIUS_KM
    brg = np.radians(bearing_deg)
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)

    sin_lat2 = np.sin(lat1) * np.cos(ang) + np.cos(lat1) * np.sin(ang) * np.cos(brg)
    lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))

    lon2 = lon1 + np.arctan2(
        np.sin(brg) * np.sin(ang) * np.cos(lat1),
        np.cos(ang) - np.sin(lat1) * sin_lat2,
    )

    # normalise longitude into [-180, 180)
    lon2_deg = (np.degrees(lon2) + 540.0) % 360.0 - 180.0

    return np.degrees(lat2), lon2_deg
