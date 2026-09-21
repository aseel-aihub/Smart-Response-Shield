"""
Central configuration for the Smart Response Shield simulation.

IMPORTANT — ACADEMIC SCOPE NOTE
-------------------------------
Every coordinate, facility and parameter in this project is SYNTHETIC and
NOTIONAL. The "Area of Operations" below is an arbitrary rectangle used only
so the simulation has a consistent coordinate frame. No real installation,
no real sensor feed and no classified data of any kind is used anywhere in
this codebase. The system is a decision-support *simulation* built to
demonstrate a software architecture, not an operational tool.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

FACILITIES_CSV = DATA_DIR / "facilities.csv"
THREAT_DATA_CSV = DATA_DIR / "synthetic_threat_data.csv"
MODEL_PATH = MODELS_DIR / "sector_classifier.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

# --------------------------------------------------------------------------
# Notional Area of Operations (AO) — synthetic coordinate frame
# --------------------------------------------------------------------------
AO_LAT_MIN, AO_LAT_MAX = 17.05, 17.75
AO_LON_MIN, AO_LON_MAX = 42.40, 43.40
AO_CENTER = (17.35, 42.90)

# Band from which simulated threats originate (south of the AO).
LAUNCH_LAT_MIN, LAUNCH_LAT_MAX = 16.95, 17.25
LAUNCH_LON_MIN, LAUNCH_LON_MAX = 42.40, 43.00

# --------------------------------------------------------------------------
# Threat kinematics envelope
# --------------------------------------------------------------------------
SPEED_MIN_MS, SPEED_MAX_MS = 300.0, 900.0
ANGLE_MIN_DEG, ANGLE_MAX_DEG = 30.0, 60.0
HEADING_MIN_DEG, HEADING_MAX_DEG = 0.0, 60.0  # compass bearing, 0 = due north

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
GRAVITY = 9.80665  # m/s^2
# Quadratic drag constant k in: a_drag = -k * |v| * v   (units 1/m).
# Tuned so ranges stay inside the notional AO for the speed envelope above.
DEFAULT_DRAG_K = 8.0e-6
INTEGRATION_DT = 0.05  # seconds

EARTH_RADIUS_KM = 6371.0088

# --------------------------------------------------------------------------
# Response rule thresholds
# --------------------------------------------------------------------------
# Time-to-impact bands (seconds)
TTI_CRITICAL_S = 60.0
TTI_HIGH_S = 120.0
TTI_MODERATE_S = 240.0

# How close an impact must be to a facility to count as "endangering" it (km)
FACILITY_DANGER_RADIUS_KM = 3.0
FACILITY_WATCH_RADIUS_KM = 8.0

# --------------------------------------------------------------------------
# Synthetic dataset generation
# --------------------------------------------------------------------------
DATASET_SIZE = 3000
RANDOM_SEED = 42
# Fraction of rows whose sector label is deliberately corrupted, representing
# sensor error / evasive manoeuvring. Keeps the task non-trivial and stops the
# classifier from reaching an unrealistic 100%.
LABEL_NOISE_RATE = 0.08

SECTORS = ["North Sector", "South Sector", "East Sector", "West Sector"]
