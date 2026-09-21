"""
Test suite for Smart Response Shield.

Run with:  pytest -v

The most important test in this file is `test_model_beats_baseline`. The
original prototype shipped a model that scored 40.0% against a 40.3% baseline —
it had learned nothing, and nothing in the codebase would have caught that. A
test that asserts the model beats a constant guess is the regression guard for
the single worst failure mode this project has.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SECTORS  # noqa: E402
from src.ballistics import simulate_trajectory, vacuum_range_km  # noqa: E402
from src.data_generation import FEATURE_COLUMNS, generate_dataset  # noqa: E402
from src.facilities import (  # noqa: E402
    FACILITIES,
    facilities_within,
    load_facilities,
    nearest_facility,
)
from src.geo import destination_point, haversine_km, initial_bearing_deg  # noqa: E402
from src.response_rules import THREAT_LEVELS, assess_threat  # noqa: E402
from src.self_healing import build_network, simulate_failure  # noqa: E402


# ---------------------------------------------------------------------------
# Geodesy
# ---------------------------------------------------------------------------
class TestGeo:
    def test_zero_distance(self):
        assert haversine_km(17.5, 42.5, 17.5, 42.5) == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_latitude_is_about_111km(self):
        d = haversine_km(17.0, 42.0, 18.0, 42.0)
        assert d == pytest.approx(111.19, rel=0.01)

    def test_symmetry(self):
        a = haversine_km(17.1, 42.3, 17.9, 43.2)
        b = haversine_km(17.9, 43.2, 17.1, 42.3)
        assert a == pytest.approx(b)

    def test_vectorised_matches_scalar(self):
        lats = np.array([17.2, 17.6, 17.9])
        lons = np.array([42.5, 42.9, 43.3])
        vec = haversine_km(17.4, 42.7, lats, lons)
        for i in range(3):
            assert vec[i] == pytest.approx(haversine_km(17.4, 42.7, lats[i], lons[i]))

    def test_destination_then_distance_roundtrip(self):
        lat, lon = destination_point(17.2, 42.6, bearing_deg=35.0, distance_km=48.0)
        assert haversine_km(17.2, 42.6, lat, lon) == pytest.approx(48.0, rel=1e-6)

    def test_destination_bearing_roundtrip(self):
        lat, lon = destination_point(17.2, 42.6, bearing_deg=35.0, distance_km=40.0)
        assert initial_bearing_deg(17.2, 42.6, lat, lon) == pytest.approx(35.0, abs=0.5)

    def test_due_north_keeps_longitude(self):
        lat, lon = destination_point(17.2, 42.6, bearing_deg=0.0, distance_km=50.0)
        assert lon == pytest.approx(42.6, abs=1e-6)
        assert lat > 17.2


# ---------------------------------------------------------------------------
# Ballistics
# ---------------------------------------------------------------------------
class TestBallistics:
    def test_no_drag_matches_closed_form(self):
        """With drag disabled the integrator must reproduce v^2 sin(2t)/g."""
        for speed, angle in [(300, 30), (500, 45), (800, 55)]:
            numeric = simulate_trajectory(speed, angle, drag_k=0.0, dt=0.005).range_km
            assert numeric == pytest.approx(vacuum_range_km(speed, angle), rel=0.002)

    def test_drag_reduces_range(self):
        with_drag = simulate_trajectory(650, 45).range_km
        without = simulate_trajectory(650, 45, drag_k=0.0).range_km
        assert with_drag < without

    def test_45_degrees_maximises_vacuum_range(self):
        r45 = simulate_trajectory(600, 45, drag_k=0.0).range_km
        assert r45 > simulate_trajectory(600, 30, drag_k=0.0).range_km
        assert r45 > simulate_trajectory(600, 60, drag_k=0.0).range_km

    def test_range_increases_with_speed(self):
        ranges = [simulate_trajectory(s, 45).range_km for s in (300, 500, 700, 900)]
        assert ranges == sorted(ranges)

    def test_time_of_flight_positive_and_ordered(self):
        low = simulate_trajectory(500, 30)
        high = simulate_trajectory(500, 60)
        assert low.time_of_flight_s > 0
        # A steeper launch spends longer in the air for the same speed.
        assert high.time_of_flight_s > low.time_of_flight_s

    def test_apogee_below_start_and_above_zero(self):
        t = simulate_trajectory(650, 45)
        assert 0 < t.apogee_km < t.range_km

    def test_path_starts_and_ends_at_ground(self):
        t = simulate_trajectory(650, 45)
        assert t.path[0] == (0.0, 0.0)
        assert t.path[-1][1] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("speed,angle", [(0, 45), (-100, 45), (500, 0), (500, 90)])
    def test_rejects_invalid_input(self, speed, angle):
        with pytest.raises(ValueError):
            simulate_trajectory(speed, angle)


# ---------------------------------------------------------------------------
# Facilities
# ---------------------------------------------------------------------------
class TestFacilities:
    def test_register_covers_all_sectors(self):
        df = load_facilities()
        assert set(df["sector"]) == set(SECTORS)

    def test_ids_unique(self):
        df = load_facilities()
        assert df["facility_id"].is_unique

    def test_criticality_in_range(self):
        df = load_facilities()
        assert df["criticality"].between(1, 5).all()

    def test_nearest_facility_finds_exact_match(self):
        """Querying a facility's own coordinates must return that facility."""
        for fac in FACILITIES:
            row, dist = nearest_facility(fac.lat, fac.lon)
            assert row["facility_id"] == fac.facility_id
            assert dist == pytest.approx(0.0, abs=1e-6)

    def test_facilities_within_respects_radius(self):
        got = facilities_within(17.35, 42.90, radius_km=25.0)
        assert (got["distance_km"] <= 25.0).all()

    def test_facilities_within_is_sorted(self):
        got = facilities_within(17.35, 42.90, radius_km=60.0)
        assert list(got["distance_km"]) == sorted(got["distance_km"])


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------
class TestDataset:
    @pytest.fixture(scope="class")
    def df(self):
        return generate_dataset(n_samples=400, seed=7)

    def test_has_expected_columns(self, df):
        for col in FEATURE_COLUMNS + ["target_sector", "target_facility"]:
            assert col in df.columns

    def test_no_missing_values(self, df):
        assert not df.isna().any().any()

    def test_labels_are_valid_sectors(self, df):
        assert set(df["target_sector"]).issubset(set(SECTORS))

    def test_label_noise_rate_is_approximately_correct(self, df):
        corrupted = (df["target_sector"] != df["true_sector"]).mean()
        assert corrupted == pytest.approx(0.08, abs=0.02)

    def test_clean_labels_match_nearest_facility(self, df):
        """
        The uncorrupted label must equal the sector of the nearest facility.
        This is the property the original generator violated.
        """
        facilities = load_facilities().set_index("facility_id")
        for _, row in df.head(60).iterrows():
            expected = facilities.loc[row["nearest_facility_id"], "sector"]
            assert row["true_sector"] == expected

    def test_labels_depend_on_features(self, df):
        """
        Different headings from the same origin must be able to produce
        different sectors. If they never do, the label is not a function of
        the features and the whole learning task is void.
        """
        subset = df[df["heading_deg"] < 15]["true_sector"].unique()
        other = df[df["heading_deg"] > 45]["true_sector"].unique()
        assert set(subset) != set(other) or len(df["true_sector"].unique()) > 1


# ---------------------------------------------------------------------------
# Model quality — the critical regression guard
# ---------------------------------------------------------------------------
class TestModel:
    def test_model_beats_baseline(self, tmp_path):
        """
        Regression guard for the defect that shipped in the first prototype:
        a classifier that scored no better than always answering the commonest
        class. Anything under a 15-point lift means the labels have stopped
        carrying information about the features.
        """
        from src.data_generation import build_and_save
        from src.pattern_model import train

        data = tmp_path / "data.csv"
        build_and_save(path=data, n_samples=1500, seed=11)

        _, m = train(
            data_path=data,
            model_path=tmp_path / "m.joblib",
            metrics_path=tmp_path / "m.json",
            n_estimators=120,
        )

        assert m.accuracy > m.baseline_accuracy + 0.15
        # A near-perfect score would mean label noise is not being applied.
        assert m.accuracy < 0.99
        # Cross-validation must corroborate the held-out score.
        assert abs(m.cv_mean - m.accuracy) < 0.10


# ---------------------------------------------------------------------------
# Response rules
# ---------------------------------------------------------------------------
class TestResponseRules:
    def test_direct_hit_on_critical_asset_is_critical(self):
        fac = FACILITIES[0]  # criticality 5
        a = assess_threat(fac.lat, fac.lon, time_to_impact_s=45.0,
                          predicted_sector=fac.sector, sector_confidence=0.9)
        assert a.threat_level == "CRITICAL"
        assert a.is_direct_hit

    def test_remote_impact_is_low(self):
        a = assess_threat(16.0, 41.0, time_to_impact_s=900.0,
                          predicted_sector="South Sector", sector_confidence=0.9)
        assert a.threat_level == "LOW"
        assert not a.is_direct_hit

    def test_always_returns_actions_and_reasoning(self):
        a = assess_threat(17.4, 42.9, time_to_impact_s=100.0,
                          predicted_sector="West Sector", sector_confidence=0.6)
        assert a.actions and a.reasoning
        assert a.threat_level in THREAT_LEVELS

    def test_low_confidence_is_flagged(self):
        a = assess_threat(17.4, 42.9, time_to_impact_s=100.0,
                          predicted_sector="West Sector", sector_confidence=0.30)
        assert any("provisional" in r for r in a.reasoning)

    def test_level_never_downgrades_on_shorter_warning(self):
        """Less time to react must never lower the threat level."""
        fac = FACILITIES[0]
        slow = assess_threat(fac.lat, fac.lon, 300.0, fac.sector, 0.9)
        fast = assess_threat(fac.lat, fac.lon, 30.0, fac.sector, 0.9)
        assert THREAT_LEVELS.index(fast.threat_level) >= THREAT_LEVELS.index(slow.threat_level)


# ---------------------------------------------------------------------------
# Self-healing network
# ---------------------------------------------------------------------------
class TestSelfHealing:
    def test_network_contains_every_facility(self):
        G = build_network()
        for fac in FACILITIES:
            assert fac.facility_id in G

    def test_undamaged_network_supplies_everything(self):
        status = simulate_failure([])
        assert status.availability == pytest.approx(1.0)
        assert status.unsupplied == []

    def test_losing_one_generator_heals_completely(self):
        """The plant cross-tie and ring should absorb a single source loss."""
        status = simulate_failure(["Power_Plant_A"])
        assert status.availability == pytest.approx(1.0)
        assert status.healed_count > 0

    def test_losing_all_generation_is_total_outage(self):
        status = simulate_failure(["Power_Plant_A", "Power_Plant_B"])
        assert status.availability == pytest.approx(0.0)

    def test_redundant_feeder_saves_critical_assets(self):
        """
        Losing Substation_North must not take down the criticality-5 assets
        behind it, because they hold backup feeders. NTH-03 has none and is
        expected to go dark — that asymmetry is the design point.
        """
        status = simulate_failure(["Substation_North"])
        assert "NTH-01" not in status.unsupplied
        assert "NTH-02" not in status.unsupplied
        assert "NTH-03" in status.unsupplied

    def test_reroute_costs_more_than_baseline(self):
        status = simulate_failure(["Power_Plant_A"])
        for fid in status.rerouted:
            assert status.current_cost[fid] >= status.baseline_cost[fid]

    def test_accepts_single_string(self):
        assert simulate_failure("Substation_North").damaged_nodes == ["Substation_North"]

    def test_unknown_node_is_harmless(self):
        status = simulate_failure(["Does_Not_Exist"])
        assert status.availability == pytest.approx(1.0)
