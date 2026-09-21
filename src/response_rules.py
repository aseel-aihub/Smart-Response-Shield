"""
Protective-action recommendation engine (Objective 4).

This was absent from the first prototype: the system predicted an impact point
but never said what to do about it, which is the part that actually saves
lives.

WHY RULES AND NOT A SECOND ML MODEL
-----------------------------------
A deliberate architectural choice. Prediction is statistical, so a model is
appropriate. Deciding whether to order an evacuation is a consequential,
auditable decision that a reviewing officer must be able to interrogate after
the fact: "why did it say evacuate?" must have an answer in plain language.
Every assessment below therefore carries the exact rules that fired, and the
thresholds are declared in config.py rather than buried in learned weights.

Every recommendation is advisory output for a simulation. Nothing here
actuates anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import (
    FACILITY_DANGER_RADIUS_KM,
    FACILITY_WATCH_RADIUS_KM,
    TTI_CRITICAL_S,
    TTI_HIGH_S,
    TTI_MODERATE_S,
)
from src.facilities import facilities_within, nearest_facility

__all__ = ["ThreatAssessment", "assess_threat", "THREAT_LEVELS"]

THREAT_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

_LEVEL_RANK = {name: i for i, name in enumerate(THREAT_LEVELS)}


@dataclass
class ThreatAssessment:
    threat_level: str
    confidence: float
    time_to_impact_s: float
    nearest_facility_name: str
    nearest_facility_id: str
    nearest_facility_km: float
    nearest_facility_criticality: int
    predicted_sector: str
    facilities_at_risk: pd.DataFrame = field(repr=False)
    actions: list = field(default_factory=list)
    reasoning: list = field(default_factory=list)

    @property
    def is_direct_hit(self) -> bool:
        return self.nearest_facility_km <= FACILITY_DANGER_RADIUS_KM


def _escalate(current: str, candidate: str) -> str:
    """Threat level only ever moves up, never down."""
    return candidate if _LEVEL_RANK[candidate] > _LEVEL_RANK[current] else current


def assess_threat(
    impact_lat: float,
    impact_lon: float,
    time_to_impact_s: float,
    predicted_sector: str,
    sector_confidence: float,
) -> ThreatAssessment:
    """
    Turn a predicted impact into a graded threat level and an ordered,
    explainable list of protective actions.
    """
    facility, distance_km = nearest_facility(impact_lat, impact_lon)
    at_risk = facilities_within(impact_lat, impact_lon, FACILITY_WATCH_RADIUS_KM)

    level = "LOW"
    reasoning: list[str] = []
    actions: list[str] = []

    # ---- Rule 1: proximity to a protected asset ----------------------------
    if distance_km <= FACILITY_DANGER_RADIUS_KM:
        level = _escalate(level, "HIGH")
        reasoning.append(
            f"Predicted impact is {distance_km:.1f} km from {facility['name']}, "
            f"inside the {FACILITY_DANGER_RADIUS_KM} km danger radius."
        )
    elif distance_km <= FACILITY_WATCH_RADIUS_KM:
        level = _escalate(level, "MODERATE")
        reasoning.append(
            f"Predicted impact is {distance_km:.1f} km from {facility['name']}, "
            f"inside the {FACILITY_WATCH_RADIUS_KM} km watch radius."
        )
    else:
        reasoning.append(
            f"Nearest protected asset ({facility['name']}) is {distance_km:.1f} km "
            "away — outside the watch radius."
        )

    # ---- Rule 2: consequence if the asset is lost --------------------------
    criticality = int(facility["criticality"])
    if criticality >= 5 and distance_km <= FACILITY_WATCH_RADIUS_KM:
        level = _escalate(level, "CRITICAL")
        reasoning.append(
            f"{facility['name']} is criticality {criticality}/5 "
            f"({facility['facility_type']}); loss would interrupt an essential service."
        )
    elif criticality >= 4 and distance_km <= FACILITY_DANGER_RADIUS_KM:
        level = _escalate(level, "HIGH")
        reasoning.append(
            f"{facility['name']} is criticality {criticality}/5 and lies in the danger radius."
        )

    # ---- Rule 3: how much time is left -------------------------------------
    if time_to_impact_s <= TTI_CRITICAL_S:
        if distance_km <= FACILITY_WATCH_RADIUS_KM:
            level = _escalate(level, "CRITICAL")
        reasoning.append(
            f"Only {time_to_impact_s:.0f} s to impact — below the "
            f"{TTI_CRITICAL_S:.0f} s immediate-action threshold."
        )
    elif time_to_impact_s <= TTI_HIGH_S:
        if distance_km <= FACILITY_DANGER_RADIUS_KM:
            level = _escalate(level, "CRITICAL")
        reasoning.append(
            f"{time_to_impact_s:.0f} s to impact — limited warning time."
        )
    elif time_to_impact_s <= TTI_MODERATE_S:
        reasoning.append(
            f"{time_to_impact_s:.0f} s to impact — sufficient time for a staged response."
        )
    else:
        reasoning.append(
            f"{time_to_impact_s:.0f} s to impact — ample warning time."
        )

    # ---- Rule 4: how much to trust the sector prediction -------------------
    if sector_confidence < 0.50:
        reasoning.append(
            f"Sector prediction confidence is only {sector_confidence*100:.0f}% — "
            "treat the sector attribution as provisional and widen the alert."
        )
    elif sector_confidence >= 0.80:
        reasoning.append(
            f"Sector prediction confidence is {sector_confidence*100:.0f}% — "
            f"{predicted_sector} attribution is well supported."
        )

    # ---- Action set, ordered by urgency ------------------------------------
    if level == "CRITICAL":
        actions = [
            f"Issue immediate impact-area warning for {facility['name']} ({facility['facility_id']}).",
            "Initiate personnel evacuation to hardened shelters in the predicted footprint.",
            f"Alert {predicted_sector} response command and place crews on standby.",
            "Pre-stage the self-healing reroute so supply transfers the moment damage is confirmed.",
            "Escalate to the national operations centre.",
        ]
    elif level == "HIGH":
        actions = [
            f"Issue a warning for {facility['name']} ({facility['facility_id']}).",
            "Move non-essential personnel clear of the predicted footprint.",
            f"Place {predicted_sector} response crews on alert.",
            "Verify alternate supply routing is available for the affected node.",
        ]
    elif level == "MODERATE":
        actions = [
            f"Notify the {facility['name']} duty officer and hold at readiness.",
            f"Increase monitoring across {predicted_sector}.",
            "Confirm backup supply paths are healthy.",
        ]
    else:
        actions = [
            "Log the track and continue routine monitoring.",
            "No protective action required at this time.",
        ]

    if len(at_risk) > 1:
        actions.insert(
            0,
            f"{len(at_risk)} facilities lie within {FACILITY_WATCH_RADIUS_KM} km "
            "of the predicted impact — widen the alert to all of them.",
        )

    return ThreatAssessment(
        threat_level=level,
        confidence=sector_confidence,
        time_to_impact_s=time_to_impact_s,
        nearest_facility_name=str(facility["name"]),
        nearest_facility_id=str(facility["facility_id"]),
        nearest_facility_km=distance_km,
        nearest_facility_criticality=criticality,
        predicted_sector=predicted_sector,
        facilities_at_risk=at_risk,
        actions=actions,
        reasoning=reasoning,
    )
