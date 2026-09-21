"""
Trajectory model for an unpowered ballistic object.

The original prototype used the textbook vacuum range formula
R = v^2 * sin(2*theta) / g. That is a closed-form solution that assumes no
atmosphere, and it overestimates range substantially at the speeds in this
simulation. This module keeps that formula available as a reference baseline
but uses numerical integration with quadratic drag as the working model, which
also yields two quantities the vacuum formula cannot give directly:

  * time of flight  -> drives the time-to-impact used by the response rules
  * the full flight path -> lets the dashboard draw the real arc

Physics used here is standard undergraduate projectile motion. The model is
deliberately generic: it takes a speed, a launch angle and a drag constant,
and it describes where a falling object lands. It is used to answer the
defensive question "what should we evacuate, and how long do we have", which
is the purpose of the whole system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import DEFAULT_DRAG_K, GRAVITY, INTEGRATION_DT

__all__ = [
    "TrajectoryResult",
    "simulate_trajectory",
    "vacuum_range_km",
]


@dataclass
class TrajectoryResult:
    """Outcome of a single trajectory integration."""

    range_km: float
    time_of_flight_s: float
    apogee_km: float
    impact_speed_ms: float
    # Sampled flight path as (downrange_km, altitude_km) pairs, for plotting.
    path: list = field(default_factory=list, repr=False)

    @property
    def range_m(self) -> float:
        return self.range_km * 1000.0


def vacuum_range_km(speed_ms: float, angle_deg: float) -> float:
    """
    Closed-form range with no atmosphere. Kept for comparison in the dashboard
    so the effect of drag is visible, and as a sanity check in the tests.
    """
    theta = np.radians(angle_deg)
    return (speed_ms**2) * np.sin(2.0 * theta) / GRAVITY / 1000.0


def simulate_trajectory(
    speed_ms: float,
    angle_deg: float,
    drag_k: float = DEFAULT_DRAG_K,
    dt: float = INTEGRATION_DT,
    max_time_s: float = 600.0,
) -> TrajectoryResult:
    """
    Integrate 2-D motion under gravity plus quadratic drag.

        dv/dt = -g * j_hat  -  k * |v| * v

    Uses velocity-Verlet-style midpoint stepping (RK2), which is markedly more
    accurate than plain Euler for the same step size and costs one extra
    force evaluation per step.

    Parameters
    ----------
    speed_ms : initial speed, metres/second
    angle_deg : launch elevation above horizontal, degrees
    drag_k : quadratic drag constant, 1/m. Pass 0.0 for the vacuum case.
    dt : integration step, seconds
    max_time_s : safety cap so a pathological input cannot loop forever

    Returns
    -------
    TrajectoryResult
    """
    if speed_ms <= 0:
        raise ValueError("speed_ms must be positive")
    if not 0.0 < angle_deg < 90.0:
        raise ValueError("angle_deg must be strictly between 0 and 90")
    if drag_k < 0:
        raise ValueError("drag_k must be non-negative")

    theta = np.radians(angle_deg)
    pos = np.array([0.0, 0.0])  # (downrange, altitude) in metres
    vel = np.array([speed_ms * np.cos(theta), speed_ms * np.sin(theta)])

    def acceleration(v: np.ndarray) -> np.ndarray:
        a = np.array([0.0, -GRAVITY])
        if drag_k > 0.0:
            a = a - drag_k * np.linalg.norm(v) * v
        return a

    path = [(0.0, 0.0)]
    t = 0.0
    prev_pos = pos.copy()
    prev_vel = vel.copy()

    while t < max_time_s:
        prev_pos, prev_vel = pos.copy(), vel.copy()

        # RK2 (midpoint)
        a1 = acceleration(vel)
        v_mid = vel + 0.5 * dt * a1
        a2 = acceleration(v_mid)

        pos = pos + dt * v_mid
        vel = vel + dt * a2
        t += dt

        if pos[1] <= 0.0:
            break

        path.append((pos[0] / 1000.0, pos[1] / 1000.0))

    # Linear interpolation back to the exact ground-crossing point, so the
    # reported range does not depend on where the last step happened to land.
    if prev_pos[1] != pos[1]:
        frac = prev_pos[1] / (prev_pos[1] - pos[1])
    else:
        frac = 0.0
    impact_pos = prev_pos + frac * (pos - prev_pos)
    impact_vel = prev_vel + frac * (vel - prev_vel)
    impact_time = t - dt + frac * dt

    path.append((impact_pos[0] / 1000.0, 0.0))

    altitudes = [p[1] for p in path]

    return TrajectoryResult(
        range_km=float(impact_pos[0] / 1000.0),
        time_of_flight_s=float(impact_time),
        apogee_km=float(max(altitudes)),
        impact_speed_ms=float(np.linalg.norm(impact_vel)),
        path=path,
    )
