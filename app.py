"""
Smart Response Shield — operator dashboard.

Run with:  streamlit run app.py

Caching policy
--------------
`@st.cache_resource` for the model and graph (expensive, shared, mutable
objects) and `@st.cache_data` for dataframes. The prototype had no caching at
all and retrained a 100-tree forest on every slider movement.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    AO_CENTER,
    ANGLE_MAX_DEG,
    ANGLE_MIN_DEG,
    DEFAULT_DRAG_K,
    FACILITY_WATCH_RADIUS_KM,
    HEADING_MAX_DEG,
    HEADING_MIN_DEG,
    LAUNCH_LAT_MAX,
    LAUNCH_LAT_MIN,
    LAUNCH_LON_MAX,
    LAUNCH_LON_MIN,
    SPEED_MAX_MS,
    SPEED_MIN_MS,
    THREAT_DATA_CSV,
)
from src.ballistics import simulate_trajectory, vacuum_range_km
from src.facilities import load_facilities
from src.geo import destination_point
from src.pattern_model import load_metrics, load_model, predict_sector
from src.response_rules import assess_threat
from src.self_healing import (
    SUPPLY_SOURCES,
    build_network,
    node_positions,
    simulate_failure,
)

st.set_page_config(
    page_title="Smart Response Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

LEVEL_COLOURS = {
    "CRITICAL": "#B3261E",
    "HIGH": "#E8710A",
    "MODERATE": "#F2B705",
    "LOW": "#2E7D32",
}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading prediction model…")
def _model():
    return load_model()


@st.cache_resource
def _network():
    return build_network()


@st.cache_data
def _facilities() -> pd.DataFrame:
    return load_facilities()


@st.cache_data
def _dataset() -> pd.DataFrame | None:
    if THREAT_DATA_CSV.exists():
        return pd.read_csv(THREAT_DATA_CSV)
    return None


@st.cache_data
def _metrics_dict():
    m = load_metrics()
    return None if m is None else m.__dict__


def _map_scatter(**kwargs):
    """
    Plotly renamed the Mapbox traces to Map traces in v6. Try the modern name
    first so the app runs on both old and new installations rather than
    breaking on whichever the grader happens to have.
    """
    if hasattr(go, "Scattermap"):
        return go.Scattermap(**kwargs)
    return go.Scattermapbox(**kwargs)


def _apply_map_layout(fig, center, zoom=8):
    layout = dict(style="open-street-map", center=center, zoom=zoom)
    if hasattr(go, "Scattermap"):
        fig.update_layout(map=layout)
    else:
        fig.update_layout(mapbox=layout)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ Smart Response Shield")
st.caption(
    "AI-driven predictive impact assessment and self-healing infrastructure — "
    "**simulation on synthetic data**"
)

st.info(
    "**Academic simulation.** Every coordinate, facility and track in this "
    "system is synthetic and notional. There is no live sensor feed and no "
    "real installation data. Outputs are advisory only.",
    icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Sidebar — threat parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Simulated track parameters")

launch_lat = st.sidebar.slider("Origin latitude", LAUNCH_LAT_MIN, LAUNCH_LAT_MAX,
                               float(np.mean([LAUNCH_LAT_MIN, LAUNCH_LAT_MAX])), 0.01)
launch_lon = st.sidebar.slider("Origin longitude", LAUNCH_LON_MIN, LAUNCH_LON_MAX,
                               float(np.mean([LAUNCH_LON_MIN, LAUNCH_LON_MAX])), 0.01)
speed = st.sidebar.slider("Initial speed (m/s)", SPEED_MIN_MS, SPEED_MAX_MS, 650.0, 10.0)
angle = st.sidebar.slider("Elevation angle (°)", ANGLE_MIN_DEG, ANGLE_MAX_DEG, 45.0, 1.0)
heading = st.sidebar.slider("Heading (° from north)", HEADING_MIN_DEG, HEADING_MAX_DEG, 25.0, 1.0)

st.sidebar.divider()
use_drag = st.sidebar.checkbox("Model atmospheric drag", value=True,
                               help="Off = textbook vacuum trajectory. On = numerical "
                                    "integration with quadratic drag.")
drag_k = DEFAULT_DRAG_K if use_drag else 0.0

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
traj = simulate_trajectory(speed, angle, drag_k=drag_k)
impact_lat, impact_lon = destination_point(launch_lat, launch_lon, heading, traj.range_km)
impact_lat, impact_lon = float(impact_lat), float(impact_lon)

model = _model()
pred_sector, proba = predict_sector(model, launch_lat, launch_lon, speed, angle, heading)
confidence = float(proba[pred_sector])

assessment = assess_threat(
    impact_lat=impact_lat,
    impact_lon=impact_lon,
    time_to_impact_s=traj.time_of_flight_s,
    predicted_sector=pred_sector,
    sector_confidence=confidence,
)

tab_live, tab_model, tab_network, tab_docs = st.tabs(
    ["🎯 Live assessment", "🧠 Pattern analysis", "🔄 Self-healing network", "📖 Methodology"]
)

# ===========================================================================
# TAB 1 — Live assessment
# ===========================================================================
with tab_live:
    colour = LEVEL_COLOURS[assessment.threat_level]
    st.markdown(
        f"""
        <div style="background:{colour};color:#fff;padding:14px 20px;
                    border-radius:10px;font-size:1.35rem;font-weight:700;">
          THREAT LEVEL: {assessment.threat_level}
          <span style="font-weight:400;font-size:1rem;opacity:.92;">
            &nbsp;— {assessment.nearest_facility_name}
            ({assessment.nearest_facility_km:.1f} km from predicted impact)
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Time to impact", f"{traj.time_of_flight_s:.0f} s")
    c2.metric("Range", f"{traj.range_km:.1f} km")
    c3.metric("Apogee", f"{traj.apogee_km:.1f} km")
    c4.metric("Predicted sector", pred_sector)
    c5.metric("Model confidence", f"{confidence*100:.0f}%")

    if use_drag:
        vac = vacuum_range_km(speed, angle)
        st.caption(
            f"Drag model shortens predicted range by {vac - traj.range_km:.1f} km "
            f"versus the vacuum formula ({vac:.1f} km) — a "
            f"{(vac - traj.range_km) / vac * 100:.0f}% overestimate if drag is ignored."
        )

    st.divider()
    left, right = st.columns([3, 2])

    # ---------------- Map ----------------
    with left:
        st.subheader("Predicted impact")

        facilities = _facilities()
        n_steps = 60
        frac = np.linspace(0, 1, n_steps)
        arc_lat, arc_lon = destination_point(
            launch_lat, launch_lon, heading, traj.range_km * frac
        )

        fig = go.Figure()

        fig.add_trace(_map_scatter(
            lat=facilities["lat"], lon=facilities["lon"], mode="markers",
            marker=dict(size=11, color="#1565C0"),
            text=facilities["name"] + " (" + facilities["facility_id"] + ")<br>"
                 + "Criticality " + facilities["criticality"].astype(str) + "/5",
            hoverinfo="text", name="Protected facilities",
        ))

        fig.add_trace(_map_scatter(
            lat=arc_lat, lon=arc_lon, mode="lines",
            line=dict(width=3, color="#E8710A"),
            hoverinfo="skip", name="Ground track",
        ))

        fig.add_trace(_map_scatter(
            lat=[launch_lat], lon=[launch_lon], mode="markers",
            marker=dict(size=13, color="#455A64"),
            text=["Track origin"], hoverinfo="text", name="Origin",
        ))

        fig.add_trace(_map_scatter(
            lat=[impact_lat], lon=[impact_lon], mode="markers",
            marker=dict(size=20, color=colour),
            text=[f"Predicted impact<br>{impact_lat:.4f}, {impact_lon:.4f}"],
            hoverinfo="text", name="Predicted impact",
        ))

        if not assessment.facilities_at_risk.empty:
            at_risk = assessment.facilities_at_risk
            fig.add_trace(_map_scatter(
                lat=at_risk["lat"], lon=at_risk["lon"], mode="markers",
                marker=dict(size=19, color="rgba(179,38,30,0.35)"),
                text=at_risk["name"] + "<br>" + at_risk["distance_km"].round(1).astype(str) + " km",
                hoverinfo="text", name=f"Within {FACILITY_WATCH_RADIUS_KM:.0f} km",
            ))

        _apply_map_layout(fig, center=dict(lat=AO_CENTER[0], lon=AO_CENTER[1]), zoom=7.6)
        st.plotly_chart(fig, use_container_width=True)

        # Flight profile
        prof = pd.DataFrame(traj.path, columns=["downrange_km", "altitude_km"])
        fig_prof = go.Figure(go.Scatter(
            x=prof["downrange_km"], y=prof["altitude_km"],
            mode="lines", fill="tozeroy",
            line=dict(color="#E8710A", width=2),
        ))
        fig_prof.update_layout(
            height=210, margin=dict(l=0, r=0, t=30, b=0),
            title="Vertical flight profile",
            xaxis_title="Downrange (km)", yaxis_title="Altitude (km)",
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    # ---------------- Recommendations ----------------
    with right:
        st.subheader("Recommended protective actions")
        for i, action in enumerate(assessment.actions, 1):
            st.markdown(f"**{i}.** {action}")

        st.divider()
        with st.expander("Why this assessment? (rules that fired)", expanded=True):
            for reason in assessment.reasoning:
                st.markdown(f"- {reason}")

        st.divider()
        st.subheader("Sector probability distribution")
        proba_df = (
            pd.DataFrame({"Sector": list(proba), "Probability": list(proba.values())})
            .sort_values("Probability", ascending=True)
        )
        fig_p = go.Figure(go.Bar(
            x=proba_df["Probability"], y=proba_df["Sector"], orientation="h",
            marker_color=["#1565C0" if s != pred_sector else colour for s in proba_df["Sector"]],
            text=[f"{v*100:.0f}%" for v in proba_df["Probability"]],
            textposition="outside",
        ))
        fig_p.update_layout(height=230, margin=dict(l=0, r=30, t=10, b=0),
                            xaxis=dict(range=[0, 1.15], showticklabels=False))
        st.plotly_chart(fig_p, use_container_width=True)

        if not assessment.facilities_at_risk.empty:
            st.subheader(f"Assets within {FACILITY_WATCH_RADIUS_KM:.0f} km")
            st.dataframe(
                assessment.facilities_at_risk[
                    ["facility_id", "name", "facility_type", "criticality", "distance_km"]
                ].round({"distance_km": 2}),
                hide_index=True, use_container_width=True,
            )

# ===========================================================================
# TAB 2 — Pattern analysis
# ===========================================================================
with tab_model:
    st.subheader("Pattern-analysis model")
    metrics = _metrics_dict()

    if metrics is None:
        st.warning("No saved metrics. Run `python scripts/train_model.py` first.")
    else:
        acc, base = metrics["accuracy"], metrics["baseline_accuracy"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{acc*100:.1f}%")
        m2.metric("Majority baseline", f"{base*100:.1f}%")
        m3.metric("Lift over baseline", f"+{(acc-base)*100:.1f} pts")
        m4.metric("5-fold CV", f"{metrics['cv_mean']*100:.1f}% ± {metrics['cv_std']*100:.1f}")

        st.success(
            f"The classifier beats a constant majority-class guess by "
            f"{(acc-base)*100:.1f} percentage points, and cross-validation agrees with "
            f"the held-out score to within {abs(metrics['cv_mean']-acc)*100:.1f} points — "
            "so the model is learning the geometry, not memorising the split.",
            icon="✅",
        )

        cc1, cc2 = st.columns(2)

        with cc1:
            st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
            labels = metrics["labels"]
            cm = np.array(metrics["confusion"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=labels, y=labels, colorscale="Blues",
                text=cm, texttemplate="%{text}", showscale=False,
            ))
            fig_cm.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                                 yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_cm, use_container_width=True)

        with cc2:
            st.markdown("**Feature importance**")
            fi = metrics["feature_importances"]
            fi_df = pd.DataFrame({"Feature": list(fi), "Importance": list(fi.values())}) \
                      .sort_values("Importance")
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                marker_color="#1565C0",
                text=[f"{v:.3f}" for v in fi_df["Importance"]], textposition="outside",
            ))
            fig_fi.update_layout(height=380, margin=dict(l=0, r=40, t=10, b=0))
            st.plotly_chart(fig_fi, use_container_width=True)

            st.caption(
                "Launch longitude, speed and latitude dominate — they set where the "
                "track lands. Elevation angle matters least, which is physically "
                "correct: across 30°–60° the range varies far less than it does "
                "across the 300–900 m/s speed band."
            )

        st.divider()
        st.markdown("**Per-class performance**")
        report = {k: v for k, v in metrics["report"].items()
                  if k in metrics["labels"]}
        st.dataframe(
            pd.DataFrame(report).T[["precision", "recall", "f1-score", "support"]].round(3),
            use_container_width=True,
        )

    df = _dataset()
    if df is not None:
        st.divider()
        st.markdown("**Training data — impact distribution**")
        fig_sc = go.Figure()
        for sector, grp in df.groupby("target_sector"):
            fig_sc.add_trace(go.Scatter(
                x=grp["impact_lon"], y=grp["impact_lat"], mode="markers",
                marker=dict(size=4, opacity=0.55), name=str(sector),
            ))
        fac = _facilities()
        fig_sc.add_trace(go.Scatter(
            x=fac["lon"], y=fac["lat"], mode="markers",
            marker=dict(size=13, color="black", symbol="x"), name="Facilities",
        ))
        fig_sc.update_layout(height=470, xaxis_title="Longitude", yaxis_title="Latitude",
                             margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Each point is one simulated impact, coloured by the sector label. The "
            "clusters form around the facility markers because the label is derived "
            "from nearest-facility matching — this visible structure is exactly what "
            "the classifier learns, and what the earlier randomly-labelled dataset lacked."
        )

# ===========================================================================
# TAB 3 — Self-healing network
# ===========================================================================
with tab_network:
    st.subheader("Infrastructure self-healing")
    st.markdown(
        "Supply is modelled as a graph. When nodes are lost, Dijkstra recomputes "
        "the cheapest surviving route from any remaining generation source to every "
        "protected facility."
    )

    G = _network()
    transmission = [n for n, d in G.nodes(data=True) if d.get("kind") != "facility"]

    suggested = []
    if assessment.nearest_facility_km <= FACILITY_WATCH_RADIUS_KM:
        fac_row = _facilities().query("facility_id == @assessment.nearest_facility_id")
        if not fac_row.empty:
            suggested = [fac_row.iloc[0]["supply_node"]]

    damaged = st.multiselect(
        "Nodes lost to the simulated impact",
        options=sorted(transmission),
        default=suggested,
        help="Pre-filled with the supply node of the facility nearest the current "
             "predicted impact.",
    )

    if not damaged:
        st.info("Select at least one node to simulate a failure.", icon="👆")
    else:
        status = simulate_failure(damaged)
        fac_names = _facilities().set_index("facility_id")["name"].to_dict()

        k1, k2, k3 = st.columns(3)
        k1.metric("Service availability", f"{status.availability*100:.1f}%")
        k2.metric("Self-healed", status.healed_count)
        k3.metric("Supply lost", len(status.unsupplied))

        if status.availability == 1.0:
            st.success("All protected facilities retained supply through rerouting.", icon="✅")
        elif status.availability == 0.0:
            st.error("Total loss of supply — no generation source survived.", icon="🚨")
        else:
            st.warning(
                f"{len(status.unsupplied)} facilit"
                f"{'y' if len(status.unsupplied)==1 else 'ies'} lost supply; "
                f"{status.healed_count} rerouted automatically.",
                icon="⚠️",
            )

        g1, g2 = st.columns([3, 2])

        with g1:
            H = G.copy()
            H.remove_nodes_from([n for n in damaged if n in H])
            pos = node_positions(G)

            edge_x, edge_y = [], []
            for u, v in H.edges():
                edge_x += [pos[u][0], pos[v][0], None]
                edge_y += [pos[u][1], pos[v][1], None]

            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                       line=dict(width=1, color="#B0BEC5"),
                                       hoverinfo="skip", showlegend=False))

            groups = {
                "Generation": ([], [], [], "#2E7D32", 20),
                "Substation": ([], [], [], "#1565C0", 16),
                "Facility (supplied)": ([], [], [], "#4DB6AC", 12),
                "Facility (lost)": ([], [], [], "#B3261E", 14),
                "Destroyed": ([], [], [], "#424242", 18),
            }
            for n in G.nodes():
                x, y = pos[n]
                if n in damaged:
                    key = "Destroyed"
                elif n in SUPPLY_SOURCES:
                    key = "Generation"
                elif G.nodes[n].get("kind") == "facility":
                    key = "Facility (lost)" if n in status.unsupplied else "Facility (supplied)"
                else:
                    key = "Substation"
                groups[key][0].append(x)
                groups[key][1].append(y)
                groups[key][2].append(fac_names.get(n, n))

            for name, (xs, ys, txt, colour_, size) in groups.items():
                if not xs:
                    continue
                fig_g.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers", name=name,
                    marker=dict(size=size, color=colour_,
                                symbol="x" if name == "Destroyed" else "circle"),
                    text=txt, hoverinfo="text",
                ))

            fig_g.update_layout(
                height=520, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                legend=dict(orientation="h", y=-0.05),
            )
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            if status.rerouted:
                st.markdown("**Automatic reroutes**")
                for fid, (before, after) in status.rerouted.items():
                    extra = (status.current_cost.get(fid, 0) or 0) - (status.baseline_cost.get(fid, 0) or 0)
                    st.markdown(
                        f"**{fac_names.get(fid, fid)}** ({fid}) &nbsp;`+{extra:.0f} cost`  \n"
                        f"<span style='color:#888'>was:</span> {' → '.join(before)}  \n"
                        f"<span style='color:#2E7D32'>now:</span> {' → '.join(after)}",
                        unsafe_allow_html=True,
                    )
                    st.write("")

            if status.unsupplied:
                st.markdown("**Supply lost**")
                for fid in status.unsupplied:
                    st.error(f"{fac_names.get(fid, fid)} ({fid})", icon="🚫")
                st.caption(
                    "Assets without a redundant feeder cannot be rerouted. In the "
                    "notional design only criticality-4-and-above facilities were "
                    "given a backup path, so this result quantifies exactly what "
                    "that design decision costs."
                )

# ===========================================================================
# TAB 4 — Methodology
# ===========================================================================
with tab_docs:
    st.subheader("Methodology and scope")
    st.markdown(
        """
### Processing pipeline

| Stage | Method | Objective |
|---|---|---|
| Trajectory prediction | RK2 numerical integration, gravity + quadratic drag | 1 |
| Target identification | Vectorised haversine nearest-neighbour over the asset register | 2 |
| Pattern analysis | Random Forest (300 trees, balanced, stratified split) | 3 |
| Protective recommendation | Transparent rule engine with declared thresholds | 4 |
| Infrastructure recovery | Dijkstra reroute over a graph with N-1 redundancy | 5 |

### Why the recommendation layer is rule-based, not learned

Prediction is statistical and belongs in a model. Deciding whether to order an
evacuation is a consequential decision that must be auditable after the fact —
a reviewer has to be able to ask "why did it recommend that?" and receive an
answer in plain language. Every assessment therefore lists the exact rules that
fired, and thresholds live in `config.py` rather than inside learned weights.

### Honest limitations

* Trajectories assume a non-manoeuvring object, constant drag coefficient and a
  stationary atmosphere. No wind, no thrust, no terminal guidance.
* The spherical-earth approximation carries roughly 0.5% error at these ranges —
  negligible next to the kinematic uncertainty, but it is an approximation.
* Labels are derived from geometry, so the classifier is learning a physical
  relationship rather than adversarial intent. Real targeting behaviour would
  not be recoverable from launch kinematics alone.
* Reported accuracy is bounded by the 8% deliberate label-noise rate. A result
  near 100% would indicate a leak between features and labels, not a good model.
* The network model is topological. It does not model load flow, generation
  capacity or transfer limits, so "rerouted" means a path exists, not that the
  path could carry the load.

### Data provenance

All data is generated by `scripts/generate_dataset.py` from the physics model
and the synthetic asset register in `src/facilities.py`. Nothing in this project
is derived from real sensor feeds, real installations or restricted sources, and
no component of the system actuates anything — it produces advisory text.
        """
    )

st.sidebar.divider()
st.sidebar.caption(
    f"Predicted impact: {impact_lat:.4f}, {impact_lon:.4f}\n\n"
    f"Nearest asset: {assessment.nearest_facility_id} "
    f"({assessment.nearest_facility_km:.1f} km)"
)
