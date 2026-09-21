"""
Infrastructure self-healing simulation (Objective 5).

FIXES OVER THE PROTOTYPE
------------------------
* The prototype ran a demonstration at import time, printing to stdout on
  every `import`. Nothing executes here on import.
* It rerouted a single hard-coded pair (Power_Plant_A -> Substation_South)
  regardless of which node the user damaged, so damaging the source itself
  produced a meaningless result. Routing is now computed per protected
  facility from whichever generation sources survive.
* It could only fail one node. Real impacts take out whatever is inside the
  footprint, so `simulate_failure` accepts any number of nodes.
* It reported only a path. It now reports which facilities lost supply,
  which were saved by rerouting, and how much the detour cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from src.facilities import load_facilities

__all__ = [
    "NetworkStatus",
    "build_network",
    "supply_sources",
    "simulate_failure",
    "node_positions",
]

# Generation sources: where supply originates.
SUPPLY_SOURCES = ["Power_Plant_A", "Power_Plant_B"]

# Transmission topology. Weight = notional transfer cost (distance/losses).
_EDGES: list[tuple[str, str, float]] = [
    ("Power_Plant_A", "Substation_North", 10),
    ("Power_Plant_A", "Substation_West", 14),
    ("Power_Plant_B", "Substation_South", 12),
    ("Power_Plant_B", "Substation_East", 16),
    # Ring interconnects — these are what make self-healing possible at all.
    ("Substation_North", "Substation_East", 9),
    ("Substation_East", "Substation_South", 8),
    ("Substation_South", "Substation_West", 11),
    ("Substation_West", "Substation_North", 7),
    # Cross-tie between the two generation plants.
    ("Power_Plant_A", "Power_Plant_B", 20),
]


@dataclass
class NetworkStatus:
    damaged_nodes: list
    supplied: dict = field(default_factory=dict)      # facility_id -> path
    unsupplied: list = field(default_factory=list)    # facility_ids with no route
    rerouted: dict = field(default_factory=dict)      # facility_id -> (before, after)
    baseline_cost: dict = field(default_factory=dict)
    current_cost: dict = field(default_factory=dict)

    @property
    def healed_count(self) -> int:
        return len(self.rerouted)

    @property
    def total_facilities(self) -> int:
        return len(self.supplied) + len(self.unsupplied)

    @property
    def availability(self) -> float:
        """Fraction of protected facilities still receiving supply."""
        if self.total_facilities == 0:
            return 0.0
        return len(self.supplied) / self.total_facilities


def build_network() -> nx.Graph:
    """
    Build the transmission graph, attaching each protected facility as a leaf
    on its designated supply node.

    An undirected graph is used deliberately. The prototype used a DiGraph,
    which meant power could only ever flow one way around the ring — so a
    reroute that required reversing flow on a single link was reported as a
    total outage even though the physical network could supply it. Real
    transmission links are bidirectional, and modelling them as such is what
    lets the ring topology actually heal.
    """
    G = nx.Graph()

    for u, v, w in _EDGES:
        G.add_edge(u, v, weight=w, kind="transmission")

    for _, fac in load_facilities().iterrows():
        G.add_node(fac["facility_id"], kind="facility", name=fac["name"],
                   criticality=int(fac["criticality"]))
        G.add_edge(fac["facility_id"], fac["supply_node"], weight=3, kind="feeder")

        # Redundant feeder, where one exists. Weighted higher than the primary
        # so Dijkstra prefers the normal route and only falls back when the
        # primary path is gone — which is exactly the self-healing behaviour.
        backup = fac.get("backup_node")
        if isinstance(backup, str) and backup:
            G.add_edge(fac["facility_id"], backup, weight=9, kind="backup_feeder")

    return G


def supply_sources(G: nx.Graph) -> list[str]:
    """Generation nodes still present in the graph."""
    return [n for n in SUPPLY_SOURCES if n in G]


def _cheapest_supply(G: nx.Graph, facility_id: str):
    """
    Cheapest route to `facility_id` from any surviving source.

    Returns (path, cost) or (None, None) if the facility is islanded.
    """
    best_path, best_cost = None, None

    for source in supply_sources(G):
        try:
            cost, path = nx.single_source_dijkstra(
                G, source, target=facility_id, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if best_cost is None or cost < best_cost:
            best_path, best_cost = path, cost

    return best_path, best_cost


def simulate_failure(damaged_nodes) -> NetworkStatus:
    """
    Remove the damaged nodes and recompute supply for every facility.

    Compares against the undamaged baseline so the result distinguishes three
    outcomes that matter operationally: unaffected, rerouted (self-healed),
    and lost.
    """
    if isinstance(damaged_nodes, str):
        damaged_nodes = [damaged_nodes]
    damaged_nodes = list(damaged_nodes)

    baseline = build_network()
    damaged = build_network()

    for node in damaged_nodes:
        if node in damaged:
            damaged.remove_node(node)

    status = NetworkStatus(damaged_nodes=damaged_nodes)

    for facility_id in [n for n, d in baseline.nodes(data=True) if d.get("kind") == "facility"]:
        base_path, base_cost = _cheapest_supply(baseline, facility_id)
        status.baseline_cost[facility_id] = base_cost

        if facility_id in damaged_nodes:
            status.unsupplied.append(facility_id)
            continue

        new_path, new_cost = _cheapest_supply(damaged, facility_id)

        if new_path is None:
            status.unsupplied.append(facility_id)
            continue

        status.supplied[facility_id] = new_path
        status.current_cost[facility_id] = new_cost

        if base_path is not None and new_path != base_path:
            status.rerouted[facility_id] = (base_path, new_path)

    return status


def node_positions(G: nx.Graph, seed: int = 7) -> dict:
    """Stable 2-D layout so the graph does not jump between reruns."""
    return nx.spring_layout(G, seed=seed, k=0.55, iterations=200)
