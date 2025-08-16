from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any

import networkx as nx

try:
    import osmnx as ox
    from shapely.geometry import Polygon, MultiPolygon
except Exception as e:
    raise RuntimeError("This project requires osmnx and shapely. Install from requirements.txt") from e


def _coerce_highway(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _edge_pref_multiplier(highway_list: list[str]) -> float:
    mult = 1.0
    for h in highway_list:
        if h in ("footway", "path", "steps", "bridleway"):
            mult *= 0.7
        elif h in ("track",):
            mult *= 0.9
        elif h in ("pedestrian", "living_street"):
            mult *= 0.95
        elif h in ("residential", "service"):
            mult *= 1.05
        elif h in ("primary", "secondary", "tertiary"):
            mult *= 1.6
        elif h in ("primary_link", "secondary_link", "tertiary_link"):
            mult *= 1.5
        elif h in ("unclassified",):
            mult *= 1.1
    return mult


def add_weight_attribute(G: nx.MultiDiGraph, trail_bias: float = 1.0, used_edge_ids: set[Tuple[int,int,int]] | None = None) -> None:
    if used_edge_ids is None:
        used_edge_ids = set()

    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 1.0))
        highway = _coerce_highway(data.get("highway"))
        mult = _edge_pref_multiplier(highway)
        weight = length * (mult ** trail_bias)
        if (u, v, k) in used_edge_ids:
            weight *= 1.4
        data["weight"] = weight


def geocode_area(place: str):
    gdf = ox.geocode_to_gdf(place)
    geom = gdf.iloc[0].geometry
    name = gdf.iloc[0].get("display_name", place)
    if isinstance(geom, (Polygon, MultiPolygon)):
        centroid = geom.centroid
        center = (centroid.y, centroid.x)
        return geom, name, center
    else:
        raise ValueError("Could not resolve a polygon for this place.")


def download_walk_graph(polygon, network: str = "walk") -> nx.MultiDiGraph:
    """
    Robust default walking graph (no custom filter), keep all components.
    """
    G = ox.graph_from_polygon(polygon, network_type=network, simplify=True, retain_all=True)
    return G


def total_length_m(G: nx.MultiDiGraph, route: List[int]) -> float:
    length = 0.0
    for u, v in zip(route[:-1], route[1:]):
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", 0))
        length += float(data.get("length", 0))
    return length


def route_nodes_to_latlon(G: nx.MultiDiGraph, route: List[int]) -> List[Tuple[float, float]]:
    return [(G.nodes[nid]["y"], G.nodes[nid]["x"]) for nid in route]


def choose_start_node(G: nx.MultiDiGraph, near_latlon: Tuple[float, float] | None = None) -> int:
    if near_latlon is None:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        sub = G.subgraph(largest_cc)
        start = max(sub.degree, key=lambda x: x[1])[0]
        return start
    else:
        lat, lon = near_latlon
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)


def _shortest_path(G: nx.MultiDiGraph, src: int, dst: int) -> List[int]:
    return nx.shortest_path(G, src, dst, weight="weight")


def loop_route(
    G: nx.MultiDiGraph,
    start: int,
    target_distance_m: float,
    trail_bias: float = 1.0,
    candidates: int = 200
) -> List[int]:
    """
    Build a loop by picking a far-ish node (weighted distance) then returning via a different path.
    """
    import random

    add_weight_attribute(G, trail_bias=trail_bias)

    # Precompute distances with NO cutoff, then sample from reachable set
    lengths = nx.single_source_dijkstra_path_length(G, start, weight="weight")
    if not lengths:
        raise RuntimeError("No reachable nodes from start; try a bigger area or different filters.")

    reachable_nodes = list(lengths.keys())
    rng = random.Random(42)
    sample = rng.sample(reachable_nodes, min(candidates, len(reachable_nodes)))

    half = target_distance_m / 2.0
    best_node = None
    best_delta = float("inf")

    for n in sample:
        d = lengths.get(n)
        if d is None:
            continue
        delta = abs(d - half)
        if delta < best_delta:
            best_delta = delta
            best_node = n

    if best_node is None:
        best_node = max(lengths.items(), key=lambda kv: kv[1])[0]

    out_path = _shortest_path(G, start, best_node)

    # Penalize outbound edges to encourage a different return path
    used_edges = set()
    for u, v in zip(out_path[:-1], out_path[1:]):
        for k in G.get_edge_data(u, v).keys():
            used_edges.add((u, v, k))
    add_weight_attribute(G, trail_bias=trail_bias, used_edge_ids=used_edges)

    back_path = _shortest_path(G, best_node, start)
    return out_path + back_path[1:]


def export_gpx(latlon_points: List[Tuple[float, float]], name: str = "route") -> str:
    import gpxpy
    import gpxpy.gpx as gpxmod

    gpx = gpxmod.GPX()
    gpx_track = gpxmod.GPXTrack(name=name)
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxmod.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for lat, lon in latlon_points:
        gpx_segment.points.append(gpxmod.GPXTrackPoint(lat, lon))

    return gpx.to_xml()


def build_route_for_place(
    place: str,
    kilometers: float,
    trail_bias: float = 1.0,
    start_latlon: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    polygon, display_name, center = geocode_area(place)
    G = download_walk_graph(polygon, network="walk")
    # Keep only the largest connected component to avoid tiny fragments
    G = ox.utils_graph.get_largest_component(G, strongly=False)

    if start_latlon is None:
        start = choose_start_node(G, near_latlon=center)
    else:
        start = choose_start_node(G, near_latlon=start_latlon)

    route = loop_route(G, start=start, target_distance_m=kilometers*1000.0, trail_bias=trail_bias)

    latlon = route_nodes_to_latlon(G, route)
    dist_m = total_length_m(G, route)

    return {
        "place": display_name,
        "center": center,
        "distance_m": dist_m,
        "num_points": len(latlon),
        "latlon": latlon,
        "gpx": export_gpx(latlon, name=f"{display_name} loop ~{kilometers:.1f}km")
    }
