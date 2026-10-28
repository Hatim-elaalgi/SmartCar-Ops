"""Download a real street network from OpenStreetMap (Overpass API) for the Real-world mode.

Usage (inside Docker):
    docker compose run --rm dashboard python -m tools.build_road_graph --bbox S,W,N,E --name "My campus"

Example bbox (south, west, north, east) - pick ~1 km x 1 km around your campus on https://bboxfinder.com
(careful: bboxfinder shows W,S,E,N; reorder to S,W,N,E).
Writes resources/road_graph.json. Delete data volume obstacles with "Reset demo data" afterwards.
"""
import argparse
import json
import sys

import networkx as nx
import requests

import config
from core import geo

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ROAD_TYPES = ("primary|secondary|tertiary|residential|unclassified|living_street|"
              "primary_link|secondary_link|tertiary_link")
DEFAULT_BBOX = (48.8420, 2.3400, 48.8520, 2.3560)   # Paris, Latin Quarter
DEFAULT_NAME = "Paris - Quartier Latin"


def fetch(bbox):
    s, w, n, e = bbox
    query = f"""[out:json][timeout:90];
way["highway"~"^({ROAD_TYPES})$"]({s},{w},{n},{e});
(._;>;);
out body;"""
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=120,
                      headers={"User-Agent": "iot-smartcar-dashboard/1.0 (student project)"})
    r.raise_for_status()
    return r.json()["elements"]


def build(elements, bbox, name):
    coords = {el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"}
    ways = [el for el in elements if el["type"] == "way"]

    # A node is an intersection if several ways use it, or if it ends a way.
    usage = {}
    for w in ways:
        for nid in w["nodes"]:
            usage[nid] = usage.get(nid, 0) + 1
    for w in ways:
        usage[w["nodes"][0]] += 1
        usage[w["nodes"][-1]] += 1

    G = nx.Graph()
    for w in ways:
        street = w.get("tags", {}).get("name", "")
        seg = [w["nodes"][0]]
        for nid in w["nodes"][1:]:
            seg.append(nid)
            if usage.get(nid, 0) >= 2 and len(seg) >= 2:
                u, v = seg[0], seg[-1]
                if u != v and all(p in coords for p in seg):
                    geom = [coords[p] for p in seg]
                    length = geo.cumulative(config.REAL, geom)[-1]
                    if not G.has_edge(u, v) or G[u][v]["length"] > length:
                        G.add_edge(u, v, length=length, geometry=geom, name=street)
                seg = [nid]

    largest = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest).copy()

    names = {n: set() for n in G.nodes}
    for u, v, d in G.edges(data=True):
        if d["name"]:
            names[u].add(d["name"])
            names[v].add(d["name"])

    def label(n):
        streets = sorted(names[n])
        if not streets:
            return f"Node {n}"
        return " / ".join(streets[:2]) if len(streets) > 1 else f"{streets[0]} (end)"

    nodes = [{"id": str(n), "x": coords[n][0], "y": coords[n][1], "label": label(n)} for n in G.nodes]
    edges = []
    for u, v, d in G.edges(data=True):
        geom = d["geometry"]
        if geom[0] != coords[u]:
            geom = list(reversed(geom))
        edges.append({"u": str(u), "v": str(v), "name": d["name"], "length": round(d["length"], 1),
                      "geometry": [[round(x, 7), round(y, 7)] for x, y in geom]})
    return {"name": name, "source": "OpenStreetMap (c) contributors, ODbL", "bbox": list(bbox),
            "nodes": nodes, "edges": edges}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", help="south,west,north,east", default=",".join(map(str, DEFAULT_BBOX)))
    p.add_argument("--name", default=DEFAULT_NAME)
    args = p.parse_args()
    bbox = tuple(float(v) for v in args.bbox.split(","))
    print(f"Downloading roads in {bbox} ...")
    data = build(fetch(bbox), bbox, args.name)
    config.ROAD_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.ROAD_GRAPH_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"Saved {len(data['nodes'])} intersections and {len(data['edges'])} road segments "
          f"to {config.ROAD_GRAPH_PATH}")


if __name__ == "__main__":
    sys.exit(main())
