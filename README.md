
# Trail‑First Walk/Hike Route Generator

A tiny Streamlit app that takes a user-specified **area** (e.g., "Central Park, NYC" or "Boulder, CO") and builds an approximate **loop route** using **existing trails and walking paths** from OpenStreetMap.

## Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run streamlit_app.py
```

Then open the URL that Streamlit prints (typically http://localhost:8501).

## How it works

- Geocodes the area to a polygon with `osmnx`.
- Downloads a walking graph with a custom filter that includes footways, paths, tracks, and quiet streets.
- Finds a node roughly half your requested distance from the start via weighted shortest paths, then returns via a **different** path by lightly penalizing the outbound edges.
- Prefers trails via a weight multiplier on OSM `highway` types (you can tune this with the **Prefer trails vs. roads** slider).
- Outputs a GPX you can load into map apps or watches.

## Notes & Limits

- The route is heuristic and "best effort" — small or sparse areas may not support the requested distance.
- Locked/unsigned private ways are filtered out where tagged, but OSM data can be imperfect. **Use common sense on the ground.**
- For long distances, use larger areas (e.g., a city or regional park system).

## Customize

If you want point-to-point instead of loops, or to force start coordinates, call the library directly:

```python
from routegen import geocode_area, download_walk_graph, choose_start_node, loop_route, route_nodes_to_latlon, export_gpx
```

Have fun and hike safely! 🥾
