
# Trail-First Walk/Hike Route Generator (patched)

This version applies stability fixes:
- Default OSMnx walking network (no custom filter)
- Keep largest connected component
- Remove Dijkstra cutoff; sample from reachable nodes
- Added `scikit-learn` for fast unprojected nearest-node queries

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py

