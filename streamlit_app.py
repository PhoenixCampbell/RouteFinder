
import streamlit as st
import folium
from streamlit_folium import st_folium

from routegen import build_route_for_place

st.set_page_config(page_title="Trail-First Route Generator", page_icon="🥾")

st.title("🥾 Trail-First Walk/Hike Route Generator")
st.write("Give me an **area** and a **target distance**, and I'll try to build a loop that uses existing trails and walking paths.")

with st.form("params"):
    place = st.text_input("Area or place name (city, park, neighborhood, etc.)", value="Golden Gate Park, San Francisco")
    km = st.slider("Target distance (km)", 1.0, 30.0, 5.0, 0.5)
    trail_bias = st.slider("Prefer trails vs. roads", 0.5, 2.0, 0.9, 0.05, help="Lower = more trail preference, Higher = more road tolerance")
    submitted = st.form_submit_button("Generate route")

if submitted:
    try:
        with st.spinner("Generating route..."):
            result = build_route_for_place(place, kilometers=km, trail_bias=trail_bias)

        st.success(f"Built a loop in **{result['place']}** — estimated distance: **{result['distance_m']/1000:.2f} km**")

        center = result["center"]
        m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
        folium.PolyLine(result["latlon"], weight=5, opacity=0.9).add_to(m)
        folium.Marker(location=result["latlon"][0], tooltip="Start/Finish").add_to(m)

        st_folium(m, width=800)

        gpx_bytes = result["gpx"].encode("utf-8")
        st.download_button("⬇️ Download GPX", data=gpx_bytes, file_name="route.gpx", mime="application/gpx+xml")

        with st.expander("Debug details"):
            st.json({
                "points": result["num_points"],
                "center": result["center"],
                "distance_m": result["distance_m"]
            })

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Try a larger or more specific area, or increase the target distance.")
else:
    st.info("Tip: parks or neighborhoods work great. Try 'Central Park, NYC' or 'Asheville, NC'.")
