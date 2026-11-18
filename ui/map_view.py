"""Folium map for one /predict response.

Layers: candidate road network (grey), observed GPS (blue), predicted route
(red), start/end markers, plus a legend and layer control. The reference route
is intentionally absent: the API never sends it to the UI.

CLI (saves an interactive HTML map for the report):
    python -m ui.map_view --api http://<laptop1-ip>:8000 --trip 00000052 --out map.html
"""
import argparse

import folium
from branca.element import Element

COLORS = dict(road="#8a8f98", gps="#1f6feb", route="#d1242f")
MAX_GPS_MARKERS = 600


def build_map(result):
    geometry = result["geometry"]
    gps = geometry["gps_points"]
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))

    road = folium.FeatureGroup(name="Road network (candidate arcs)", show=True)
    for segment in geometry["road_context"]:
        folium.PolyLine(segment, color=COLORS["road"], weight=2, opacity=0.5).add_to(road)
    road.add_to(fmap)

    observed = folium.FeatureGroup(name="Observed GPS track", show=True)
    folium.PolyLine(gps, color=COLORS["gps"], weight=2, opacity=0.6, dash_array="4 4").add_to(observed)
    step = max(1, len(gps) // MAX_GPS_MARKERS)
    for index in range(0, len(gps), step):
        folium.CircleMarker(gps[index], radius=2.5, color=COLORS["gps"], fill=True,
                            fill_opacity=0.9, weight=0,
                            tooltip=f"GPS #{index}").add_to(observed)
    observed.add_to(fmap)

    predicted = folium.FeatureGroup(name="Predicted route", show=True)
    for order, segment in enumerate(geometry["predicted_route"]):
        folium.PolyLine(segment, color=COLORS["route"], weight=5, opacity=0.85,
                        tooltip=f"arc {result['route_arc_ids'][order]}").add_to(predicted)
    predicted.add_to(fmap)

    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(Element(_legend(result["trip_id"])))
    return fmap


def _legend(trip_id):
    rows = "".join(
        f'<div><span style="display:inline-block;width:22px;height:4px;background:{color};'
        f'vertical-align:middle;margin-right:6px"></span>{label}</div>'
        for color, label in [(COLORS["gps"], "Observed GPS"), (COLORS["route"], "Predicted route"),
                             (COLORS["road"], "Road network (candidates)")])
    return (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            f'padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);'
            f'font:13px sans-serif;color:#111"><b>Trip {trip_id}</b>{rows}</div>')


if __name__ == "__main__":
    from .api_client import PathSnapClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--trip", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_map(PathSnapClient(args.api).predict(args.trip)).save(args.out)
    print(f"saved {args.out}")
