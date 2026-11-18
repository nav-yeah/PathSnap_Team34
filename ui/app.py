"""PathSnap evaluator UI (Laptop 2).

Run:  PATHSNAP_API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
The UI only sends a trip id and renders what the API returns. It holds no model,
no road-network processing and no reference labels.
"""
import os
import sys
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui.api_client import ApiError, PathSnapClient  # noqa: E402
from ui.map_view import build_map  # noqa: E402

st.set_page_config(page_title="PathSnap — Team 34", page_icon="🗺️", layout="wide")


def pct(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API URL (Laptop 1)",
                            value=os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"))
    client = PathSnapClient(api_url)
    try:
        health = client.health()
        if health["model_loaded"]:
            st.success("API connected")
        else:
            st.warning(f"API reachable but model not loaded: {health['model_error']}")
        model = health.get("model") or {}
        if model:
            st.markdown(f"**{model['family']}**")
            params = ", ".join(f"{k}={v}" for k, v in model["key_parameters"].items())
            st.caption(f"Version: `{model['version']}`  \nSHA-256: `{model['sha256'][:12]}…`  \n"
                       f"Params: {params}")
            for warning in model.get("version_warnings", []):
                st.warning(f"scikit-learn mismatch: {warning}")
        st.caption(f"Decoder: `{health['decoder']}`  \nTrips on server: {health['trips_available']}")
    except ApiError as error:
        health = None
        st.error(str(error))
    if st.button("Reconnect", use_container_width=True):
        st.session_state.pop("trips", None)
        st.rerun()

st.title("PathSnap — GPS Map Matching & Route Reconstruction")
st.caption("Team 34 · Select an evaluation trip; Laptop 1 runs candidate generation, the "
           "classifier and route decoding, and scores the result against its private reference data.")

if health is None:
    st.stop()

if st.session_state.get("trips_url") != api_url or "trips" not in st.session_state:
    try:
        st.session_state.trips = client.trips()
        st.session_state.trips_url = api_url
    except ApiError as error:
        st.error(str(error))
        st.stop()

trips = st.session_state.trips
if not trips:
    st.warning("The API reports no trips. Check PATHSNAP_TRIPS_DIR on Laptop 1.")
    st.stop()

labels = {t["trip_id"]: f"Trip {t['trip_id']} · {t['observations']} GPS points" for t in trips}
left, right = st.columns([3, 1])
trip_id = left.selectbox("Evaluation trip", list(labels), format_func=labels.get)
right.write("")
right.write("")
run = right.button("Reconstruct route", type="primary", use_container_width=True,
                   disabled=not health["model_loaded"])

if run:
    with st.spinner(f"Laptop 1 is processing trip {trip_id}…"):
        try:
            st.session_state.result = client.predict(trip_id)
        except ApiError as error:
            st.session_state.pop("result", None)
            st.error(str(error))

result = st.session_state.get("result")
if result:
    st.subheader(f"Trip {result['trip_id']}")
    metrics, quality = result["metrics"], result["route_quality"]
    cols = st.columns(5)
    if metrics:
        cols[0].metric("Top-1 accuracy", pct(metrics["top1_accuracy"]),
                       help="Share of confidently-labelled GPS points matched to the reference arc. "
                            "Computed on Laptop 1.")
        cols[1].metric("Candidate recall", pct(metrics["candidate_recall"]),
                       help="Reference arc present in the candidate set. Upper bound for Top-1, "
                            "not model accuracy.")
        cols[2].metric("Route arc precision", pct(metrics["route_arc_precision"]))
        cols[3].metric("Route arc recall", pct(metrics["route_arc_recall"]))
    else:
        cols[0].info("No reference route on the server for this trip; accuracy not computed.")
    cols[4].metric("Route continuity", pct(quality["continuity"]),
                   help="Share of arc changes where the next arc starts where the previous one ends.")

    cols = st.columns(5)
    if metrics:
        cols[0].metric("Classifier-only Top-1", pct(metrics["classifier_top1_accuracy"]))
        cols[1].metric("Label coverage", pct(metrics["label_coverage"]),
                       help="GPS points with a confident reference-proxy label (others excluded).")
    cols[2].metric("Disconnected jumps", quality["disconnected_jumps"])
    cols[3].metric("GPS points", result["observations"])
    cols[4].metric("Server time", f"{result['timings_ms']['total_s'] / 1000:.2f} s")
    st.caption(f"Bridge arcs added: {quality['inserted_bridge_arcs']} · Route breaks: "
               f"{quality['route_breaks']} · GPS-gap breaks: {quality['gps_gap_breaks']}")

    st_folium(build_map(result), height=620, use_container_width=True, returned_objects=[],
              key=f"map-{result['trip_id']}")

    with st.expander("Response details"):
        st.json({k: v for k, v in result.items() if k not in ("geometry", "predicted_arc_ids")})
