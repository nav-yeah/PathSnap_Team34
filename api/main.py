"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pathsnap_ml.inference import load

from . import pipeline
from .config import Settings
from .decoders import resolve
from .evaluation import ReferenceStore, score
from .schemas import Health, PredictRequest, PredictResponse, TripSummary
from .trip_store import InvalidTripData, TripNotFound, TripStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("pathsnap.api")


class State:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.trips = TripStore(settings.trips_dir)
        self.references = ReferenceStore(settings.reference_dir)
        self.decoder_name, self.decoder = resolve(settings.decoder)
        self.artifact, self.model_error = None, None
        try:
            self.artifact = load(settings.model_path)
        except FileNotFoundError:
            self.model_error = f"model artifact not found: {settings.model_path}"
        except Exception as error:  # incompatible schema, corrupt file, version mismatch
            self.model_error = f"model artifact could not be loaded: {error}"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        state = app.state.pathsnap = State(settings)
        log.info("model=%s version=%s decoder=%s trips_dir=%s trips=%d",
                 settings.model_path, state.artifact and state.artifact["version"],
                 state.decoder_name, settings.trips_dir, len(state.trips.trip_ids()))
        if state.model_error:
            log.error(state.model_error)
        yield

    app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan,
                  description="GPS map matching and route reconstruction — Team 34")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        client = request.client.host if request.client else "-"
        log.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path,
                 response.status_code, (time.perf_counter() - start) * 1000)
        return response

    def state(request: Request) -> State:
        return request.app.state.pathsnap

    @app.get("/health", response_model=Health)
    def health(request: Request):
        s = state(request)
        return Health(status="ok" if s.artifact else "degraded", model_loaded=s.artifact is not None,
                      model_version=s.artifact["version"] if s.artifact else None,
                      model_error=s.model_error, decoder=s.decoder_name,
                      trips_available=len(s.trips.trip_ids()))

    @app.get("/trips", response_model=list[TripSummary])
    def trips(request: Request):
        s = state(request)
        summaries = []
        for trip_id in s.trips.trip_ids():
            try:
                summaries.append(TripSummary(**s.trips.summary(trip_id)))
            except InvalidTripData as error:
                log.warning("skipping trip %s: %s", trip_id, error)
        return summaries

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request):
        s = state(request)
        if s.artifact is None:
            raise HTTPException(503, detail=s.model_error or "model not loaded")
        started = time.perf_counter()
        try:
            trip = s.trips.load(body.trip_id)
        except TripNotFound:
            raise HTTPException(404, detail=f"unknown trip_id {body.trip_id!r}")
        except InvalidTripData as error:
            raise HTTPException(422, detail=f"trip {body.trip_id} has invalid data: {error}")
        log.info("predict trip=%s observations=%d arcs=%d", trip.trip_id, len(trip.track),
                 len(trip.arcs))
        try:
            result = pipeline.run(trip, s.artifact, s.decoder)
        except Exception as error:
            log.exception("pipeline failed for trip %s", trip.trip_id)
            raise HTTPException(500, detail=f"inference failed: {type(error).__name__}: {error}")

        # ---- Prediction is final here. Reference data is only touched below. ----
        metrics, clock = None, time.perf_counter()
        route = s.references.route(trip.trip_id)
        if route is not None:
            try:
                metrics = score(result, route)
            except Exception as error:
                log.warning("evaluation unavailable for trip %s: %s", trip.trip_id, error)
        result.timings["evaluation_s"] = time.perf_counter() - clock
        result.timings["total_s"] = time.perf_counter() - started

        route_ids = pipeline.collapse(result.decoded_arcs)
        quality = pipeline.continuity(result.decoded_arcs, trip.arcs) | dict(
            unmatched_observations=int((result.decoded_arcs < 0).sum()))
        context_ids = np.unique(result.candidate_arcs)[: s.settings.max_context_arcs]
        log.info("trip=%s top1=%s classifier_top1=%s continuity=%.3f total=%.2fs",
                 trip.trip_id, _fmt(metrics, "top1_accuracy"),
                 _fmt(metrics, "classifier_top1_accuracy"), quality["continuity"],
                 result.timings["total_s"])
        return PredictResponse(
            trip_id=trip.trip_id, model_version=s.artifact["version"], decoder=s.decoder_name,
            observations=len(trip.track), predicted_arc_ids=result.decoded_arcs.tolist(),
            route_arc_ids=route_ids, route_quality=quality,
            evaluation_available=metrics is not None, metrics=metrics,
            timings_ms={k: round(v * 1000, 1) for k, v in result.timings.items()},
            geometry=dict(
                gps_points=np.round(trip.track[:, 1::-1], 7).tolist(),
                predicted_route=pipeline.segment_coords(route_ids, trip.nodes, trip.arcs),
                road_context=pipeline.segment_coords(context_ids, trip.nodes, trip.arcs)))

    return app


def _fmt(metrics, key):
    return "n/a" if not metrics or metrics[key] is None else f"{metrics[key]:.4f}"


app = create_app()
