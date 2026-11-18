"""Request/response contracts shared by the API (Laptop 1) and UI (Laptop 2)."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

LatLon = list[float]


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    model_error: Optional[str]
    decoder: str
    trips_available: int


class TripSummary(BaseModel):
    trip_id: str
    observations: int
    duration_s: float


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: str = Field(min_length=1, max_length=16, pattern=r"^[0-9]+$")


class Metrics(BaseModel):
    """Aggregate scores computed on Laptop 1. Reference labels never leave the server."""
    top1_accuracy: Optional[float]
    classifier_top1_accuracy: Optional[float]
    candidate_recall: Optional[float]
    label_coverage: float
    scored_observations: int
    route_arc_precision: Optional[float]
    route_arc_recall: Optional[float]


class RouteQuality(BaseModel):
    continuity: float
    transitions: int
    disconnected_jumps: int
    unmatched_observations: int


class Geometry(BaseModel):
    gps_points: list[LatLon]                   # observed track, [lat, lon]
    predicted_route: list[list[LatLon]]        # ordered predicted arcs as segments
    road_context: list[list[LatLon]]           # candidate arcs near the track


class PredictResponse(BaseModel):
    trip_id: str
    model_version: str
    decoder: str
    observations: int
    predicted_arc_ids: list[int]               # one per observation, -1 = unmatched
    route_arc_ids: list[int]                   # ordered, consecutive repeats removed
    route_quality: RouteQuality
    evaluation_available: bool
    metrics: Optional[Metrics]
    timings_ms: dict[str, float]
    geometry: Geometry
