"""Request/response contracts shared by the API (Laptop 1) and UI (Laptop 2)."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

LatLon = list[float]


class ModelInfo(BaseModel):
    """Which artifact is serving. Lets evaluators verify the demo uses the committed model."""
    model_config = ConfigDict(protected_namespaces=())
    version: str
    family: str
    estimator: str
    key_parameters: dict
    fitted_iterations: Optional[int]
    schema_id: str = Field(alias="schema")
    features: list[str]
    sha256: str
    size_bytes: int
    sklearn_runtime: str
    sklearn_trained: Optional[str]
    version_warnings: list[str]


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: Optional[str]
    model_error: Optional[str]
    model: Optional[ModelInfo]
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
    inserted_bridge_arcs: int
    route_breaks: int
    gps_gap_breaks: int


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
    route_arc_ids: list[int]                   # stitched ordered route; bridge arcs included
    route_break_indices: list[int]             # starts of route components after a break
    route_quality: RouteQuality
    evaluation_available: bool
    metrics: Optional[Metrics]
    timings_ms: dict[str, float]
    geometry: Geometry
