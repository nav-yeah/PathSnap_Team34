"""Runtime configuration for the PathSnap inference server (Laptop 1).

All settings come from environment variables so the same code runs on any
laptop without edits. Paths are resolved relative to the repository root.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(name, default):
    value = Path(os.environ.get(name, default)).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


@dataclass(frozen=True)
class Settings:
    # Trusted, team-produced artifact (never a user upload).
    model_path: Path = field(default_factory=lambda: _path(
        "PATHSNAP_MODEL", "artifacts/model.joblib"))
    # Inference inputs: <trips_dir>/<trip_id>/<trip_id>.{track,nodes,arcs}
    trips_dir: Path = field(default_factory=lambda: _path(
        "PATHSNAP_TRIPS_DIR", "data/eval_trips"))
    # Reference routes (.route) for server-side scoring only. Defaults to the
    # trips folder, but can be pointed at a separate, restricted folder.
    reference_dir: Path = field(default_factory=lambda: _path(
        "PATHSNAP_REFERENCE_DIR", os.environ.get("PATHSNAP_TRIPS_DIR", "data/eval_trips")))
    # "classifier_argmax" or "package.module:function" (e.g. the team HMM).
    decoder: str = field(default_factory=lambda: os.environ.get(
        "PATHSNAP_DECODER", "classifier_argmax"))
    cors_origins: tuple = field(default_factory=lambda: tuple(
        origin.strip() for origin in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",")
        if origin.strip()))
    # Upper bound on road-network segments returned for map context.
    max_context_arcs: int = field(default_factory=lambda: int(
        os.environ.get("PATHSNAP_MAX_CONTEXT_ARCS", "6000")))
