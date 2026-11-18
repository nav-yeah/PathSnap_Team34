"""Pluggable route decoders.

A decoder turns per-candidate classifier scores into one road arc per GPS
observation. The contract is deliberately small so the team HMM/Viterbi module
can be dropped in via ``PATHSNAP_DECODER=package.module:function`` without
changing the API:

    decode(scores, candidate_arcs, context) -> np.ndarray[int] of shape (N,)

* ``scores``          (N, C) float, independent binary scores in [0, 1]
* ``candidate_arcs``  (N, C) int, arc ids aligned with ``scores``
* ``context``         DecodeContext (reference-free trip inputs)

Return -1 for an observation the decoder leaves unmatched.
"""
import importlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecodeContext:
    track: np.ndarray        # (N, 3) lon, lat, t
    nodes: np.ndarray        # (M, 2) lon, lat
    arcs: np.ndarray         # (K, 2) from, to
    points_m: np.ndarray     # (N, 2) trip-local metric coordinates
    starts_m: np.ndarray     # (K, 2) arc start, same metric frame
    ends_m: np.ndarray       # (K, 2) arc end, same metric frame


def classifier_argmax(scores, candidate_arcs, context):
    """Classifier-only baseline: highest score wins; ties keep candidate order."""
    return candidate_arcs[np.arange(len(scores)), np.argmax(scores, axis=1)]


BUILTIN = {"classifier_argmax": classifier_argmax}


def resolve(spec):
    if spec in BUILTIN:
        return spec, BUILTIN[spec]
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"PATHSNAP_DECODER must be a builtin or 'module:function', got {spec!r}")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError(f"{spec} is not callable")
    return spec, function
