"""Decoders used only by tests to exercise the PATHSNAP_DECODER plug-in path."""
import numpy as np


def first_candidate(scores, candidate_arcs, context):
    return candidate_arcs[:, 0]


def wrong_shape(scores, candidate_arcs, context):
    return candidate_arcs[:3, 0]


def leave_half_unmatched(scores, candidate_arcs, context):
    decoded = candidate_arcs[:, 0].copy()
    decoded[::2] = -1
    return decoded
