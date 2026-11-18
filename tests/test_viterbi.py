import numpy as np

from api.decoders import DecodeContext
from pathsnap_ml.viterbi import decode


def context():
    nodes = np.array([[0., 0.], [10., 0.], [20., 0.], [15., 1.], [30., 1.]])
    arcs = np.array([[0, 1], [1, 2], [3, 4]])
    points = np.array([[5., 0.], [15., 0.]])
    return DecodeContext(np.column_stack((points, [0, 1])), nodes, arcs,
                         points, nodes[arcs[:, 0]], nodes[arcs[:, 1]])


def test_directed_transition_overrides_disconnected_candidate():
    ids = [[0, -1], [2, 1]]
    np.testing.assert_equal(decode([[.9, .1], [.99, .4]], ids, context()), [0, 1])


def test_invalid_observation_and_restart():
    np.testing.assert_equal(decode([[np.nan, np.inf], [0, 1]],
                                   [[0, 1], [2, 999]], context()), [-1, 2])


def test_ties_are_repeatable_and_empty_input():
    np.testing.assert_equal(decode([[1, 1], [1, 1]], [[0, 0], [1, 1]], context()), [0, 1])
    current = context()
    empty = DecodeContext(current.track[:0], current.nodes, current.arcs,
                          current.points_m[:0], current.starts_m, current.ends_m)
    assert decode(np.empty((0, 0)), np.empty((0, 0), dtype=int), empty).size == 0
