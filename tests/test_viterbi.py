import numpy as np

from api.decoders import DecodeContext
from api.pipeline import stitch_route, stitched_continuity
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


def test_stitch_inserts_shortest_directed_bridge():
    arcs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    nodes = np.array([[0., 0.], [10., 0.], [20., 0.], [30., 0.]])
    starts, ends = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    route, breaks, inserted, gaps = stitch_route(
        [0, 2], [0, 1], arcs, np.array([[5., 0.], [25., 0.]]), starts, ends)
    assert route == [0, 1, 2]
    assert breaks == [] and inserted == 1 and gaps == 0
    assert stitched_continuity(route, breaks, arcs) == dict(
        continuity=1.0, transitions=2, disconnected_jumps=0)


def test_stitch_does_not_bridge_gaps_or_disconnected_components():
    arcs = np.array([[0, 1], [1, 2], [5, 6]])
    nodes = np.array([[0., 0.], [10., 0.], [20., 0.], [30., 0.], [40., 0.],
                      [50., 0.], [60., 0.]])
    starts, ends = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    route, breaks, inserted, gaps = stitch_route(
        [0, 1, 2], [0, 40, 41], arcs, np.array([[5., 0.], [15., 0.], [55., 0.]]),
        starts, ends)
    assert route == [0, 1, 2]
    assert breaks == [1, 2] and inserted == 0 and gaps == 1
    assert stitched_continuity(route, breaks, arcs) == dict(
        continuity=1.0, transitions=0, disconnected_jumps=0)


def test_unmatched_observation_starts_new_stitched_segment():
    arcs = np.array([[0, 1], [1, 2]])
    nodes = np.array([[0., 0.], [10., 0.], [20., 0.]])
    starts, ends = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    route, breaks, inserted, gaps = stitch_route(
        [0, -1, 1], [0, 1, 2], arcs, np.array([[5., 0.], [10., 0.], [15., 0.]]),
        starts, ends)
    assert route == [0, 1]
    assert breaks == [1] and inserted == 0 and gaps == 0
