import numpy as np

from api.decoders import DecodeContext
from pathsnap_ml.viterbi import decode


def test_bridge_uses_only_forward_directed_arc():
    nodes = np.array([[0., 0.], [10., 0.], [20., 0.], [30., 0.], [20., 1.], [30., 1.]])
    points = np.array([[5., 0.], [25., 0.]])
    track = np.column_stack((points, [0, 1]))
    scores = [[.9, 0], [.4, .99]]
    candidates = [[0, -1], [2, 3]]
    for bridge, expected in [([1, 2], [0, 2]), ([2, 1], [0, 3])]:
        arcs = np.array([[0, 1], bridge, [2, 3], [4, 5]])
        context = DecodeContext(track, nodes, arcs, points,
                                nodes[arcs[:, 0]], nodes[arcs[:, 1]])
        np.testing.assert_array_equal(decode(scores, candidates, context), expected)


def test_single_observation_zero_scores_and_invalid_ids():
    nodes = np.array([[0., 0.], [10., 0.]])
    arcs = np.array([[0, 1]])
    points = np.array([[5., 0.]])
    context = DecodeContext(np.column_stack((points, [0])), nodes, arcs, points,
                            nodes[:1], nodes[1:])
    np.testing.assert_array_equal(decode([[1., 0.]], [[99, 0]], context), [0])
