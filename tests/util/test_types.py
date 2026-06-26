from contextlib import contextmanager

import numpy as np
import pytest
from numpy.testing import assert_, assert_equal, assert_raises

from deeptime.util.data import TrajectoryDataset
from deeptime.util.types import to_dataset, ensure_reweighting_factors_tuple, atleast_nd, ensure_array


@pytest.mark.parametrize("ary,ndim,pos,expected_shape", [
    ([1, 2, 3], 4, 0, (1, 1, 1, 3)),
    (np.zeros((2, 3)), 4, -1, (2, 3, 1, 1)),
    (np.zeros((2, 3)), 4, 0, (1, 1, 2, 3)),
    (5, 2, 0, (1, 1)),
    (np.zeros((2, 3)), 1, 0, (2, 3)),  # no-op when already >= ndim
])
def test_atleast_nd(ary, ndim, pos, expected_shape):
    # regression: atleast_nd used np.normalize_axis_index (removed from np namespace)
    # and np.array(..., copy=False) which rejects list input on numpy>=2.
    out = atleast_nd(ary, ndim=ndim, pos=pos)
    assert_equal(out.shape, expected_shape)


def test_atleast_nd_invalid_pos():
    with assert_raises(np.exceptions.AxisError):
        atleast_nd(np.zeros((2, 3)), ndim=4, pos=5)


def test_ensure_array_promotes_dimension():
    # ensure_array routes through atleast_nd when ndim must be increased
    out = ensure_array([1, 2, 3], ndim=2)
    assert_equal(out.shape, (1, 3))


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize("data,lagtime,expectation", [
    (np.zeros((100, 5)), 5, does_not_raise()),
    (np.zeros((100, 5)), None, assert_raises(ValueError)),
    (np.zeros((100, 5)), 0, assert_raises(AssertionError)),
    (np.zeros((100, 5)), 96, assert_raises(AssertionError)),
    ((np.zeros((100, 5)), np.zeros((100, 5))), None, does_not_raise()),
    ((np.zeros((100, 5)), np.zeros((105, 5))), None, assert_raises(AssertionError)),
    (TrajectoryDataset.from_trajectories(5, [np.zeros((55, 5)), np.zeros((55, 5))]), None, does_not_raise())
], ids=[
    "Trajectory with lagtime",
    "Trajectory without lagtime",
    "Trajectory with zero lagtime",
    "Trajectory with too large lagtime",
    "X-Y tuple of data",
    "X-Y tuple of data, length mismatch",
    "Custom concat dataset of list of trajectories",
])
def test_to_dataset(data, lagtime, expectation):
    with expectation:
        ds = to_dataset(data, lagtime=lagtime)
        assert_(len(ds) in (100, 95))
        data = ds[:]
        assert_equal(len(data), 2)
        assert_equal(len(data[0]), len(ds))
        assert_equal(data[0].shape[1], 5)
        assert_equal(len(data[1]), len(ds))
        assert_equal(data[1].shape[1], 5)

testdata = [
    (([np.ones((6,))],[np.ones((6,))]),([np.ones((6,))],[np.ones((6,))])),
    ((np.ones((6,)),np.ones((6,))),([np.ones((6,))],[np.ones((6,))])),
    ([[np.ones((6,))],[np.ones((6,))]],([np.ones((6,))],[np.ones((6,))])), 
    ([np.ones((6,)),np.ones((6,))],([np.ones((6,))],[np.ones((6,))])),
    (([np.ones((6,)),np.ones((6,)),np.ones((6,))],[np.ones((6,)),np.ones((6,)),np.ones((6,))]),([np.ones((6,)),np.ones((6,)),np.ones((6,))],[np.ones((6,)),np.ones((6,)),np.ones((6,))])),
    ([[np.ones((6,)),np.ones((6,))],[np.ones((6,)),np.ones((6,))]],([np.ones((6,)),np.ones((6,))],[np.ones((6,)),np.ones((6,))]))
]
@pytest.mark.parametrize("reweighting_factors, output", testdata)
def test_ensure_reweighting_factors_tuple(reweighting_factors, output):
    assert_equal(ensure_reweighting_factors_tuple(reweighting_factors), output)
