import pytest
from numpy.testing import *

import sktime
from sktime.data.ellipsoids_dataset import Ellipsoids


def test_laziness():
    with assert_raises(ValueError):
        sktime.data.ellipsoids(laziness=0.5)
    with assert_raises(ValueError):
        sktime.data.ellipsoids(laziness=1.01)
    assert_equal(sktime.data.ellipsoids().msm.transition_matrix.shape, (2, 2))
    assert_equal(sktime.data.ellipsoids(laziness=0.6).msm.transition_matrix[0, 0], 0.6)
    assert_equal(sktime.data.ellipsoids(laziness=0.7).msm.transition_matrix[1, 1], 0.7)


@pytest.mark.parametrize("dim", [1, 2, 5, 100], ids=lambda d: f"dim={d}")
def test_observations(dim):
    ds = sktime.data.ellipsoids()
    if dim < 2:
        with assert_raises(ValueError):
            ds.observations(n_steps=1000, n_dim=dim, noise=True)
    else:
        obs = ds.observations(n_steps=1000, n_dim=dim, noise=True)
        assert_equal(obs.shape, (1000, dim))