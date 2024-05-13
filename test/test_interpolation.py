# %%
"""
Test tabular interpolation
"""

import h5py
import numpy as np

from thermontfa import TabularInterpolation


def test_tab_inter_init():
    theta = np.linspace(0, 1, 10)
    val = np.linspace(2, 5, 10).reshape((-1, 1))

    tab_inter = TabularInterpolation(temps=theta, data=val)
    assert tab_inter.temp_min == theta.min()
    assert tab_inter.temp_max == theta.max()
    assert np.allclose(tab_inter.temps.ravel(), theta.ravel())
    assert np.allclose(tab_inter.data.ravel(), val.ravel())


def test_tab_inter_from_h5():
    theta = np.arange(5)
    val = theta**2
    idx = np.random.permutation(np.arange(5))
    theta_perm = theta[idx]
    val_perm = val[idx]
    val_perm = val_perm.reshape((-1, 1))
    with h5py.File("test.h5", "w") as file:
        file.create_dataset("/theta", data=theta_perm)
        file.create_dataset("/data", data=val_perm)

    tab_inter = TabularInterpolation.from_h5(
        file_name="test.h5", dset_temps="/theta", dset_data="/data"
    )
    assert tab_inter.temp_min == theta.min()
    assert tab_inter.temp_max == theta.max()
    assert np.allclose(tab_inter.temps.ravel(), theta.ravel())
    assert np.allclose(tab_inter.data.ravel(), val.ravel())
