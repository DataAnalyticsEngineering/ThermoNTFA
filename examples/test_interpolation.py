"""
Test tabular interpolation
"""

def test_tab_inter_init():
    theta = np.arange(5)
    val = theta ** 2
    idx = np.random.permutation(np.arange(5))
    theta = theta[idx]
    val = val[idx]
    val.reshape((5, 1))
    F = h5py.File('test.h5', "w")
    F.create_dataset('/theta', data=theta)
    F.create_dataset('/data', data=val)
    F.close()

    tab_inter = TabularInterpolation()
    tab_inter.InitH5("test.h5", "/theta", "/data")
    print(tab_inter.t_min)
    print(tab_inter.t_max)
    print(tab_inter.t)
    print(tab_inter.data)


a = np.linspace(2, 5, 10).reshape((-1, 1))
t = np.linspace(0, 1, 10)
inter = TabularInterpolation(a, t)
