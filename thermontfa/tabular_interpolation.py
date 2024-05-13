#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tabular interpolation for the NTFA

(c) 2024,
Felix Fritzen <fritzen@simtech.uni-stuttgart.de>,
Julius Herb <julius.herb@mib.uni-stuttgart.de>,
Shadi Sharba <shadi.sharba@isc.fraunhofer.de>

University of Stuttgart, Institute of Applied Mechanics, Chair for Data Analytics in Engineering

**Funding acknowledgment**
The IGF-Project no.: 21.079 N / DVS-No.: 06.3341 of the “Forschungsvereinigung Schweißen und verwandte Verfahren e.V.”
of the German Welding Society (DVS), Aachener Str. 172, 40223 Düsseldorf, Germany, was funded by the Federal Ministry
for Economic Affairs and Climate Action (BMWK) via the German Federation of Industrial Research Associations (AiF)
in accordance with the policy to support the Industrial Collective Research (IGF) on the orders of the German Bundestag.
Felix Fritzen is funded by the German Research Foundation (DFG) --
390740016 (EXC-2075); 406068690 (FR2702/8-1); 517847245 (FR2702/10-1).
"""

from typing import Optional, Tuple, Self

import h5py
import numpy as np


class TabularInterpolation:
    """
    Tabular interpolation for the thermo-mechanical NTFA

    Performs a linear interpolation of the NTFA system matrices for a given temperature
    given tabular data for sufficiently many temperature points.
    It can be initialized for given data or based on a HDF5 file.
    """

    temp_min: float = 0.0
    temp_max: float = 0.0
    dim: Tuple[int, ...] = ()
    temps: np.ndarray = np.array([])
    data: np.ndarray = np.array([])
    const_extrapolate: bool = False

    def __init__(
        self, temps: np.ndarray = None, data: np.ndarray = None, const_extrapolate: bool = False
    ) -> None:
        """
        Initialize the tabular interpolator for given `data` at prescribed temperatures `temps`.

        :param temps: temperature points on which tabular data is available.
            The shape of the `numpy` array is expected to be `(N_t,)`.
        :type temps: np.ndarray
        :param data: tabular data, e.g., a batch of NTFA system matrices with shape `(N_t, ...)`.
        :type data: np.ndarray
        :param const_extrapolate: If true, a constant extrapolation instead of a linear extrapolation is performed.
            The default value is false.
        :type const: bool
        """
        if temps is None or data is None:
            raise ValueError("The arguments `temps` and `data` are required.")

        idx = np.argsort(temps)

        self.data = data[idx, ...]
        self.temps = temps[idx]
        self.temp_min = self.temps[0]
        self.temp_max = self.temps[-1]
        self.dim = data[0].shape
        self.const_extrapolate = const_extrapolate

        n = self.temps.shape[0]

        if data.ndim == 1:
            self.data = self.data.reshape((-1, 1))
        assert (
            self.temps.size == n
        ), "Parameter must be an array of scalars (i.e. shape=[n] or [n, 1] or [n, 1, 1] or ...)"
        assert (
            self.data.shape[0] == n
        ), f"Number of scalar parameters not matching dimension of available data ({n} vs. {self.data.shape[0]}."

    @classmethod
    def from_h5(
        cls,
        file_name: str,
        dset_temps: str,
        dset_data: str,
        transpose_dims: Optional[Tuple[int, ...]] = None,
        const_extrapolate: bool = False,
    ) -> Self:
        """
        Initialize the tabular interpolator based on tabular data stored in a HDF5 file (*.h5).

        This is a factory method and returns a new instance of the `TabularInterpolation` class.
        It is expected that the HDF5 file contains a data set with path `dset_temps` that contains
        a list of the temperature points on which tabular data is available. The shape of this
        dataset is expected to be `(N_t,)`.
        Additionaly, the HDF5 file must contain a data set with path `dset_data` that contains the
        tabular data, e.g., a batch of NTFA system matrices with shape `(N_t, ...)`.
        The order of axes/dimensions of the data set with path `dset_data` can be changed by transposing
        to the axis order given in `transpose_dims`.

        :param file_name: path of the HDF5 file
        :type file_name: str
        :param dset_param: path to the desired dataset in the HDF5 file
        :type dset_param: str
        :param dset_data:
        :type dset_param: str
        :param const_extrapolate: "linear" or "constant"
        :type extrapolate: bool
        :param transpose_dims: axis order for transposition
        :type transpose_dims: Tuple[int, ...], optional
        :return: new instance  of the `TabularInterpolation` class
        :rtype: TabularInterpolation
        """
        with h5py.File(file_name, "r") as file:
            temps = np.array(file[dset_temps])

            if transpose_dims is None:
                data = np.array(file[dset_data])
            else:
                data = np.array(file[dset_data]).transpose(transpose_dims)
        
        return cls(temps=temps, data=data, const_extrapolate=const_extrapolate)

    def interpolate(self, temp: float) -> np.ndarray:
        """
        Perform a linear interpolation based on the available tabular data at a given temperature `temp`

        :param temp: temperature point for interpolation
        :type temp: float
        :return: interpolated quantity
        :rtype: np.ndarray
        """
        if temp < self.temp_min:
            if self.const_extrapolate:
                return self.data[0, ...]
            else:
                # linear extrapolation
                alpha = (self.temp_min - temp) / (self.temp[1] - self.temp[0])
                return self.data[0, ...] - alpha * (self.data[1, ...] - self.data[0, ...])

        if temp >= self.temp_max:
            if self.const_extrapolate:
                return self.data[-1, ...]
            else:
                # linear extrapolation
                alpha = (temp - self.temp_max) / (self.temp[-1] - self.temp[-2])
                return self.data[-1, ...] + alpha * (self.data[-1, ...] - self.data[-2, ...])

        idx = np.searchsorted(self.temp > temp, 1) - 1
        t1 = self.t[idx]
        t2 = self.t[idx + 1]
        alpha = (temp - t1) / (t2 - t1)
        interp_data = self.data[idx, ...] + alpha * (self.data[idx + 1, ...] - self.data[idx, ...])
        return interp_data
