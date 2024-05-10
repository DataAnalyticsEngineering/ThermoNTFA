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

from typing import Optional, Tuple

import h5py
import numpy as np


class TabularInterpolation:
    """
    Tabular interpolation for the NTFA
    """

    t_min: float = 0.0
    t_max: float = 0.0
    dim: Tuple[int, ...] = ()
    t: np.ndarray = np.array([])
    data: np.ndarray = np.array([])
    const_extrapolate: bool = False

    def __init__(
        self, data: Optional[np.ndarray] = None, param: Optional[np.ndarray] = None
    ):
        """
        Initialize the tabular interpolator

        :param data:
        :param param:
        """
        if data is not None and param is not None:
            idx = np.argsort(param)

            self.data = data[idx, :]
            self.t = param[idx]
            self.t_min = self.t[0]
            self.t_max = self.t[-1]
            self.dim = data[0].shape

    def init_h5(
        self,
        file_name: str,
        dset_param: str,
        dset_data: str,
        extrapolate: str = "linear",
        transpose: Optional[Tuple[int, ...]] = None,
    ):
        """
        Initialize the tabular interpolator using data in a H5 file

        :param file_name: path of the H5 file
        :type file_name: str
        :param dset_param: path to the desired dataset in the H5 file
        :type dset_param: str
        :param dset_data:
        :type dset_param: str
        :param extrapolate: "linear" or "constant"
        :type extrapolate: str
        :param transpose: axis order for transposition
        :type transpose: Tuple[int, ...], optional
        """
        F = h5py.File(file_name, "r")
        self.t = np.array(F[dset_param])
        if transpose is None:
            self.data = np.array(F[dset_data])
        else:
            self.data = np.array(F[dset_data]).transpose(transpose)
        F.close()
        self.dim = self.data.shape[1:]

        if extrapolate == "linear":
            self.const_extrapolate = False
        else:
            if extrapolate == "constant":
                self.const_extrapolate = True
            else:
                raise ValueError(
                    f'extrapolate can only take values "linear" or "constant" (received: "{extrapolate}"'
                )

        if self.data.ndim == 1:
            self.data = self.data.reshape((-1, 1))
        n = self.t.shape[0]
        assert (
            self.t.size == n
        ), "ERROR: parameter must be an array of scalars (i.e. shape=[n] or [n, 1] or [n, 1, 1] or ...)"
        assert (
            self.data.shape[0] == n
        ), f"ERROR: number of scalar parameters not matching dimension of available data ({n} vs. {self.data.shape[0]}."
        idx = np.argsort(self.t)
        self.t = self.t[idx]
        self.data = self.data[idx, :]
        self.t_min = self.t[0]
        self.t_max = self.t[-1]

    def interpolate(self, t: float) -> np.ndarray:
        """
        Perform a linear interpolation at a given temperature t

        :param t: temperature point for interpolation
        :type t: float
        :return: interpolated quantity
        :rtype: np.ndarray
        """
        if t < self.t_min:
            if self.const_extrapolate:
                return self.data[0, :]
            else:
                # linear extrapolation
                alpha = (self.t_min - t) / (self.t[1] - self.t[0])
                return self.data[0, :] - alpha * (self.data[1, :] - self.data[0, :])

        if t >= self.t_max:
            if self.const_extrapolate:
                return self.data[-1, :]
            else:
                # linear extrapolation
                alpha = (t - self.t_max) / (self.t[-1] - self.t[-2])
                return self.data[-1, :] + alpha * (self.data[-1, :] - self.data[-2, :])

        idx = np.searchsorted(self.t > t, 1) - 1
        t1 = self.t[idx]
        t2 = self.t[idx + 1]
        alpha = (t - t1) / (t2 - t1)
        return self.data[idx, :] + alpha * (self.data[idx + 1, :] - self.data[idx, :])
