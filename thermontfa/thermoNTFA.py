#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermo-mechanical NTFA

Represents a material routine that describes the effective behavior of a thermo-elasto-plastic composite material
with temperature-dependent material parameters in both phases.

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

from typing import Callable, Optional, Tuple, List

import h5py
import numpy as np

from .tabular_interpolation import TabularInterpolation


class ThermoMechNTFA:
    """
    Thermo-mechanical NTFA

    Represents a material routine that describes the effective behavior of a thermo-elasto-plastic composite material
    with temperature-dependent material parameters in both phases.
    """

    # TODO: Document properties
    file_name: str
    group_name: str
    sig_y: Callable[[float, float, bool], float]
    tol: float
    verbose: bool = False
    A_bar: TabularInterpolation
    C_bar: TabularInterpolation
    D_xi: TabularInterpolation
    tau_theta: TabularInterpolation
    tau_xi: TabularInterpolation
    sig_phases: List

    def __init__(
        self,
        file_name: str,
        group_name: str,
        sig_y: Callable[[float, float, bool], float],
        N_max: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the thermo-mechanical NTFA from an HDF5 file (:code:`*.h5`)

        Seek the data in HDF5 file named :code:`file_name` within the group :code:`group_name`.
        The following data sets containing tabular data for the NTFA are expected in the group:
        - `temperatures`: list of temperature points of the tabular data, shape: :code:`(N_temp,)`
        - :code:`A_bar`: shape: :code:`(N_temp, N_modes, 6)`
        - :code:`C_bar`: shape: :code:`(N_temp, 6, 6)`
        - :code:`D_xi`: shape: :code:`(N_temp, N_modes, N_modes)`
        - :code:`tau_theta`: shape: :code:`(N_temp, 6)`
        - :code:`tau_xi`: shape: :code:`(N_temp, N_modes)`

        In addition, the group in the HDF5 file must contain the following data sets:
        - :code:`v_frac`: volume fraction of the different phases
        - :code:`SIG_phases`: stress data different phases

        :param file_name: path to the HDF5 file
        :type file_name: str
        :param group_name: group in the HDF5 file that contains the NTFA tabular data
        :type group_name: str
        :param sig_y: function/callable that returns the yield stress :code:`sig_y(theta, q_n, derivative)`
            given the temperature :code:`theta` and the current isotropic hardening variable :code:`q_n`.
            If :code:`derivative = True`, the derivative should also be returned.
        :type sig_y: Callable[[float, float, bool], float]
        :param N_max: maximum number of NTFA modes that should be used. If None, all available modes are used.
        :type N_max: int
        :param verbose: If debug information should be printed.
        :type verbose: bool
        """
        self.file_name = file_name
        self.group_name = group_name
        self.sig_y = sig_y
        self.tol = tol
        self.N_max = N_max
        self.verbose = verbose

        self.A_bar = TabularInterpolation.from_h5(
            file_name=self.file_name,
            dset_temps=self.group_name + "/temperatures",
            dset_data=self.group_name + "/A_bar",
        )

        self.C_bar = TabularInterpolation.from_h5(
            file_name=self.file_name,
            dset_temps=self.group_name + "/temperatures",
            dset_data=self.group_name + "/C_bar",
        )

        self.D_xi = TabularInterpolation.from_h5(
            file_name=self.file_name,
            dset_temps=self.group_name + "/temperatures",
            dset_data=self.group_name + "/D_xi",
        )

        self.tau_theta = TabularInterpolation.from_h5(
            file_name=self.file_name,
            dset_temps=self.group_name + "/temperatures",
            dset_data=self.group_name + "/tau_theta",
        )

        self.tau_xi = TabularInterpolation.from_h5(
            file_name=self.file_name,
            dset_temps=self.group_name + "/temperatures",
            dset_data=self.group_name + "/tau_xi",
        )

        with h5py.File(self.file_name, "r") as file:
            self.v_frac = np.array(file[self.group_name + "/v_frac"])
            sig_phases_data = np.array(file[self.group_name + "/SIG_phases"])

        N_input = self.A_bar.dim[0]
        if self.N_max is not None:
            # truncate modes if needed
            assert (
                self.N_max <= N_input
            ), f"N_modes on input ({N_input}) is smaller than the provided N_max ({self.N_max})"
            # TODO: truncate in tabular interpolation?
            self.A_bar.data = self.A_bar.data[:, : self.N_max, :]
            self.D_xi.data = self.D_xi.data[:, : self.N_max, : self.N_max]
            self.tau_xi.data = self.tau_xi.data[:, : self.N_max]
            sig_phases_data = sig_phases_data[:, :, :, : (self.N_max + 7)]

        self.sig_phases = []
        for sig_data in sig_phases_data:
            self.sig_phases.append(TabularInterpolation(self.A_bar.temps, sig_data))

        self.theta_i = 0.0  # last interpolation temperature
        self.C = self.C_bar.interpolate(self.theta_i)
        self.A = self.A_bar.interpolate(self.theta_i)
        self.D = self.D_xi.interpolate(self.theta_i)
        self.s_th = self.tau_th.interpolate(self.theta_i)
        self.t_xi = self.tau_xi.interpolate(self.theta_i)
        self.n_modes = self.D_xi.data.shape[-1]

    def interpolate(self, theta: float) -> None:
        """
        Interpolate NTFA matrices to current temperature :code:`theta` if the given tolerance is exceeded

        :param theta: Temperature
        """
        if np.abs(theta - self.theta_i) > self.tol:
            self.theta_i = theta
            self.C = self.C_bar.interpolate(self.theta_i)
            self.A = self.A_bar.interpolate(self.theta_i)
            self.D = self.D_xi.interpolate(self.theta_i)
            self.s_th = self.tau_th.interpolate(self.theta_i)
            self.t_xi = self.tau_xi.interpolate(self.theta_i)

    def stress(
        self,
        eps: np.ndarray,
        theta: float,
        xi: np.ndarray,
        i_phase: Optional[int] = None,
    ):
        """
        Compute the stress given strain :code:`eps`, plastic mode activations :code:`xi`.
        If :code:`i_phase` is given, the stress is computed only for the phase with index :code:`i_phase`.

        :param eps: Strain
        :type eps: np.ndarray
        :param theta: Temperature
        :type eps: float
        :param xi: Plastic mode activations
        :type eps: np.ndarray
        :param i_phase: Phase index for the stress computation. If :code:`None`, overall stress is computed.
        :type eps: int, optional
        :return: Computed stress
        """
        self.interpolate(theta)
        if i_phase is None:
            return self.C @ eps + self.s_th - self.A.T @ xi
        else:
            zeta = np.hstack((eps, 1, xi))
            return self.sig_phases[i_phase].interpolate(theta) @ zeta

    def UMAT_mixed(
        self,
        eps_idx: np.ndarray,
        eps_n: np.ndarray,
        deps: np.ndarray,
        sig_bc: np.ndarray,
        theta: float,
        q_n: float,
        xi_n: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Run the UMAT using partial eps-BC, e.g., uniaxial tension test.

        :param eps_idx: Indices of eps that are prescribed.
            If :code:`None`, then all components of :code:`sig_bc` are prescribed.
        :type eps_idx: nd-array, dtype=int
        :param eps_n: Strain at the beginning of the increment.
        :type eps_n: nd-array, dtype=float, shape=[6]
        :param deps: Strain increment. Only deps[eps_idx] is used.
        :type deps: nd-array, dtype=float, shape=[6]
        :param sig_bc: Stress at the end of the increment. Only non :code:`eps_idx` components are used.
        :type sig_bc: nd-array, dtype=float, shape=[6]
        :param theta: Temperature at the end of the time increment.
        :type theta: float
        :param q_n: Hardening variable at the beginning of the time increment.
        :type q_n: float
        :param xi_n: Reduced coefficients at the beginning of the time increment.
        :type xi_n: nd-array, dtype=float
        :return: **E** -
            Full strain tensor at the end of the increment, i.e.
            the entries not within eps_idx are set
        :rtype: np.ndarray, dtype=float, shape=[6]
        :return: **S** -
            Stress at the end of the increment. Only S[eps_idx] is non-zero.
        :rtype: np.ndarray, dtype=float
        :return: **C** -
            Stiffness tensor at the end of the increment
        :rtype: `np.ndarray`, `dtype=float`, `shape=[6, 6]`
        :return: **q** -
            Hardening variable at the end of the time increment
        :rtype: float
        :return: **xi** -
            Reduced coefficients at the end of the time increment
        :rtype: `nd.ndarray`, `dtype=float`
        """
        q = q_n
        # partial strain update:
        sig_idx = np.setdiff1d(np.arange(6), eps_idx)

        eps = eps_n.copy()
        eps[eps_idx] += deps[eps_idx]
        deps[sig_idx] = 0.0

        self.tol_sig = 1e-7 * self.sig_y(theta, q_n, derivative=False)
        err_sig = 1000.0 * self.tol_sig
        it = 0
        it_max = 100

        # eleminate thermo-elastic effects and approximate deps
        sig = self.stress(eps_n, theta, xi_n)

        # initial guess for eps asserting elastic behaviour
        C = self.C
        deps[sig_idx] += np.linalg.solve(
            C[sig_idx][:, sig_idx],
            -(sig[sig_idx] - sig_bc[sig_idx]) - C[sig_idx, :] @ deps,
        )

        # compute the actual stress state
        eps = eps_n + deps
        sig, q, xi, C = self.solve(eps, deps, theta, q_n, xi_n)
        err_sig = np.linalg.norm(sig[sig_idx])
        while (err_sig > self.tol_sig) and (it < it_max):
            ddeps = -np.linalg.solve(
                C[sig_idx][:, sig_idx], sig[sig_idx] - sig_bc[sig_idx]
            )
            factor = 1.0
            deps[sig_idx] += factor * ddeps
            eps = eps_n + deps
            sig, q, xi, C = self.solve(eps, deps, theta, q_n, xi_n)
            err_sig = np.linalg.norm(sig[sig_idx])
            it = it + 1

            if self.verbose:
                err_eps = np.linalg.norm(ddeps)
                print(
                    f"it {it:3d} .. err_eps {err_eps:10.3e} (self.tol: {self.tol_eps:8.3e}) "
                    + ".. err_sig {err_sig:10.3e} (self.tol: {self.tol_sig:8.3e})"
                )

        eps = eps_n + deps

        if self.verbose:
            print(f"UMAT_mixed completed after {it} iterations...")

        return eps, sig, C, q, xi

    def solve(
        self,
        eps: np.ndarray,
        deps: np.ndarray,
        theta: float,
        q_n: float,
        xi_n: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Solve for stress :code:`S`, hardening variable :code:`q`, reduced coefficients :code:`xi`,
        and stiffness :code:`C` given the strain :code:`eps`, strain increment :code:`deps`,
        temperature :code:`theta`, hardening variable :code:`q_n`, and reduced coefficients :code:`xi_n`.

        :param eps: Strain
        :type eps: np.ndarray
        :param deps: Strain increment
        :type deps: np.ndarray
        :param theta: Temperature
        :type theta: float
        :param q_n: Hardening variable at the beginning of the time increment
        :type q_n: float
        :param xi_n: Reduced coefficients at the beginning of the time increment
        :type xi_n: np.ndarray
        :return: **S** - Stress
        :rtype: np.ndarray
        :return: **q** - Hardening variable at the end of the time increment
        :rtype: float
        :return: **xi** - Reduced coefficients at the end of the time increment
        :rtype: np.ndarray
        :return: **C** - Stiffness
        :rtype: np.ndarray
        """
        self.interpolate(theta)
        Di = np.linalg.inv(self.D)
        tau_tr = self.A @ eps + self.t_xi + self.D @ xi_n

        s23 = np.sqrt(2.0 / 3.0)
        c_p = self.v_frac[0]

        phi = np.linalg.norm(tau_tr) - s23 * c_p * self.sig_y(
            theta, q_n, derivative=False
        )
        if phi > 0.0:
            # elasto-plastic step
            n = self.n_modes
            tau = tau_tr
            dxi = np.zeros(n)  # Di@(tau - self.A@eps - self. t_xi) - xi_n
            dlambda = np.linalg.norm(dxi)

            # solve the system
            # F_1 = 0 = xi_n + dlambda tau/|tau| - D^-1 ( tau-A*eps-t_xi )
            # F_2 = 0 = |tau| - sqrt(2/3)*c_p*sig_y

            # precompute some quantities
            F1_0 = xi_n + Di @ (self.A @ eps + self.t_xi)
            sc = False
            n_it = 0
            n_it_max = 25
            F = np.zeros(n + 1)
            q = q_n + s23 * dlambda
            self.tol = 1e-5
            dF = np.zeros((n + 1, n + 1))
            while n_it < n_it_max and not sc:
                n_it += 1
                tau_eq = np.linalg.norm(tau)
                normal = tau / tau_eq
                sf, dsf = self.sig_y(theta, q, derivative=True)
                if self.verbose:
                    print(f"sf: {sf/1000}")
                F[:n] = F1_0 + dlambda * normal - Di @ tau
                F[-1] = tau_eq - s23 * c_p * sf
                dF[:n, -1] = normal
                dF[-1, :n] = normal
                dF[-1, -1] = -2.0 / 3.0 * c_p * dsf
                dF[:n, :n] = (
                    -Di - (dlambda / tau_eq) * normal[:, None] * normal[None, :]
                )
                dF[range(n), range(n)] += dlambda / tau_eq
                res = np.linalg.norm(F[:n]) + np.abs(F[-1]) / sf
                sc = res < self.tol
                if not sc:
                    dx = -np.linalg.solve(dF, F)
                    dlambda += dx[-1]
                    tau += dx[:-1]
                    q = q_n + s23 * dlambda
                if self.verbose:
                    print(
                        f"it {n_it:5d} res {res:12.2e} phi {F[-1]/1000:12.2e} MPa ;  dlambda {dlambda:10.6f}"
                    )

            xi = xi_n + dlambda * normal
            q = q_n + s23 * dlambda
            S = self.stress(eps, theta, xi)
            # C = self.C - \
            #     self.A.T @ Di @ np.linalg.inv(dF)[:n, :n]@(Di@self.A)
            # M = - np.linalg.inv(dF)[:, :n] @ (Di@self.A)
            # dlambda_deps = M[-1, :]
            # dtau_deps = M[:n, :]
            # dxi_deps = Di@dtau_deps
            # # self.dxi_deps = dxi_deps
            # C = self.C - self.A.T @ (dxi_deps +
            #                          normal[:, None] * dlambda_deps[None, :])

            J = np.zeros((2 * n + 1, 2 * n + 1))
            J[:n, :n] = -self.D
            J[n : (2 * n), :n] = np.eye(n)
            J[:n, n : (2 * n)] = np.eye(n)
            J[n : (2 * n), n : (2 * n)] = (
                -dlambda / tau_eq * (np.eye(n) - normal[:, None] * normal[None, :])
            )
            J[2 * n, 2 * n] = -2 / 3 * c_p * dsf
            J[-1, n : (2 * n)] = normal
            J[n : (2 * n), -1] = normal
            dxi_deps = np.linalg.inv(J)[:n, :n] @ self.A
            C = self.C - self.A.T @ dxi_deps
            C = 0.5 * (C + C.T)
        else:
            xi = xi_n
            q = q_n
            C = self.C
            # self.dxi_deps = np.zeros((self.n_modes, 6))
            S = self.stress(eps, theta, xi_n)
        return S, q, xi, C
