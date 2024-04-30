#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 22:48:42 2023

@author: fritzen
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from tabular_interpolation import TabularInterpolation


class ThermoMechNTFA:
    def __init__(self, fname: str, grpname: str, sigf, Nmax: int = None):
        """ Initialize the th.-mech.-NTFA from an H5 file

        Seek the data in H5-file named fname within the gorup grpname.
        The following datasets are expected:

        """

        self.A_bar = TabularInterpolation()
        self.A_bar.InitH5(fname, grpname + "/temperatures", grpname + "/A_bar")
        self.C_bar = TabularInterpolation()
        self.C_bar.InitH5(fname, grpname + "/temperatures", grpname + "/C_bar")
        self.D_xi = TabularInterpolation()
        self.D_xi.InitH5(fname, grpname + "/temperatures", grpname + "/D_xi")
        self.tau_th = TabularInterpolation()
        self.tau_th.InitH5(fname, grpname + "/temperatures",
                           grpname + "/tau_theta")
        self.tau_xi = TabularInterpolation()
        self.tau_xi.InitH5(fname, grpname + "/temperatures",
                           grpname + "/tau_xi")
        with h5py.File(fname, "r") as F:
            self.v_frac = np.array(F[grpname+"/v_frac"])
            sig_phases_data = np.array(F[grpname+"/SIG_phases"])

        N_input = self.A_bar.data.shape[1]
        if (Nmax is not None):
            # truncate modes if needed
            assert (Nmax <= N_input), \
                f"N_modes on input ({N_input}) is smaller " + \
                "than the provided Nmax ({Nmax})"
            self.A_bar.data = self.A_bar.data[:, :Nmax, :]
            self.D_xi.data = self.D_xi.data[:, :Nmax, :Nmax]
            self.tau_xi.data = self.tau_xi.data[:, :Nmax]
            sig_phases_data = sig_phases_data[:, :, :, :(Nmax+7)]

        self.sig_phases = []
        for d in sig_phases_data:
            self.sig_phases.append(TabularInterpolation())
            self.sig_phases[-1].Initialize(self.A_bar.t, d)
        self.sigf = sigf
        self.theta_i = 0.  # last interpolation temperature
        self.C = self.C_bar.Interpolate(self.theta_i)
        self.A = self.A_bar.Interpolate(self.theta_i)
        self.D = self.D_xi.Interpolate(self.theta_i)
        self.s_th = self.tau_th.Interpolate(self.theta_i)
        self.t_xi = self.tau_xi.Interpolate(self.theta_i)
        self.n_modes = self.D_xi.data.shape[-1]

    def Interpolate(self, theta):
        TOL = 1e-4
        if (np.abs(theta-self.theta_i) > TOL):
            self.theta_i = theta
            self.C = self.C_bar.Interpolate(self.theta_i)
            self.A = self.A_bar.Interpolate(self.theta_i)
            self.D = self.D_xi.Interpolate(self.theta_i)
            self.s_th = self.tau_th.Interpolate(self.theta_i)
            self.t_xi = self.tau_xi.Interpolate(self.theta_i)

    def stress(self, eps, theta, xi, i_phase: int = None):
        self.Interpolate(theta)
        if (i_phase is None):
            return self.C @ eps + self.s_th \
                - self.A.T @ xi
        else:
            zeta = np.hstack((eps, 1, xi))
            return self.sig_phases[i_phase].Interpolate(theta)@zeta

    def solve(self, eps, deps, theta, q_n, xi_n):
        self.Interpolate(theta)
        Di = np.linalg.inv(self.D)
        tau_tr = self.A @ eps + self.t_xi + self.D@xi_n

        s23 = np.sqrt(2./3.)
        c_p = self.v_frac[0]

        phi = np.linalg.norm(tau_tr) - s23 * c_p * self.sigf(theta, q_n)
        if (phi > 0.):
            # elasto-plastic step
            n = self.n_modes
            tau = tau_tr
            dxi = np.zeros(n)  # Di@(tau - self.A@eps - self. t_xi) - xi_n
            dlambda = np.linalg.norm(dxi)
            # solve the system
            # F_1 = 0 = xi_n + dlambda tau/|tau| - D^-1 ( tau-A*eps-t_xi )
            # F_2 = 0 = |tau| - sqrt(2/3)*c_p*sigf
            # precompute some quantities
            F1_0 = xi_n + Di@(self.A@eps + self.t_xi)
            sc = False
            n_it = 0
            n_it_max = 25
            F = np.zeros(n+1)
            q = q_n + s23*dlambda
            TOL = 1e-5
            dF = np.zeros((n+1, n+1))
            while (n_it < n_it_max and not sc):
                n_it += 1
                tau_eq = np.linalg.norm(tau)
                normal = tau / tau_eq
                sf, dsf = self.sigf(theta, q, derivative=True)
                # print(f'sf: {sf/1000}')
                F[:n] = F1_0 + dlambda * normal - Di@tau
                F[-1] = tau_eq - s23*c_p*sf
                dF[:n, -1] = normal
                dF[-1, :n] = normal
                dF[-1, -1] = -2./3.*c_p*dsf
                dF[:n, :n] = -Di - (dlambda / tau_eq) \
                    * normal[:, None]*normal[None, :]
                dF[range(n), range(n)] += dlambda/tau_eq
                res = np.linalg.norm(F[:n]) + np.abs(F[-1])/sf
                sc = (res < TOL)
                if (not sc):
                    dx = -np.linalg.solve(dF, F)
                    dlambda += dx[-1]
                    tau += dx[:-1]
                    q = q_n + s23*dlambda
                # print(
                #     f'it {n_it:5d} res {res:12.2e} phi {F[-1]/1000:12.2e} MPa ;  dlambda {dlambda:10.6f}')
            xi = xi_n + dlambda*normal
            q = q_n + s23*dlambda
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

            J = np.zeros((2*n+1, 2*n+1))
            J[:n, :n] = - self.D
            J[n:(2*n), :n] = np.eye(n)
            J[:n, n:(2*n)] = np.eye(n)
            J[n:(2*n), n:(2*n)] = -dlambda/tau_eq * \
                (np.eye(n) - normal[:, None]*normal[None, :])
            J[2*n, 2*n] = -2/3*c_p*dsf
            J[-1, n:(2*n)] = normal
            J[n:(2*n), -1] = normal
            dxi_deps = np.linalg.inv(J)[:n, :n] @ self.A
            C = self.C - self.A.T @ dxi_deps
            C = 0.5*(C + C.T)
        else:
            xi = xi_n
            q = q_n
            C = self.C
            # self.dxi_deps = np.zeros((self.n_modes, 6))
            S = self.stress(eps, theta, xi_n)
        return S, q, xi, C

    def UMAT_mixed(self, eps_idx, eps_n, deps, sig_bc, theta, q_n, xi_n):
        """
        Run the UMAT using partial eps-BC, e.g., uniaxial tension test.

        Parameters
        ----------
        eps_idx : nd-array, dtype=int
            Indices of eps that are prescribed. If None, then all
            components of sigma are prescribed.
        eps_n : nd-array, dtype=float, shape=[6]
            Strain at the beginning of the increment.
        deps : nd-array, dtype=float, shape=[6]
            Strain increment. Only deps[eps_idx] is used.
        sig_bc : nd-array, dtype=float, shape=[6]
            Stress at the end of the increment. Only non eps_idx
            entires are used.
        theta : float
            Temperature at the end of the time increment.
        q_n : float
            Hardening variable at the beginning of the time increment.
        xi_n : TYPE
            Reduced coefficients at the beginning of the time increment.
        Returns
        -------
        E : nd-array, dtype=float, shape=[6]
            Full strain tensor at the end of the increment, i.e.
            the entries not within eps_idx are set
        S : nd-array, dtype=float, shape=[6]
            Stress at the end of the increment. Only S[eps_idx] is non-zero.
        C : nd-array, dtype=float, shape=[6, 6]
            Stiffness tensor at the end of the increment
        q : float
            Hardening variable at the end of the time increment
        xi : nd-array, dtype=float
            Reduced coefficients at the end of the time increment
        """
        q = q_n
        # partial strain update:
        sig_idx = np.setdiff1d(np.arange(6), eps_idx)

        eps = eps_n.copy()
        eps[eps_idx] += deps[eps_idx]
        deps[sig_idx] = 0.

        TOL_sig = 1e-7*self.sigf(theta, q_n, derivative=False)
        err_sig = 1000.*TOL_sig
        it = 0
        it_max = 100

        # eleminate thermo-elastic effects and approximate deps
        sig = self.stress(eps_n, theta, xi_n)
        # initial guess for eps asserting elastic behaviour
        C = self.C
        deps[sig_idx] += np.linalg.solve(C[sig_idx][:, sig_idx],
                                         - (sig[sig_idx]-sig_bc[sig_idx]) - C[sig_idx, :]@deps)
        # compute the actual stress state
        eps = eps_n + deps
        sig, q, xi, C = self.solve(eps, deps, theta, q_n, xi_n)
        err_sig = np.linalg.norm(sig[sig_idx])
        while ((err_sig > TOL_sig) and (it < it_max)):
            ddeps = - np.linalg.solve(C[sig_idx][:, sig_idx], sig[sig_idx]-sig_bc[sig_idx])
            factor = 1.
            deps[sig_idx] += factor * ddeps
            eps = eps_n + deps
            # err_eps = np.linalg.norm(ddeps)
            sig, q, xi, C = self.solve(eps, deps, theta, q_n, xi_n)
            err_sig = np.linalg.norm(sig[sig_idx])
            it = it + 1
            # print(
            #     f'it {it:3d} .. err_eps {err_eps:10.3e} (TOL: {TOL_eps:8.3e}) .. err_sig {err_sig:10.3e} (TOL: {TOL_sig:8.3e}) ')
        eps = eps_n + deps
        # print(f'needed {it} iterations...')
        return eps, sig, C, q, xi


def my_sigf(theta, q, derivative=False):
    if (theta < 966.06):
        sig0 = 1.12133e+02 * theta + 3.49810e+04 \
            + 1.53393e+05 * np.tanh((635.754-theta) / 206.958)
    else:
        sig0 = 2023.286
    H = 5e5  # equivalent to 500 MPa
    if (derivative is False):
        return sig0 + H*q
    else:
        return sig0 + H*q, H


# def Demo():
ntfa_material = ThermoMechNTFA(
    'ms9p_fix_ntfa16_B1-6_10s_N24.h5', '/', sigf=my_sigf, Nmax=24)

# basis = [np.sqrt(2/3) * np.array([1., -.5, -.5, 0, 0, 0]),
#          np.sqrt(0.5)*np.array([0., 1., -1., 0, 0, 0]),
#          np.array([0, 0, 0, 1, 0, 0]),
#          np.array([0, 0, 0, 0, 1, 0]),
#          np.array([0, 0, 0, 0, 0, 1]),
#          np.array([1, 1, 1, 0, 0, 0])]

# i_load = 1
# i_theta = 0

# with h5py.File("all_results_ms9p_16x16x16_10s_N24.h5", "r") as F:
#     E = np.array(F["/eps"][i_theta][i_load])
#     fans_S = np.array(F['fans/sig'][i_theta][i_load]) / 1000.
#     fans_S0 = np.array(F['fans/sig0'][i_theta][i_load]) / 1000.
#     fans_S1 = np.array(F['fans/sig1'][i_theta][i_load]) / 1000.
#     test_S = np.array(F['ntfa/sig'][i_theta][i_load]) / 1000.
#     theta = F['/temperature'][i_theta]

# # amp = np.linspace(0, 0.02, 11)
# # E = amp[:, None] * \
# #     basis[i_load]
# # print(E)
# T = np.ones(E.shape[0])*theta
# q = 0.
# xi = np.zeros(ntfa_material.n_modes)
# S = np.zeros(6)
# Spred = np.zeros(6)
# C = np.zeros((6, 6))
# deps = np.zeros(6)
# ntfa_S = np.zeros_like(E)
# for i in range(E.shape[0]):
#     eps = E[i]
#     if (i > 0):
#         deps = E[i]-E[i-1]
#     else:
#         deps = np.zeros(6)
#     theta = T[i]
#     Spred = (S + C@deps)
#     xi_n = xi
#     S, q, xi, C = ntfa_material.solve(eps, deps, theta, q, xi)
#     # recast into MPa (instead of kPa)
#     ntfa_S[i, :] = S/1000.
#     pred_err = np.linalg.norm(Spred-S)/np.linalg.norm(S)
#     print(f'step {i+1:3d} ... S: {S/1000}  (lin. error: {pred_err*100:5.1f}%)')


# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# amp = np.linalg.norm(E, axis=1)
# ax.plot(amp, np.linalg.norm(fans_S - ntfa_S, axis=1) /
#         np.linalg.norm(fans_S, axis=1))
# ax.plot(amp, np.linalg.norm(test_S - ntfa_S, axis=1) /
#         np.linalg.norm(test_S, axis=1))


# with np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
#     formatter = dict( float = lambda x: "%.3g" % x )):

theta_list =  np.linspace(300, 1300, 5)
eps_list = []
sig_list = []
sig0_list = []
sig1_list = []
q_list = []
xi_list = []
with h5py.File("ms9p_uniaxial_stress_data_mod.h5", "w") as F:
    for iload in range(6):
        G = F.create_group(f"loading{iload:1d}")
        # part 1: theta ramp up in 5 steps
        # part 2: add on top a uniaxial loading
        for theta in theta_list:
            eps_idx = np.array([iload,])
            sig_idx = np.setdiff1d(np.arange(6), eps_idx)
            eps = np.zeros(6)
            eps[eps_idx] = 1.
            eps = np.linspace(0, 0.02, 11)[:, None] * eps
            ntfa_material.Interpolate(theta)
            eps_th_el =  - np.linalg.solve(ntfa_material.C, ntfa_material.s_th )

            eps_initial, sig_initial, C_initial, q_initial, xi_initial = ntfa_material.UMAT_mixed(None, np.zeros(6), np.zeros(6), np.zeros(6), theta, 0., np.zeros(ntfa_material.n_modes))
            print("q_ini: ", q_initial, eps_initial-eps_th_el)

            # eps = eps + eps_th_el[None, :]
            eps = eps + eps_initial
            # ntfa_material.A@eps[0] + ntfa_material.t_xi,
            # print("tau_initial: ",  np.linalg.norm( ntfa_material.A@eps[0] + ntfa_material.t_xi), ";  sig_f: ", ntfa_material.sigf(theta,0.,False) * np.sqrt(2./3.) * ntfa_material.v_frac[0])
            # print("before: ", eps[0])
            xi = np.zeros(ntfa_material.n_modes)
            q = 0.
            C = np.zeros((6, 6))
            np.set_printoptions(formatter=dict(float=lambda x: "%10.3g" % x))
            sig = np.zeros_like(eps)
            sig0 = np.zeros_like(eps)
            sig1 = np.zeros_like(eps)
            xi = np.zeros(( eps.shape[0], ntfa_material.n_modes) )
            q = np.zeros(eps.shape[0])
            q_n = 0
            xi_n = np.zeros(ntfa_material.n_modes)
            for i in range(eps.shape[0]):

                if( i == 0 ):
                    deps = eps[0].copy()
                    eps_n = np.zeros(6)
                else:
                    deps = eps[i] - eps[i-1]
                    eps_n = eps[i-1]
                eps[i], sig[i], C, q[i], xi[i] = ntfa_material.UMAT_mixed(
                    eps_idx, eps_n, deps, np.zeros(6), theta, q_n, xi_n)
                xi_n = xi[i].copy()
                q_n = q[i]
                sig0[i] = ntfa_material.stress(eps[i], theta, xi[i], i_phase=0)
                sig1[i] = ntfa_material.stress(eps[i], theta, xi[i], i_phase=1)
                # if( i==0 ):
                #     eps_th_el = np.zeros(6)
                #     eps_th_el = - np.linalg.solve(ntfa_material.C, ntfa_material.s_th )
                #     # eps_th_el[sig_idx] = - np.linalg.solve(ntfa_material.C[sig_idx,:][:,sig_idx], ntfa_material.s_th[sig_idx] )
                #     print( f"theta: {theta:6.1f} -- ", (eps[i] - eps_th_el), q[i] )
                # print(f'{i:3d} - {sig[i]} - {q[i]:6.4f}')
            # print("after:  ", eps[0], q[0])
           # print(eps)
            eps_list.append(eps)
            sig_list.append(sig)
            sig0_list.append(sig0)
            sig1_list.append(sig1)
            xi_list.append(xi)
            q_list.append(q)
            # print(f"ntfa_T{theta:06.1f}")
            GG = G.create_group(f"ntfa_T{theta:06.1f}")
            GG.create_dataset("eps", data=eps)
            GG.create_dataset("sig", data=sig)
            GG.create_dataset("sig0", data=sig0)
            GG.create_dataset("sig1", data=sig1)
            GG.create_dataset("q", data=q)
            GG.create_dataset("xi", data=xi)

#%%

# run uniaxial strain controlled tests at different temperatures
# the initial state is gained by ramping up the temperature from 293 K
# subsequently, a strain controlled loading is superimposed in the iload-th
# component (2% loading).


theta_list =  np.linspace(300, 1300, 5)
eps_list = []
sig_list = []
sig0_list = []
sig1_list = []
q_list = []
xi_list = []
n_ramp = 10
with h5py.File("ms9p_uniaxial_stress_data_mod.h5", "w") as F:
    for iload in range(6):
        G = F.create_group(f"loading{iload:1d}")
        # part 1: theta ramp up in 5 steps
        # part 2: add on top a uniaxial loading
        for theta in theta_list:
            eps_idx = np.array([iload,])
            sig_idx = np.setdiff1d(np.arange(6), eps_idx)
            eps = np.zeros(6)
            eps[eps_idx] = 1.
            # the actual loading we are seeking:
            eps_bc = np.zeros((10 + n_ramp, 6))
            eps_bc[(n_ramp-1):, eps_idx] = np.linspace(0, 0.02, 11)[:, None]

            # part 1: ramp up theta from 293 K to theta
            T = np.zeros(n_ramp+10)
            theta_ramp = np.linspace(293.0, theta, n_ramp)
            T[:n_ramp] = theta_ramp
            T[n_ramp:] = theta


            # ntfa_material.A@eps[0] + ntfa_material.t_xi,
            # print("tau_initial: ",  np.linalg.norm( ntfa_material.A@eps[0] + ntfa_material.t_xi), ";  sig_f: ", ntfa_material.sigf(theta,0.,False) * np.sqrt(2./3.) * ntfa_material.v_frac[0])
            # print("before: ", eps[0])
            xi = np.zeros(ntfa_material.n_modes)
            q = 0.
            C = np.zeros((6, 6))
            np.set_printoptions(formatter=dict(float=lambda x: "%10.3g" % x))
            eps = np.zeros_like(eps_bc)
            sig = np.zeros_like(eps)
            sig0 = np.zeros_like(eps)
            sig1 = np.zeros_like(eps)
            xi = np.zeros((eps.shape[0], ntfa_material.n_modes))
            q = np.zeros(eps.shape[0])
            q_n = 0.
            xi_n = np.zeros(ntfa_material.n_modes)
            for i in range(eps.shape[0]):
                t = T[i]
                if (i == 0):
                    deps = eps_bc[0].copy()
                    eps_n = np.zeros(6)
                else:
                    deps = eps_bc[i] - eps[i-1]
                    eps_n = eps[i-1]
                if (i < n_ramp):
                    # this induces stress free loading with free strains
                    eps_idx = None
                else:
                    eps_idx = np.array([iload,])
                eps[i], sig[i], C, q[i], xi[i] = ntfa_material.UMAT_mixed(
                    eps_idx, eps_n, deps, np.zeros(6), t, q_n, xi_n)
                xi_n = xi[i].copy()
                q_n = q[i]
                sig0[i] = ntfa_material.stress(eps[i], t, xi[i], i_phase=0)
                sig1[i] = ntfa_material.stress(eps[i], t, xi[i], i_phase=1)
                # print(t, eps[i], q[i])
                if (i == n_ramp-1):
                    # update the BC!
                    eps_bc[n_ramp:, :] += eps[n_ramp-1][None, :]
                    # print(t, eps[i], q[i])
                # if( i==0 ):
                #     eps_th_el = np.zeros(6)
                #     eps_th_el = - np.linalg.solve(ntfa_material.C, ntfa_material.s_th )
                #     # eps_th_el[sig_idx] = - np.linalg.solve(ntfa_material.C[sig_idx,:][:,sig_idx], ntfa_material.s_th[sig_idx] )
                #     print( f"theta: {theta:6.1f} -- ", (eps[i] - eps_th_el), q[i] )
                # print(f'{i:3d} - {sig[i]} - {q[i]:6.4f}')
            # print("after:  ", eps[0], q[0])
            # print(eps)
            eps_list.append(eps)
            sig_list.append(sig)
            sig0_list.append(sig0)
            sig1_list.append(sig1)
            xi_list.append(xi)
            q_list.append(q)
            # print(f"ntfa_T{theta:06.1f}")
            GG = G.create_group(f"ntfa_T{theta:06.1f}")
            GG.create_dataset("T", data=T)
            GG.create_dataset("eps", data=eps)
            GG.create_dataset("sig", data=sig)
            GG.create_dataset("sig0", data=sig0)
            GG.create_dataset("sig1", data=sig1)
            GG.create_dataset("q", data=q)
            GG.create_dataset("xi", data=xi)


#%% Figuring when palsticity kicks in using the mixed UMAT
# 1. set the stress to 0
# 2. ramp the temperature from 293K
# 3. check for q >= q_crit_0, q_crit_1, ...
#    e.g. q_crit_0 = 0.002 (i.e. 0.2%)
# draw the results of theta

n_ramp = 1300-293+1
with h5py.File("ms9p_thermal_rampup.h5", "w") as F:
    eps_idx = None
    sig_idx = np.arange(6)
    eps = np.zeros(6)
    # the actual loading we are seeking:
    eps_bc = np.zeros((n_ramp, 6))

    # part 1: ramp up theta from 293 K to theta
    T = np.linspace(293.0, 1300.0, n_ramp)
    q = 0.
    C = np.zeros((6, 6))
    np.set_printoptions(formatter=dict(float=lambda x: "%10.3g" % x))
    eps = np.zeros_like(eps_bc)
    sig = np.zeros_like(eps)
    sig0 = np.zeros_like(eps)
    sig1 = np.zeros_like(eps)
    xi = np.zeros((eps.shape[0], ntfa_material.n_modes))
    q = np.zeros(eps.shape[0])
    q_n = 0.
    xi_n = np.zeros(ntfa_material.n_modes)
    for i in range(eps.shape[0]):
        t = T[i]
        if (i == 0):
            deps = eps_bc[0].copy()
            eps_n = np.zeros(6)
        else:
            deps = eps_bc[i] - eps[i-1]
            eps_n = eps[i-1]
        eps[i], sig[i], C, q[i], xi[i] = ntfa_material.UMAT_mixed(
            eps_idx, eps_n, deps, np.zeros(6), t, q_n, xi_n)
        xi_n = xi[i].copy()
        q_n = q[i]
        sig0[i] = ntfa_material.stress(eps[i], t, xi[i], i_phase=0)
        sig1[i] = ntfa_material.stress(eps[i], t, xi[i], i_phase=1)
        # print(t, eps[i], q[i])
    # print(f"ntfa_T{theta:06.1f}")
    F.create_dataset("T", data=T)
    F.create_dataset("eps", data=eps)
    F.create_dataset("sig", data=sig)
    F.create_dataset("sig0", data=sig0)
    F.create_dataset("sig1", data=sig1)
    F.create_dataset("q", data=q)
    F.create_dataset("xi", data=xi)

#%%
q_crit = [ 1e-5, 0.002, 0.005, 0.01 ]
with h5py.File("ms9p_thermal_rampup.h5", "r") as F:
    q = np.array(F["q"])
    T = np.array(F["T"])
    eps = np.array(F["eps"])
    sig0 = np.array(F["sig0"])
    sig1 = np.array(F["sig1"])
    fig, ax = plt.subplots(1,1,figsize=(12,7))
    ax.plot(T, 100*q, color='red', lw=2, label=r'NTFA $\overline{q}$')
    for qc in q_crit:
        ax.plot([T[0], T[-1]], [100*qc, 100*qc], color='black', lw=1)
        i = np.searchsorted(q, qc)
        ax.annotate(text=f'$q_c={qc*100}\%, T={T[i]}K$', xy=[T[i], 100*q[i]],
                    xytext=[T[i]-150, 100*q[i]+0.2], color='blue',
                    arrowprops = dict( width=2 ) )

ax.grid()
ax.set_xlim(T[0], T[-1])
ax.set_xlabel(r"temperature $T$ [K]")
ax.set_ylabel(r"hardening variable $\overline{q}$ [%]")

#%%



sig_fe_list = []
sig0_fe_list = []
sig1_fe_list = []

# with h5py.File("ms9p_uniaxial_stress_data_loading0-5.h5", "r") as F:
with h5py.File("ms9p_uniaxial_stress_data_mod_loading0.h5", "r") as F:
    for iload in range(1):
        G = F[f"loading{iload:1d}"]
        for theta in theta_list:
            GG = G[f"ntfa_T{theta:06.1f}"]
            sig_fe = np.array(GG["fe_sig"])
            sig0_fe = np.array(GG["fe_sig0"])
            sig1_fe = np.array(GG["fe_sig1"])
            sig_fe_list.append(sig_fe)
            sig0_fe_list.append(sig0_fe)
            sig1_fe_list.append(sig1_fe)


def rel_error(A, A_ref, r_min=None):
    if(r_min is None):
        return np.linalg.norm(A-A_ref, axis=1)/np.linalg.norm(A_ref, axis=1)
    else:
        return np.linalg.norm(A-A_ref, axis=1) / \
            (np.maximum(r_min, np.linalg.norm(A_ref, axis=1)))



n = len(sig_fe_list)

for k in [0, 1, 2, 3, 4]:
    fig, ax = plt.subplots(1, 2, figsize=(8,4.5))
    # fig.suptitle(rf"$T={theta_list[np.mod(k,5)]}$ - rel. error (overall/matrix/inclusion)")
    temp = theta_list[np.mod(k,5)]
    fig.suptitle(rf"$T={temp}$ - rel. error (matrix/inclusion)")
    c1 = ntfa_material.v_frac[0]
    c2 = ntfa_material.v_frac[1]
    sig_list[k] = c1 * sig0_list[k] + c2 * sig1_list[k]
    # delta =   c1 * sig0_list[k] + c2 * sig1_list[k] - sig_list[k]
    # print(rel_error(  c1 * sig0_fe_list[k] + c2 * sig1_fe_list[k], sig_fe_list[k] ))
    err_sig = rel_error(sig_list[k], sig_fe_list[k], r_min=1e4)
    err_sig0 = rel_error(sig0_list[k], sig0_fe_list[k], r_min=1e4)
    err_sig1 = rel_error(sig1_list[k], sig1_fe_list[k], r_min=1e4)
    t = np.linspace(0, 1, sig_list[k].shape[0])
    ax[0].plot(t[:], err_sig0[:]*100)
    ax[1].plot(t[:], err_sig1[:]*100)
    for A in ax:
        A.grid()
        A.set_xlabel('rel. time [-]')
        A.set_ylabel('rel. error [%]')
        A.set_ylim([0, 10])
    ax[0].text(.55, 1, 'plastic matrix', backgroundcolor='#004191', color='white')
    ax[1].text(.55, 1, 'elastic particles', backgroundcolor='#004191', color='white')
    fig.tight_layout()
    plt.savefig(f'rel_error_uniaxial_T{temp:.0f}.jpg')
    # ax[0].plot(t[:], err_sig[:]*100)
    # ax[1].plot(t[:], err_sig0[:]*100)
    # ax[2].plot(t[:], err_sig1[:]*100)

    # print( sig_list[k] - ntfa_material.v_frac[0] * sig0_list[k] - ntfa_material.v_frac[1] * sig1_list[k] )
    # print(f'rel. error in sig   {100*err_sig}%')
    # print(f'rel. error in sig0  {100*err_sig0}%')
    # print(f'rel. error in sig1  {100*err_sig1}%')
