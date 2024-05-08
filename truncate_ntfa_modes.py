#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:26:08 2023

@author: fritzen
"""

import numpy as np
import h5py
from material_parameters import *

# fname = "modes/simple_3d_rve_B1-6_4x4x4_10samples.h5"
# fname = "new/simple_3d_rve_B1-B6_8x8x8_10samples.h5"
# fname = "new/simple_3d_rve_4x4x4_2samples_new.h5"
# fname = "new/simple_3d_rve_B1-B6_8x8x8_10samples.h5"
fname = "data/simple_3d_rve_B1-B6_16x16x16_10samples_fix.h5"
basename = "/ms_9p/dset0_ntfa/"

F = h5py.File(fname, "r")
A_bar = np.array(F[basename+"A_bar"])
A0 = np.array(F[basename+"A0"])
A1 = np.array(F[basename+"A1"])
C0 = np.array(F[basename+"C0"])
C1 = np.array(F[basename+"C1"])
C_bar = np.array(F[basename+"C_bar"])
D_theta = np.array(F[basename+"D_theta"])
D_xi = np.array(F[basename+"D_xi"])
tau_theta = np.array(F[basename+"tau_theta"])
tau_xi = np.array(F[basename+"tau_xi"])
temperatures = np.array(F[basename+"temperatures"])
vol_frac = np.array(F["ms_9p/dset0_ntfa"].attrs["combo_volume_fraction"])

# vol_frac = np.zeros(2)
# vol_frac[1] = np.mean(F["ms_1p/dset0_sim/mat_id"])
# vol_frac[0] = 1. - vol_frac[1]
# print(vol_frac)
mu = np.array( F["/ms_9p/dset0_sim/plastic_modes"])
mu_eff = np.linalg.norm(mu, axis=2).sum(axis=0)
# print(mu_eff/vol_frac[0])

# print(np.mean(mu[:,:12,:], axis=0))

F.close()


Ntemp = temperatures.size

Nmax = 24
# truncate and change order of the data
# print(A_bar.shape)
# phase average stress localization
A0 = np.transpose( A0[:,:Nmax+7,:], axes=[2,0,1] )
A1 = np.transpose( A1[:,:Nmax+7,:], axes=[2,0,1] )

sig_ph = np.zeros( (2,) + A0.shape )
sig_ph[0] = A0
sig_ph[1] = A1

# phase stiffness tensors at the different temperatures
C0 = np.transpose( C0, axes=[2,0,1] )
C1 = np.transpose( C1, axes=[2,0,1] )

C_ph = np.zeros( (2,) + C0.shape )
C_ph[0] = C0
C_ph[1] = C1

eps_th_ph = np.zeros( (2, Ntemp, 6) )
eps_th_ph[0,:,:] = eps_th_cu(temperatures)[:,None] * I2[None,:]
eps_th_ph[1,:,:] = eps_th_wsc(temperatures)[:,None] * I2[None,:]


A_bar = -np.transpose( A_bar[:,:Nmax,:], axes=[2,1,0] )
# print(A_bar.shape)
D_xi = -np.transpose( D_xi[:Nmax,:Nmax,:], axes=[2,0,1] )
# print(np.linalg.eigvals(D_xi[0,:3,:3]))
tau_xi = -np.transpose(tau_xi[:Nmax, :], axes=[1,0])
tau_theta = np.transpose(tau_theta)
C_bar = np.transpose(C_bar, axes=[2,0,1])
A = A_bar[0]
XX = A.T@np.linalg.pinv(A@A.T)@A

# print(XX)

# print(( vol_frac[1]*A1[0,:,:7]+vol_frac[0]*A0[0,:,:7]) )
x  = C_bar[0,:,:].flatten()
x0 = A0[0,:,:6].flatten()
x1 = A1[0,:,:6].flatten()

alpha = np.dot( x-x1, x0-x1 )/np.dot( x0-x1, x0-x1 )
# print(alpha, vol_frac)
alpha = 37/64

# print("c_Bar:")
# print(C_bar[0,:,:])

xx = alpha*x0+(1-alpha)*x1
# print(xx.reshape((6,6))-C_bar[0,:,:])

# print('error:', xx/np.linalg.norm(x))

# print((C_bar[0,:,:]), tau_theta[0,:])
fname = f"data_new/ms9p_fix_ntfa16_B1-6_10s_N{Nmax}.h5"
F = h5py.File(fname, "w")
F.create_dataset("SIG_phases", data=sig_ph)
F.create_dataset("C_phases", data=C_ph)
F.create_dataset("eps_th_phases", data=eps_th_ph)
F.create_dataset("A_bar", data=A_bar)
F.create_dataset("v_frac", data=vol_frac)
F.create_dataset("C_bar", data=C_bar)
F.create_dataset("D_xi", data=D_xi)
F.create_dataset("D_theta", data=D_theta)
F.create_dataset("tau_xi", data=tau_xi)
F.create_dataset("tau_theta", data=tau_theta)
F.create_dataset("temperatures", data=temperatures)
F.close()

# eps_th_500 = np.linalg.solve(C_bar[20,:,:], tau_theta[20,:])
# print(eps_th_500)

# print(tau_theta)