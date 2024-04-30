#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:26:08 2023

@author: fritzen
"""

import numpy as np
import h5py

min_temperature = 293.00
max_temperature = 1300.00

I2 = np.asarray([1., 1., 1., 0, 0, 0])
I4 = np.eye(6)
IxI = np.outer(I2, I2)
P1 = IxI / 3.0
P2 = I4 - P1


poisson_ratio_cu = lambda x: 3.40000e-01 * x ** 0
conductivity_cu = lambda x: 4.20749e+05 * x ** 0 + -6.84915e+01 * x ** 1
heat_capacity_cu = lambda x: 2.94929e+03 * x ** 0 + 2.30217e+00 * x ** 1 + -2.95302e-03 * x ** 2 + 1.47057e-06 * x ** 3
cte_cu = lambda x: 1.28170e-05 * x ** 0 + 8.23091e-09 * x ** 1
eps_th_cu = lambda x: 1.28170e-05 * (x-min_temperature) + 8.23091e-09 * 0.5 * ( x*x - min_temperature**2 )
elastic_modulus_cu = lambda x: 1.35742e+08 * x ** 0 + 5.85757e+03 * x ** 1 + -8.16134e+01 * x ** 2
hardening_cu = lambda x: 20e+06 * x ** 0

shear_modulus_cu = lambda x: elastic_modulus_cu(x) / (2. * (1. + poisson_ratio_cu(x)))
bulk_modulus_cu = lambda x: elastic_modulus_cu(x) / (3. * (1. - 2. * poisson_ratio_cu(x)))
stiffness_cu = lambda x: bulk_modulus_cu(x) * IxI + 2. * shear_modulus_cu(x) * P2


poisson_ratio_wsc = lambda x: 2.80000e-01 * x ** 0
conductivity_wsc = lambda x: 2.19308e+05 * x ** 0 + -1.87425e+02 * x ** 1 + 1.05157e-01 * x ** 2 + -2.01180e-05 * x ** 3
heat_capacity_wsc = lambda x: 2.39247e+03 * x ** 0 + 6.62775e-01 * x ** 1 + -2.80323e-04 * x ** 2 + 6.39511e-08 * x ** 3
cte_wsc = lambda x: 5.07893e-06 * x ** 0 + 5.67524e-10 * x ** 1
eps_th_wsc = lambda x: 5.07893e-06 * (x-min_temperature) + 5.67524e-10 * 0.5 * ( x*x - min_temperature**2 )
elastic_modulus_wsc = lambda x: 4.13295e+08 * x ** 0 + -7.83159e+03 * x ** 1 + -3.65909e+01 * x ** 2 + 5.48782e-03 * x ** 3


shear_modulus_wsc = lambda x: elastic_modulus_wsc(x) / (2. * (1. + poisson_ratio_wsc(x)))
bulk_modulus_wsc = lambda x: elastic_modulus_wsc(x) / (3. * (1. - 2. * poisson_ratio_wsc(x)))
stiffness_wsc = lambda x: bulk_modulus_wsc(x) * IxI + 2. * shear_modulus_wsc(x) * P2

# fname = "modes/simple_3d_rve_B1-6_4x4x4_10samples.h5"
# fname = "new/simple_3d_rve_B1-B6_8x8x8_10samples.h5"
# fname = "new/simple_3d_rve_4x4x4_2samples_new.h5"
# fname = "new/simple_3d_rve_B1-B6_8x8x8_10samples.h5"
fname = "fix/simple_3d_rve_B1-B6_16x16x16_10samples_fix.h5"
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
fname = f"modes/ms9p_fix_ntfa16_B1-6_10s_N{Nmax}.h5"
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