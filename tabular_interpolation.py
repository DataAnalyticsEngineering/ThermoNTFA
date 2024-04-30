#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:11:37 2023

@author: fritzen
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

class TabularInterpolation:

    t_min = 0
    t_max = 0
    dim = ()
    t = np.array([])
    data = np.array([])
    const_extrapolate = False

    def __init__(self, data=None, param=None):
        if( data is not None and param is not None ):
            idx = np.argsort(param)

            self.data = data[idx,:]
            self.t = param[idx]
            self.t_min = self.t[0]
            self.t_max = self.t[-1]
            self.dim = data[0].shape


    def InitH5(self, fname, dset_param, dset_data, extrapolate = 'linear', transpose=None):
        F = h5py.File(fname, "r")
        self.t = np.array(F[dset_param])
        if( transpose is None ):
            self.data = np.array(F[dset_data])
        else:
            self.data = np.array(F[dset_data]).transpose(transpose)
        F.close()
        self.dim = self.data.shape[1:]

        if( extrapolate == "linear" ):
            self.const_extrapolate = False
        else:
            if( extrapolate == 'constant' ):
                self.const_extrapolate = True
            else:
                raise ValueError(f"extrapolate can only take values \"linear\" or \"constant\" (received: \"{extrapolate}\"")

        if( self.data.ndim == 1):
            self.data = self.data.reshape((-1,1))
        n = self.t.shape[0]
        assert( self.t.size == n), \
            "ERROR: parameter must be an array of scalars (i.e. shape=[n] or [n, 1] or [n, 1, 1] or ...)"
        assert( self.data.shape[0] == n), \
            f"ERROR: number of scalar parameters not matching dimension of the available data ({n} vs. {self.data.shape[0]}."
        idx = np.argsort(self.t)
        self.t = self.t[idx]
        self.data = self.data[idx,:]
        self.t_min = self.t[0]
        self.t_max = self.t[-1]

    def Interpolate( self, t ):
        if( t < self.t_min ):
            if self.const_extrapolate:
                return self.data[0,:]
            else:
                # linear extrapolation
                alpha = (self.t_min - t)/(self.t[1]-self.t[0])
                return self.data[0,:] - alpha*(self.data[1,:]-self.data[0,:])

        if( t >= self.t_max ):
            if self.const_extrapolate:
                return self.data[-1,:]
            else:
                # linear extrapolation
                alpha = (t - self.t_max)/(self.t[-1]-self.t[-2])
                return self.data[-1,:] + alpha*(self.data[-1,:]-self.data[-2,:])

        idx = np.searchsorted(self.t>t, 1) - 1
        t1 = self.t[idx]
        t2 = self.t[idx+1]
        alpha = (t - t1)/(t2 - t1)
        return self.data[idx,:] + alpha*(self.data[idx+1,:]-self.data[idx,:])


def test_tab_inter_init():
    theta = np.arange(5)
    val = theta**2
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

a = np.linspace(2,5,10).reshape((-1,1))
t = np.linspace(0,1,10)
inter = TabularInterpolation(a, t)