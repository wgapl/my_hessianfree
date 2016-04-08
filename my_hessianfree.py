#! /usr/bin/env python
"""
file: my_hessianfree.py
author: thomas wood (thomas@wgapl.com)
description: function to perform Hessian-Free optimization.
"""
import numpy as np

def hess_free(f, x0, y, max_steps=5000, tol=1e-6):
    """

    f(x, y) is a scalar valued objective function with parameters x
    and data y that returns the value of the objective function as
    well as the direction of greatest change df. Related to probabilistic
    likelihood that our parameters are x given the data y.

    """
    # Get an initial value for gradient
    f0, df0 = f(x0,y)
    # Define initial direction
    d0 = -df0
    # Need an epsilon to find directional derivative along x.
    e0 = 0.0001
    # step a distance e0 in two directions, find gradients there
    _, df_x = f(x0+ e0*x0, y)
    _, df_d = f(x0+ e0*d0, y)
    
    ##########################################################################
    ##!! NO NO NO. This is all wrong. Why not take a look at some other   !!##
    ##!! people's code and see how you can use forward automatic          !!##
    ##!! differentiation to calculate this value instead of approximating !!##
    ##!! it with finite differences!                                      !!##
    ##########################################################################
    
    # approximate directional derivative of objective function along x0 and d0
    # Hx = (df_x - df0)/e0 # product of hessian and x0
    #Hd = (df_d - df0)/e0 # product of hessian and d0
    # Calculate optimal initial step size once outside loop with bootstrap Hd
    a = - np.dot(d0.T, Hx0 + df0) / np.dot(d0.T, Hd)
    d = d0
    dprev = d
    for _ in range(max_steps):
        # Update "position"
        x = x + a * d
        v, df = f(x, y)
        _, dfx = f(x + e0*x, y)
        # Calculate new direction derivate with new x
        Hx = (dfx - df) / e0
        if abs(v) < tol:
            break
        b = np.dot(df.T, Hd)/ np.dot(d.T,Hd)
        # This variable isn't needed, but order of code gets messy w/o it.
        dprev = d
        # update new conjugate direction after calculating b
        d = -df + b*d
        # calculate new directional derivative for new direction
        Hd = (df - dprev) / a
        # Find stepsize using x_i and d_i
        a = - np.dot(d.T, Hx + df) / np.dot(d.T,Hd)

    return x
