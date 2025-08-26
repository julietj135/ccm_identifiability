#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:42:33 2025

@author: julietjiang
"""

"""
Same as the identifiability_cleaned.py script only with 
new objective function "quadratic_obj_N" and gradient/Hessian estimation.
"""

############ LOADING MODULES AND FUNCTIONS

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os
from scipy.interpolate import interp1d
import math
from tqdm import tqdm

#%%

############ DATA PRE-PROCESSING

# ADNI CSF data (for exam dates and hippocampal volume)
df0 = pd.read_excel('/Users/julietjiang/Downloads/ADProjectFiles/AIM1/ADNIMERGE.xlsx')

# ADNI PET data
df = pd.read_excel('../UCBERKELEY_PET_CLEAN_ODE.xlsx')

# spreadsheet to obtain RIDs of subjects with sufficient data
df4 = pd.read_excel("../Parameters_PET_TRAIN_ICV.xlsx")

RID = df.RID
RID = list(set(RID))

RID_train = df4.RID.tolist()

DX = df.DX_closest
DX_bl = df0.DX_bl

# get ages
age = {}
for i in range(0,len(RID)):
    AGE = df[df['RID'] == RID[i]]['AGE'].dropna().unique()
    if len(AGE)==0:
        age[RID[i]] = AGE
    else:
        age[RID[i]] = AGE[0]

age_time = {}
df0['datetimes'] = pd.to_datetime(df0.EXAMDATE, format='%m/%d/%Y', errors='coerce')
df['datetimes'] = pd.to_datetime(df.EXAMDATE, format='%m/%d/%y', errors='coerce')

# for each patient, find their exam dates and add the earliest exam date to age_time
for i in range(0,len(RID)):
    date_min = df0.loc[df0['RID'] == RID[i], 'datetimes'].min()
    age_time[RID[i]] = date_min

# collects Hippocampus and ICV columns, divides them, finds percentiles
hippocampus = df.HIPPOCAMPUS_VOLUME
ICV = df.ICV
ratio = hippocampus/ICV # normalization by ICV

r_upper = ratio.quantile(0.95)
r_lower = ratio.quantile(0.05)

ratio = (ratio - r_lower) / (r_upper - r_lower)
ratio = 1 - ratio
N = ratio

# same process but for PET ABETA
ABETA = df.SUMMARY_SUVR             
AB_upper = ABETA.quantile(0.95)
AB_lower = ABETA.quantile(0.05)
ABETA = (ABETA - AB_lower) / (AB_upper - AB_lower)

# same process for PET TAU
PTAU = df.META_TEMPORAL_SUVR   
T_upper = PTAU.quantile(0.95)
PTAU = PTAU/T_upper


#%%

def Right_hand_side_A(t, u, x):
    ABETA = u[0]
    KBETA = x[0]
    lambdaBETA = x[1]
    a = lambdaBETA*ABETA*(KBETA*10 - ABETA)
    return a

def Right_hand_side_T(t, u, x, para_A):
    ABETA = u[0]
    PTAU = u[1]
    KBETA = para_A[0]
    lambdaBETA = para_A[1]
    KPTAU = x[0]
    lambdaPTAUBETA = x[1]
    lambdaPTAU = x[2]
    a = lambdaBETA * ABETA * (KBETA*10 - ABETA)
    t = lambdaPTAUBETA*ABETA + lambdaPTAU*PTAU*(KPTAU*10 - PTAU)
    return np.array([a, t])


def Right_hand_side_N(t, u, x, para_AT):
    ABETA = u[0]
    PTAU = u[1]
    N = u[2]
    KBETA = para_AT[0]
    lambdaBETA = para_AT[1]
    KPTAU = para_AT[3]
    lambdaPTAUBETA = para_AT[4]
    lambdaPTAU = para_AT[5]
    lambdaNtp = x[1]
    lambdaN = x[2]
    KN = x[0]
    a = lambdaBETA * ABETA * (KBETA*10 - ABETA)
    t = lambdaPTAUBETA * ABETA + lambdaPTAU * PTAU * (KPTAU*10 - PTAU)
    n = lambdaNtp * PTAU + lambdaN * N * (KN*10 - N)
    return np.array([a, t, n])


def para_est_obj_A(x, ABETA_data, AGE_dataA, abeta_lambda):
    TimeA = AGE_dataA
    initial = [math.exp(x[2]*33.4-35)]
    # initial = [x[2]]
    t_span = (0,100)
    t = np.arange(0.0, 101, 1)
    
    sigma = np.abs(0.1*0.1*np.array(ABETA_data)/1.96)
    if np.mean(ABETA_data) < 0.01:
        sigma *= 10
        
    sigma = 0.004144736842105263
    
    try:
        results = solve_ivp(Right_hand_side_A, t_span, initial, t_eval = t, args=(x,), rtol = 1e-5)
        TimeA = np.floor(TimeA).astype(int)
        res = []
        for i in range(len(TimeA)):
            res.append(results.y[0][TimeA[i]])
        ret = sum(np.square(res-ABETA_data)/sigma)+abeta_lambda*(results.y[0][-1]-1)**2
    except:
        ret = -1
    return ret


def para_est_obj_T(x, para_A, PTAU_data, AGE_dataP, tau_lambda):
    TimeP = AGE_dataP
    initial = np.array([math.exp(para_A[2]*33.4-35), 0])
    # initial = np.array([para_A[2], 0])
    t_span = np.array([0,100])
    t = np.arange(0.0, 101, 1)
    
    sigma = np.abs(0.1*0.1*np.array(PTAU_data)/1.96)
    sigma = 0.018200630914826495
    
    try:
        results = solve_ivp(Right_hand_side_T, t_span, initial, t_eval = t, args=(x, para_A,), rtol = 1e-5)
        TimeP = np.floor(TimeP).astype(int)
        res1 = []
        for i in range(len(TimeP)):
            res1.append(results.y[1][TimeP[i]])
        ret = sum(np.square(res1 - PTAU_data)/sigma)+tau_lambda*(results.y[1][-1]-1)**2 
    except:
        ret = -1

    return ret
                    

def para_est_obj_N(x, para_AT, N_data, AGE_dataN, neuro_lambda):
    TimeN = AGE_dataN
    initial = np.array([math.exp(para_AT[2]*33.4-35), 0, 0])
    # initial = np.array([para_AT[2], 0, 0])
    t_span = np.array([0,100])
    t = np.arange(0.0, 101, 1)
    
    sigma = np.abs(0.1*0.1*np.array(N_data)/1.96)
    sigma = 0.012209177405209213
    
    try: 
        results = solve_ivp(Right_hand_side_N, t_span, initial, t_eval = t, args=(x, para_AT), rtol = 1e-5)
        TimeN = np.floor(TimeN).astype(int)
        res1 = []
        for i in range(len(TimeN)):
            res1.append(results.y[2][TimeN[i]])
        ret = sum(np.square(res1 - N_data)/sigma)+neuro_lambda*(results.y[2][-1]-1)**2
    except:
        ret = -1
    return ret


def plot_para_est_obj_A(x, ABETA_data, AGE_dataA, ct):
    TimeA = AGE_dataA
    initial = [math.exp(x[2]*33.4-35)]
    # initial = [x[2]]
    # initial = [0.001]
    t_span = (0,100)
    t = np.arange(0.0, 101, 1)
    results = solve_ivp(Right_hand_side_A, t_span, initial, t_eval = t, args=(x,), rtol = 1e-5)
    TimeA = np.floor(TimeA).astype(int)
    return [t, results.y[0]]


def plot_para_est_obj_T(x, para_A, PTAU_data, AGE_dataP, ct):
    TimeP = AGE_dataP
    initial = np.array([math.exp(para_A[2]*33.4-35), 0])
    # initial = np.array([para_A[2], 0])
    t_span = np.array([0,100])
    t = np.arange(0.0, 101, 1)
    results = solve_ivp(Right_hand_side_T, t_span, initial, t_eval = t, args=(x, para_A,), rtol = 1e-5)
    TimeP = np.floor(TimeP).astype(int)
    return [t, results.y[1]]
                    

def plot_para_est_obj_N(x, para_AT, N_data, AGE_dataN, ct):
    TimeN = AGE_dataN
    initial = np.array([math.exp(para_AT[2]*33.4-35), 0, 0])
    # initial = np.array([para_AT[2], 0, 0])
    t_span = np.array([0,100])
    t = np.arange(0.0, 101, 1)
    results = solve_ivp(Right_hand_side_N, t_span, initial, t_eval = t, args=(x, para_AT), rtol = 1e-5)
    TimeN = np.floor(TimeN).astype(int)
    return [t, results.y[2]]

#################################### NEW QUADRATIC FUNCTION HERE

from scipy.misc import derivative
import warnings
warnings.filterwarnings("ignore")

def quadratic_obj_N(x, jacob, hess, para_N, para_AT, N_data, AGE_dataN, neuro_lambda):
    '''
    x = new N parameters, length 3
    Finds new optimized parameter vector based on quadratic
    jacob = 3x1
    hess = 3x3
    '''
    theta_star_G = para_est_obj_N(para_N, para_AT, N_data, AGE_dataN, neuro_lambda)
    diff = (x - para_N).T # 3x1
    diff[0] *= 10
    second_term = np.matmul(diff.T, hess) # 1x3
    linear_term = np.matmul(jacob.T, diff)
    theta_G = theta_star_G + linear_term + np.sum(np.matmul(second_term, diff)/2)
    return theta_G


###########################################


# def Right_hand_side_A(t, u, x):
#     ABETA = u[0]
#     KBETA = x[0]
#     lambdaBETA = x[1]
#     a = lambdaBETA*ABETA*(KBETA - ABETA)
#     return a

# def Right_hand_side_T(t, u, x, para_A):
#     ABETA = u[0]
#     PTAU = u[1]
#     KBETA = para_A[0]
#     lambdaBETA = para_A[1]
#     KPTAU = x[0]
#     lambdaPTAUBETA = x[1]
#     lambdaPTAU = x[2]
#     a = lambdaBETA * ABETA * (KBETA - ABETA)
#     t = lambdaPTAUBETA*ABETA + lambdaPTAU*PTAU*(KPTAU - PTAU)
#     return np.array([a, t])


# def Right_hand_side_N(t, u, x, para_AT):
#     ABETA = u[0]
#     PTAU = u[1]
#     N = u[2]
#     KBETA = para_AT[0]
#     lambdaBETA = para_AT[1]
#     KPTAU = para_AT[3]
#     lambdaPTAUBETA = para_AT[4]
#     lambdaPTAU = para_AT[5]
#     lambdaNtp = x[1]
#     lambdaN = x[2]
#     KN = x[0]
#     a = lambdaBETA * ABETA * (KBETA - ABETA)
#     t = lambdaPTAUBETA * ABETA + lambdaPTAU * PTAU * (KPTAU - PTAU)
#     n = lambdaNtp * PTAU + lambdaN * N * (KN - N)
#     return np.array([a, t, n])


# def para_est_obj_A(x, ABETA_data, AGE_dataA, abeta_lambda):
#     TimeA = AGE_dataA
#     initial = [x[2]]
#     # initial = [x[2]]
#     t_span = (0,100)
#     t = np.arange(0.0, 101, 1)
    
#     sigma = np.abs(0.1*0.1*np.array(ABETA_data)/1.96)
#     if np.mean(ABETA_data) < 0.01:
#         sigma *= 10
        
#     sigma = 0.004144736842105263
    
#     try:
#         results = solve_ivp(Right_hand_side_A, t_span, initial, t_eval = t, args=(x,), rtol = 1e-5)
#         TimeA = np.floor(TimeA).astype(int)
#         res = []
#         for i in range(len(TimeA)):
#             res.append(results.y[0][TimeA[i]])
#         # ret = sum(np.square(res-ABETA_data))+100*(results.y[0][-1]-1)**2
#         # ret = sum(np.square(res-ABETA_data)/np.abs(0.1*0.1*np.array(ABETA_data)/1.96))+100*(results.y[0][-1]-1)**2
#         ret = sum(np.square(res-ABETA_data)/sigma)+abeta_lambda*(results.y[0][-1]-1)**2
#         # ret = sum(np.square(res-ABETA_data)/sigma)+abeta_lambda*(results.y[0][100]-1)**2
#         # ret = sum(np.square(res-ABETA_data)/sigma)
#     except:
#         ret = -1
#     return ret


# def para_est_obj_T(x, para_A, PTAU_data, AGE_dataP, tau_lambda):
#     TimeP = AGE_dataP
#     initial = np.array([para_A[2], 0])
#     t_span = np.array([0,100])
#     t = np.arange(0.0, 101, 1)
    
#     sigma = np.abs(0.1*0.1*np.array(PTAU_data)/1.96)
#     sigma = 0.018200630914826495
    
#     try:
#         results = solve_ivp(Right_hand_side_T, t_span, initial, t_eval = t, args=(x, para_A,), rtol = 1e-5)
#         TimeP = np.floor(TimeP).astype(int)
#         res1 = []
#         for i in range(len(TimeP)):
#             res1.append(results.y[1][TimeP[i]])
#         ret = sum(np.square(res1 - PTAU_data)/sigma)+tau_lambda*(results.y[1][-1]-1)**2 
#         # ret = sum(np.square(res1 - PTAU_data)/sigma)+tau_lambda*(results.y[1][100]-1)**2 
#         # ret = sum(np.square(res1 - PTAU_data)/sigma)
#     except:
#         ret = -1

#     return ret
                    

# def para_est_obj_N(x, para_AT, N_data, AGE_dataN, neuro_lambda):
#     TimeN = AGE_dataN
#     initial = np.array([para_AT[2], 0, 0])
#     t_span = np.array([0,100])
#     t = np.arange(0.0, 101, 1)
    
#     sigma = np.abs(0.1*0.1*np.array(N_data)/1.96)
#     sigma = 0.012209177405209213
    
#     try: 
#         results = solve_ivp(Right_hand_side_N, t_span, initial, t_eval = t, args=(x, para_AT), rtol = 1e-5)
#         TimeN = np.floor(TimeN).astype(int)
#         res1 = []
#         for i in range(len(TimeN)):
#             res1.append(results.y[2][TimeN[i]])
#         ret = sum(np.square(res1 - N_data)/sigma)+neuro_lambda*(results.y[2][-1]-1)**2
#         # ret = sum(np.square(res1 - N_data)/sigma)+neuro_lambda*(results.y[2][100]-1)**2
#         # ret = sum(np.square(res1 - N_data)/sigma)
#     except:
#         ret = -1
#     return ret

# def plot_para_est_obj_A(x, ABETA_data, AGE_dataA, ct):
#     TimeA = AGE_dataA
#     initial = [x[2]]
#     t_span = (0,100)
#     t = np.arange(0.0, 101, 1)
#     results = solve_ivp(Right_hand_side_A, t_span, initial, t_eval = t, args=(x,), rtol = 1e-5)
#     TimeA = np.floor(TimeA).astype(int)
#     return [t, results.y[0]]


# def plot_para_est_obj_T(x, para_A, PTAU_data, AGE_dataP, ct):
#     TimeP = AGE_dataP
#     initial = np.array([para_A[2], 0])
#     t_span = np.array([0,100])
#     t = np.arange(0.0, 101, 1)
#     results = solve_ivp(Right_hand_side_T, t_span, initial, t_eval = t, args=(x, para_A,), rtol = 1e-5)
#     TimeP = np.floor(TimeP).astype(int)
#     return [t, results.y[1]]
                    

# def plot_para_est_obj_N(x, para_AT, N_data, AGE_dataN, ct):
#     TimeN = AGE_dataN
#     initial = np.array([para_AT[2], 0, 0])
#     t_span = np.array([0,100])
#     t = np.arange(0.0, 101, 1)
#     results = solve_ivp(Right_hand_side_N, t_span, initial, t_eval = t, args=(x, para_AT), rtol = 1e-5)
#     TimeN = np.floor(TimeN).astype(int)
#     return [t, results.y[2]]


# def quadratic_obj_N(x, jacob, hess, para_N, para_AT, N_data, AGE_dataN, neuro_lambda):
#     '''
#     x = new N parameters, length 3
#     Finds new optimized parameter vector based on quadratic
#     jacob = 3x1
#     hess = 3x3
#     '''
#     theta_star_G = para_est_obj_N(para_N, para_AT, N_data, AGE_dataN, neuro_lambda)
#     diff = (x - para_N).T # 3x1
#     second_term = np.matmul(diff.T, hess) # 1x3
#     linear_term = np.matmul(jacob.T, diff)
#     # linear_term = 0.0
#     theta_G = theta_star_G + linear_term + np.sum(np.matmul(second_term, diff)/2)
#     return theta_G


ct=1


#%%

# PROFILE LIKELIHOOD - SCRATCH

import random as r
import nlopt

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1).data, arr.std(axis=-1).data

def profile(x, loss_func, theta_num, theta_init, lb, ub, loss_crit, epsilon):
    theta_init_rest = np.delete(theta_init, theta_num)
    lb = np.delete(lb, theta_num)
    ub = np.delete(ub, theta_num)  

    def loss_func_rest(theta_rest):
        theta_full = np.concatenate((theta_rest[0:theta_num], [x], theta_rest[theta_num:len(theta_rest)]), axis=0)
        loss = loss_func(theta_full)
        return loss
   
    theta_opt, y_opt, ret = opt.fmin_l_bfgs_b(loss_func_rest, x0=theta_init_rest, approx_grad=True, bounds=np.array([lb, ub]).T, epsilon=epsilon, maxiter=1000)

    loss = loss_func_rest(theta_opt)
    loss = loss_crit if loss < 0 else loss
    # print(theta_opt, x, loss)
    print(x, loss, ret['warnflag'])

    theta_opt = np.concatenate((theta_opt[0:theta_num], [x], theta_opt[theta_num:]), axis=0)
    return loss, theta_opt, ret

def get_one_endpoint(opt_param_vals, param_num, scan_bnds, bnds, step_num, loss_crit, loss_func, epsilon, log_bools):
    x_2 = opt_param_vals[param_num]
    x_1 = x_2
    theta_init = opt_param_vals
    theta_opt = opt_param_vals
    step = (scan_bnds[param_num][1]-scan_bnds[param_num][0])/step_num
    print(scan_bnds[param_num], step)
    
    if step < 0:
        bound = scan_bnds[i][0]
    else:
        bound = scan_bnds[i][1]

    losses, xs, flags = [], [], []
    loss = loss_crit - chi2.ppf(0.95, df=1)
    theta_opts = []
    while True:
        new_loss, theta_opt, ret = profile(x_2, loss_func, param_num, theta_opt, bnds[:,0], bnds[:,1], loss_crit, epsilon)
        if type(ret) == int:
            if ret > 0:
                flags.append(0)
            else:
                flags.append(2)
        else:
            flags.append(ret['warnflag'])
        if (step>0 and x_2>bound) or (step<0 and x_2<bound):
            print("reached bound {}, no endpoint found".format(bound))
            return losses, xs, flags, None, theta_opts
        if new_loss >= loss_crit:
            print("found endpoint: ", x_2, "loss: ", new_loss)
            xs.append(x_2)
            losses.append(new_loss)
            theta_opts.append(theta_opt)
            return losses, xs, flags, x_2, theta_opts
        
        xs.append(x_2)
        losses.append(new_loss)
        theta_opts.append(theta_opt)
        
        if x_2 != x_1 and log_bools[param_num]<1:
            scaled_step = (x_2 - x_1) / (new_loss - loss)
        elif log_bools[param_num]>0:
            if step < 0:
                log_arr = np.logspace(math.log10(bound+0.1*max([step,-1*bound])), math.log10(x_2-2*step), num=abs(step_num))
                id_curr = np.argmin(np.abs(log_arr-x_2))
                scaled_step = log_arr[id_curr-1] - log_arr[id_curr]
            else:
                # print(x_2,bound,step,x_2-0.1*min([step,x_2]), bound+2*step, step_num)
                log_arr = np.logspace(math.log10(x_2-0.1*min([step,x_2])), math.log10(bound+2*step), num=step_num)
                id_curr = np.argmin(np.abs(log_arr-x_2))
                scaled_step = log_arr[id_curr+1] - log_arr[id_curr]
        else:
            scaled_step = step
            
        true_step = np.argmin([abs(step), abs(scaled_step)])
        # print(step, scaled_step)
        
        loss = new_loss
        x_1 = x_2
        x_2 += [step,abs(scaled_step)*(abs(step)/step)][true_step]
        

def plot_profile(x_values, y_values, flags, inits, loss_crit, ep1s, ep2s, labels, error=False):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    colors = ['g','y','r']
    print(x_values, y_values, inits, ep1s, ep2s)
    for xval, yval, flag, init, ep1, ep2, i in zip(x_values, y_values, flags, inits,
                                                 ep1s, ep2s, np.arange(3)):

        if error:
            axs[i].plot(xval, yval[0], '-', c='#1f77b4', linewidth = 3)
            # plt.fill_between(xval, yval[0]+yval[1], yval[0]-yval[1], alpha=0.2)
        else:
            axs[i].plot(xval, yval, '-', c='#1f77b4', linewidth = 3)
        axs[i].axhline(y=loss_crit, c='k')

        axs[i].set_xlabel(labels[i])
        axs[i].set_ylabel("Profile Likelihood")

    plt.tight_layout()
    plt.show() 

def create_grid(bounds, num_points):
    # Determine the dimensionality from the length of the bounds
    dim = len(bounds) // 2
    
    if dim == 1:  # 1D case
        start, end = bounds
        values = np.linspace(start, end, num_points)
        return [(value,) for value in values]
    
    elif dim == 2:  # 2D case
        x_start, x_end, y_start, y_end = bounds
        x_values = np.linspace(x_start, x_end, num_points)
        y_values = np.linspace(y_start, y_end, num_points)
        return [[(x, y) for x in x_values] for y in y_values]

    elif dim == 3:  # 3D case
        x_start, x_end, y_start, y_end, z_start, z_end = bounds
        x_values = np.linspace(x_start, x_end, num_points)
        y_values = np.linspace(y_start, y_end, num_points)
        z_values = np.linspace(z_start, z_end, num_points)
        return [[[(x, y, z) for x in x_values] for y in y_values] for z in z_values]

df5 = pd.read_excel('Identifiability_Hyperparams_loosenbnds.xlsx')
# df5 = pd.read_excel('Identifiability_Hyperparams.xlsx')


# juliet's expanded
bnds = np.array(((0,10), (0,1), (0,.2), 
                  (0,10), (0,1), (0,1), 
                  (0,10), (0,1), (0,1)))

all_param_names = [["KBETA","lambdaBETA","A_0"],["KPTAU",'lambdaPTAUBETA','lambdaPTAU'],["KN","lambdaNtp","lambdaN"]]
iterations = 30

for rid in [6661]:  # 127, 684, 4043, 767
    print(rid)

    steps = np.array([[40,600,80],
                      [80,120,120],
                      [40,80,80]])
    scan_bnds = np.array(((.9,1.2), (0.001,1), (1e-20, 0.2),
                          (0,7), (1e-35,.1), (0.0000000000001,1),
                          (0,3), (1e-35,1), (1e-25,1)))
    epsilon =  1E-10
    
    log_bools = np.array([[0,0,1],
                      [0,1,0],
                      [0,1,0]])
    
    abeta_epsilon, tau_tol, neuro_epsilon = 2e-10, 1e-6, 2e-10
    abeta_lambda, tau_lambda, neuro_lambda = 5000, 1000, 5000
    identifiable = np.zeros((3,3))
    solve_params = False 
    plot_params = True
    find_identifiability = False 
    plot_uncertainty = False
    linearized = True
    
    redo = False
    
    if rid in df5.RID.tolist() and redo == False:
        hyperparams = np.array(df5[df5['RID']==rid])[0]
        steps = hyperparams[37:46].reshape((3,3))
        scan_bnds = hyperparams[19:37].reshape((9,2))
        identifiable = hyperparams[46:55].reshape((3,3))
        para_A = hyperparams[7:10]
        para_T = hyperparams[10:13]
        para_N = hyperparams[13:16]
        loss_A, loss_T, loss_N = hyperparams[16:19]
        abeta_epsilon, tau_tol, neuro_epsilon = hyperparams[1:4]
        abeta_lambda, tau_lambda, neuro_lambda = hyperparams[4:7]
        epsilon = hyperparams[55]
        solve_params = False
        find_identifiability = False
        plot_uncertainty = False
        linearized = True
        
    # find rid_hyperparams file, if exists, then take model parameters for profile likelihood plotting
    if os.path.exists("{}_hyperparams.xlsx".format(rid)) and rid not in df5.RID.tolist():
        df6 = pd.read_excel("{}_hyperparams.xlsx".format(rid))
        hyperparams = np.array(df6[0])
        para_A = hyperparams[7:10]
        para_T = hyperparams[10:13]
        para_N = hyperparams[13:16]
        loss_A, loss_T, loss_N = hyperparams[16:19]
        abeta_lambda, tau_lambda, neuro_lambda = hyperparams[4:7]
        solve_params = False
        find_identifiability = True
        plot_uncertainty = False
    
    PID = df.loc[df.loc[df.RID == rid].index] 
    PID_indexes = PID.index
    
    PTAU_PID = PTAU.loc[PID.index]
    AAgeP = []
    AAgeP_indexes = [] 
    for j in range(0,len(PID)): 
        l = PID_indexes[j] 
        agept = age[rid] + (df['datetimes'][l].year - age_time[rid].year) 
        if np.isnan(PTAU_PID[l]) == False:
            AAgeP.append(agept)
    
    
    ABETA_PID = ABETA.loc[PID.index]
    AAgeA = []
    for j in range(0,len(PID)):
        l = PID_indexes[j] 
        agea = age[rid] + (df['datetimes'][l].year - age_time[rid].year)
        if np.isnan(ABETA_PID[l]) == False:
            AAgeA.append(agea)
    
    
    N_PID = N.loc[PID.index]
    AAgeN = []
    for j in range(0,len(PID)):
        l = PID_indexes[j] 
        agen = age[rid] + (df['datetimes'][l].year - age_time[rid].year)
        if (np.isnan(N_PID[l]) == False):
            AAgeN.append(agen)
    
    ab = ABETA_PID.to_numpy()
    ab = ab[~np.isnan(ab)]
    pt = PTAU_PID.to_numpy()
    pt = pt[~np.isnan(pt)]
    nn = N_PID.to_numpy()
    nn = nn[~np.isnan(nn)]
    aab = AAgeA
    apt = AAgeP
    nnt = AAgeN
    
    print(ab,aab)
    
    ##### REOPTIMIZE WITH NEW LOSS FUNCTION
    
    if solve_params:        
        initials = (1, 0.12, 0.010)
        para_A, loss_A, opt_info = opt.fmin_l_bfgs_b(para_est_obj_A, x0=initials, args=(ab,aab,abeta_lambda), approx_grad=True, bounds=bnds[:3], epsilon=abeta_epsilon, maxiter=500)
        print(opt_info, loss_A)
        minloss = loss_A 
        grada = []
         
        for i in range(0,iterations): 
            r1 = r.uniform(.8*para_A[0],1.2*para_A[0]) 
            r2 = r.uniform(.8*para_A[1],1.2*para_A[1]) 
            r3 = r.uniform(.8*para_A[2],1.2*para_A[2]) 
            initials = [r1, r2, r3] 
            x_opt, loss_A, opt_info = opt.fmin_l_bfgs_b(para_est_obj_A, x0=initials, args=(ab,aab,abeta_lambda), approx_grad=True, bounds=bnds[:3], epsilon=abeta_epsilon, maxiter=500)
            if (loss_A < minloss): 
                minloss = loss_A
                para_A = x_opt
        
        loss_A = minloss
        
        
    
        print("final Loss_A", loss_A)
        print("final para_A", para_A)
        
        
        # T
    
        minloss = 500
        success = 0
        for i in range(0,8,1):
            initials = (i+0.5, 0.025, 0.05)
            # initials = (i+0.5, 0.1, 0.7)
            try:
                res = opt.minimize(para_est_obj_T, initials, args=(para_A,pt,apt,tau_lambda), method='trust-constr', hess=None, hessp=None, bounds=bnds[3:6], tol=tau_tol, options={'initial_tr_radius': 0.01})
                loss_T = para_est_obj_T(res.x,para_A,pt,apt,tau_lambda)
                if (loss_T < minloss and loss_T > 0):
                    minloss = loss_T
                    para_T = res.x
                success+=1
            except:
                print("fail", i)
                if i == 16 or success > 4:
                    break
                continue
    
        loss_T = minloss    
        print("Final para_T", para_T)
        print("final loss_T", loss_T)
        
        
        # N
    
        minloss = 200
        gradn = []
        
        para_AT = np.concatenate((para_A,para_T))
        initials = [1,0.025,0.4]
        # initials = [0.92, 0.01, 0.1]
        # initials = [1, 0.01, 0.1]
        x_opt, y_opt, opt_info = opt.fmin_l_bfgs_b(para_est_obj_N, x0=initials, args=(para_AT,nn,nnt,neuro_lambda), approx_grad=True, bounds=bnds[6:], epsilon=neuro_epsilon, maxiter=500)
        loss_N = para_est_obj_N(x_opt,para_AT,nn,nnt,neuro_lambda)
        minloss = loss_N
        para_N = x_opt 
            
        for i in range(1,8):
            initials = [i,0.025,0.2]
            x_opt, y_opt, opt_info = opt.fmin_l_bfgs_b(para_est_obj_N, x0=initials, args=(para_AT,nn,nnt,neuro_lambda), approx_grad=True, bounds=bnds[6:], epsilon=neuro_epsilon, maxiter=500)
            print(x_opt,y_opt)
            print(opt_info)
            loss_N = para_est_obj_N(x_opt,para_AT,nn,nnt,neuro_lambda)
            if (loss_N < minloss and loss_N > 0):
                print("STORE")
                minloss = loss_N
                para_N = x_opt 
                gradn = opt_info["grad"]
                optinfo = opt_info
                
        loss_N = minloss
            
        print("final loss_N", loss_N)
        print("final para_N", para_N)
    
    if linearized:
        def central_der(func, input_val, dx):
            upper = func(input_val+dx)
            lower = func(input_val-dx)
            return (upper-lower)/(dx*2)
        
        # find jacobian and hessian at para_N using central difference
        hessian = []
        para_N=list(para_N)
        dx = np.array([1e-10, 1e-10, 1e-10])
        dx = [1e-11]*3
        
        for i in range(len(para_N)):
            row_i = [] # second-order partial derivatives of the i-th row
            gradient = []
            for j in range(len(para_N)):
                pd_j = lambda x: central_der(lambda x_j: para_est_obj_N(x[:j] + [x_j] + x[j+1:], para_AT, nn, nnt, neuro_lambda), x[j], dx[j])
                gradient.append(pd_j(para_N))
                # print(gradient)
                second_func = lambda x_i: pd_j(para_N[:i] + [x_i] + para_N[i+1:])
                pd_ij = central_der(second_func, para_N[i], dx[i])
                row_i.append(pd_ij)
            hessian.append(row_i)
        
        print("Numerical gradient: ", gradient)
        print("Numerical hessian: ", hessian)

        initials = hyperparams[13:16]
        initials[1] *= 200
        initials[0], initials[2] = 1, 0.4 # [1, 4.854464568821866e-05, 0.4]
        # initials = [1, 2e-7, 0.5]
        initials = [1, 0.1, 0.4]
        x_opt, y_opt, opt_info = opt.fmin_l_bfgs_b(quadratic_obj_N, 
                                                   x0=initials, args=(np.array(gradient), 
                            np.array(hessian),np.array(para_N) , para_AT,nn,nnt,neuro_lambda), 
                            approx_grad=True, bounds=bnds[6:], epsilon=neuro_epsilon, maxiter=500) # factr=1000000
        print("Re-optimized using quadratic objective: ", x_opt)
        print("Old para_N: ", para_N)
        print(opt_info)
        para_N = x_opt
        
    
    para_AT = np.concatenate([para_A,para_T])
    
    
    def plot():
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        num=0

        t, a_traj = plot_para_est_obj_A(para_A, ab, aab, ct)
        t, t_traj = plot_para_est_obj_T(para_T, para_A, pt, apt, ct)
        t, n_traj = plot_para_est_obj_N(para_N, para_AT, nn, nnt, ct)
        # ax = axs[num]
        ax = axs
        ax.plot(aab, ab, '.r', label="abeta")
        ax.plot(apt, pt, '.b', label="tau")
        ax.plot(nnt, nn, '.y', label="neurodegen")
        ax.plot(t, a_traj, '-r')
        ax.plot(t, t_traj, '-b')
        ax.plot(t, n_traj, '-y')
        ax.set_title(rid)
        
    if plot_params:
        plot()
        # continue
    

    opt_param_vals = np.array([para_A,para_T,para_N])
    
    print(opt_param_vals)
    
    a_loss_crit = chi2.ppf(0.95, df=1) + para_est_obj_A(opt_param_vals[0], ab, aab, abeta_lambda)
    t_loss_crit = chi2.ppf(0.95, df=1) + para_est_obj_T(opt_param_vals[1], opt_param_vals[0], pt, apt, tau_lambda)
    n_loss_crit = chi2.ppf(0.95, df=1) + para_est_obj_N(opt_param_vals[2], np.concatenate([opt_param_vals[0],opt_param_vals[1]]), nn, nnt, neuro_lambda)
    
    
    print(a_loss_crit, t_loss_crit, n_loss_crit)
    
    # continue
    
    all_theta_opts = [[[],[],[]],[[],[],[]],[[],[],[]]]
    all_eps = [[],[]]
    
    if find_identifiability:
        bio_losses = [a_loss_crit, t_loss_crit, n_loss_crit]
        bio_objs = [lambda x: para_est_obj_A(x, ab, aab, abeta_lambda), lambda x: para_est_obj_T(x, para_A, pt, apt, tau_lambda), lambda x: para_est_obj_N(x, np.concatenate([para_A,para_T]), nn, nnt, neuro_lambda)]
        for bio in [2]:
            ep1s = []
            ep2s = []
            x_values = []
            y_values = []
            flags = []
            for i in [0,1,2]:    
                right_losses, right_xs, right_flags, right_ep, right_theta_opts = get_one_endpoint(opt_param_vals[bio], i, 
                                                                                                    scan_bnds[bio*3:bio*3+3], bnds[bio*3:bio*3+3], int(steps[bio][i]), 
                                                                                                    bio_losses[bio], bio_objs[bio], epsilon, log_bools[bio])
                left_losses, left_xs, left_flags, left_ep, left_theta_opts = get_one_endpoint(opt_param_vals[bio], i, 
                                                                                              scan_bnds[bio*3:bio*3+3], bnds[bio*3:bio*3+3], int(-steps[bio][i]), 
                                                                                              bio_losses[bio], bio_objs[bio], epsilon, log_bools[bio])
                yval = list(reversed(left_losses))+right_losses
                xval = list(reversed(left_xs))+right_xs
                theta_opts = list(reversed(left_theta_opts))+right_theta_opts
                all_theta_opts[bio][i] = theta_opts
                y_values.append(yval)
                x_values.append(xval)
                
                flags.append(list(reversed(left_flags))+right_flags)
                ep1s.append(left_ep)
                ep2s.append(right_ep)
                
                if (left_ep is not None) and (right_ep is not None):
                    identifiable[bio][i] = 1
                    
            all_eps[0].append(ep1s)
            all_eps[1].append(ep2s)
            
            plot_profile(x_values, y_values, flags, opt_param_vals[0], bio_losses[bio], ep1s, ep2s, all_param_names[bio], error=False)
    
    def save():
        array = np.concatenate([[rid, abeta_epsilon, tau_tol, neuro_epsilon, abeta_lambda, tau_lambda, neuro_lambda], para_A, para_T, para_N, [loss_A, loss_T, loss_N], scan_bnds.flatten(), steps.flatten(), identifiable.flatten(), [epsilon]])
        array_df = pd.DataFrame(array)
        array_df.to_excel("{}_hyperparams.xlsx".format(rid))


#%%


############ UNCERTAINTY QUANTIFICATION

from scipy.stats import beta
import scipy.stats as stats
from pymc_functions import *
from scipy.stats import norm
from scipy.linalg import cholesky, solve
from scipy.stats import multivariate_normal
from matplotlib.ticker import ScalarFormatter


df5 = pd.read_excel('Identifiability_Hyperparams_loosenbnds.xlsx')

# parameters have been scaled to (0,1)
bnds = np.array(((0,1), (0,1), (0,1), 
                  (0,1), (0,1), (0,1), 
                  (0,1), (0,1), (0,1)))

all_param_names = [[r'$K_{A_{\beta}}$',r'$\lambda_{A_{\beta}}$',r'$A_0$'],[r'$K_{\tau_{\rho}}$',r'$\lambda_{\tau_{\rho}A_{\beta}}$',r'$\lambda_{\tau_{\rho}}$'],[r'$K_N$',r'$\lambda_{N\tau_{\rho}}$',r'$\lambda_N$']]

iterations = 30

def log_likelihood_MALA(x, biomarker, data, age_data, additional_params, lambda_bio):
    if biomarker == 0:
        loglik = para_est_obj_A(x, data, age_data, lambda_bio)
    elif biomarker == 1:
        loglik = para_est_obj_T(x, additional_params, data, age_data, lambda_bio)
    else:
        loglik = para_est_obj_N(x, additional_params, data, age_data, lambda_bio)

    return loglik/(-2)

def logpost(x, biomarker, data, 
            age_data, additional_params, 
            param_num, optimal_vars, scale, bnds, df5, lambda_bio):
    # print(x)
    new_params = optimal_vars[biomarker].copy()
    
    ##### Gaussian prior around optimal values
    scale = 0.1
    cov = np.eye(len(x))*scale
    prob = multivariate_normal.pdf(x, mean=optimal_vars[biomarker][param_num], cov=cov)
    logprior = np.log(prob)
    
    loglik = log_likelihood_MALA(new_params, biomarker, data, age_data, additional_params, lambda_bio)
    
    return loglik + logprior

def loglik_approxgrad(x, biomarker, data, age_data, optimal_vars, param_num, additional_params, lambda_bio):
    grad = np.zeros((len(x),1))
    eps = np.array([1e-14,1e-14,1e-14])
    for i, j in zip(np.arange(len(x)), param_num):
        x_plus, x_minus = optimal_vars[biomarker].copy(), optimal_vars[biomarker].copy()
        for k, l in zip(np.arange(len(x)), param_num):
            x_plus[l] = x.T[0][k]
            x_minus[l] = x.T[0][k]
        x_plus[j] += eps[i]
        x_minus[j] -= eps[i]
        y_plus = log_likelihood_MALA(x_plus, biomarker, data, age_data, additional_params, lambda_bio)
        y_minus = log_likelihood_MALA(x_minus, biomarker, data, age_data, additional_params, lambda_bio)
        grad[i] = (y_plus-y_minus)/(2*eps[i])

    return grad

def lgradient(x, biomarker, data, 
            age_data, additional_params, 
            param_num, optimal_vars, scale, df5, lambda_bio):
    
    loglik_gr = loglik_approxgrad(x, biomarker, data, age_data, optimal_vars, param_num, additional_params, lambda_bio)
    
    ##### Gaussian prior
    sigma = np.eye(len(x))*0.1
    center = np.array(optimal_vars[biomarker][param_num]).reshape(-1,1)
    prior_gr = -np.matmul(np.linalg.inv(sigma),(x-center))
    
    return loglik_gr + prior_gr

def MALA(num_samples, burn_in, epsilon, S,
         biomarker, data, bnds, param_num,
         age_data, additional_params, optimal_vars, scale, initial, p, df5, lambda_bio, 
         gradient, hessian, para_N):
    out = np.zeros((num_samples, p))
    params = np.array(initial).reshape((p,1))
    epsilon = epsilon.reshape((p,1))
    
    A = cholesky(S, lower=True)
    S1 = np.linalg.inv(S)
    
    lgrad = S @ lgradient(params, biomarker, data, 
                age_data, additional_params, param_num, optimal_vars, 
                scale, df5, lambda_bio, gradient, hessian, para_N)

    logp = logpost(params, biomarker, data, 
                age_data, additional_params, param_num, 
                optimal_vars, scale, bnds, df5, lambda_bio, 
                gradient, hessian, para_N)

    sigma2 = epsilon**2 / p**(1 / 3)
    sigma = np.sqrt(sigma2)
    
    counts = 0
    acceptance_rate = 0
    for rn in tqdm(range(burn_in + num_samples), desc="MCMC Sampling Progress"):
        new_params = params + sigma2 / 2 * lgrad + sigma * (A @ np.random.normal(size=p)[:, np.newaxis])

        logpnew = logpost(new_params, biomarker, data, 
                    age_data, additional_params, param_num, 
                    optimal_vars, scale, bnds, df5, lambda_bio, gradient, hessian, para_N)
        lgrad_new = S @ lgradient(new_params, biomarker, data, 
                    age_data, additional_params, param_num, 
                    optimal_vars, scale, df5, lambda_bio, gradient, hessian, para_N)

        
        diffold = params - new_params - sigma2 / 2 * lgrad_new
        diffnew = new_params - params - sigma2 / 2 * lgrad
        
        
        qold = -diffold.T @ S1 @ diffold / (2 * sigma2)
        qnew = -diffnew.T @ S1 @ diffnew / (2 * sigma2)
        
        
        exp_ = (logpnew - logp + (qold - qnew).sum()).astype(float)
        
        alpha = np.min([np.ones(exp_.shape),np.exp(exp_)],axis=0)
        
        if np.all(np.random.uniform() < alpha) and np.all(new_params > bnds[:,0].reshape((p,1))) and np.all(new_params < bnds[:,1].reshape((p,1))):
            logp = logpnew
            lgrad = lgrad_new
            params = new_params
                        
            if rn >= burn_in:
                counts+=1
                    
        if rn >= burn_in:
            out[rn - burn_in, :] = params.flatten()

    return out[out != 0], counts

np.random.seed(42)


    
epsilons = {
    416: [[0.0003, 0.0005, 0.0011],
                  [0,0,0],
                   [0.0004, 0.0013, 0]],
    31: [[0.00026,0.0004,0.0012],
        [0.,0.0015,0.0],
        [0,0,0]],
    127: [[0.0003,0.0002,0.001],
          [0.0007,0.005,0.0],
          [0.0,0.001,0.002]],
    337:[[0.0,0.,0.00011],
         [0.0005,0.00512,0.0],
         [0.0,0.008,0.0]],
    377:[[0.0,0.000005,0.0003],
         [0.0,0.0,0.0],
         [0.,0.000008,0.0]],
    467:[[0.0,0.000001,0.0003],
         [0.0,0.0,0.0],
         [0.0005,0.0,0.0]],
    618: [[0.0002, 0.0004, 0.0007],
          [0,0,0],
          [0.0,0.001,0.0]],
    677: [[0.00025,0.0,0.0005],
          [0,0,0],
          [0.0,0.0003,0.0]],
    6013: [[0.0,0.00003,0.0004],
           [0.0008,0.004,0.0],
           [0.0002,0.004,0.0]],
    6216: [[0.0002,0.0,0.0],
           [0,0,0],
           [0,0,0]],
    6264: [[0.001,0.0,0.0],
           [0,0,0],
           [0,0,0]],
    6689: [[0.001,0.0,0.0],
           [0,0.00019,0],
           [0,0,0]],
    6683: [[0.0004,0.0,0.0],
           [0,0.0,0],
           [0,0.00001,0.000006]],
    679: [[0.0002,0.0007,0.0013],
           [0,0.0,0],
           [0,0.0011,0.0014]],
    6142: [[0.0,0.000004,0.0003],
           [0,0.0,0],
           [0,0.0,0.0]],
    6549: [[0.001,0.0,0.0],
           [0,0.0,0],
           [0,0.00003,0.0]], 
    6433: [[0.0004,0.0,0.0],
           [0,0.0,0],
           [0.0003,0,0.001]], 
    6377: [[0.0,0.00025,0.0],
           [0,0.0,0],
           [0.0003,0.0,0.0]], 
    1052: [[0.0,0.0001,0.0003],
           [0.0007,0.005,0],
           [0.0,0.0011,0.002]], 
    1378: [[0.0003567,0.001,0.0],
           [0.0006,0.007,0],
           [0.0005,0.0000001,0.0]], 
    2380: [[0.0,0.0003,0.001],
           [0.0006,0.0,0],
           [0.0,0.0045,0.0]], 
    4149: [[0.0003,0.001,0.0020],
           [0,0.0,0],
           [0,0.0,0]], 
    4187: [[0.0,0.00001,0.0003],
           [0,0.0,0],
           [0,0.00008,0]], 
    4199: [[0.0,0.00004,0.0],
           [0.0009,0.007,0],
           [0.0,0.00003,0.002]], 
    4212: [[0.0,0.00001,0.0003],
           [0.0,0.0,0.0],
           [0.0,0.0,0.02]], 
    4272: [[0.0002,0.00055,0.0011],
           [0.0,0.0012,0.002],
           [0.0,0.003,0.0]], 
    4293: [[0.0,0.0,0.0004],
           [0.0009,0.007,0.0],
           [0.0,0.0001,0.0]], 
    4404: [[0.0,0.0001,0.0],
           [0.0,0.0,0.0],
           [0.0,0.0000009,0.000015]],
    2187: [[0.0,0.00001,0.00035],
           [0.0015,0.004,0.0],
           [0.0,0.0001,0.0]],
    2379: [[0.0,0.000011,0.00022],
           [0.0,0.0,0.0],
           [0.0,0.0000062,0.00005]], 
    4036: [[0.0,0.000015,0.000185],
           [0.0013,0.0062,0.0],
           [0.0,0.000012,0.0009]], 
    6039: [[0.0008,0.0002,0.0],
           [0.0007,0.005,0.0],
           [0.0005,0.0000015,0.0]], 
    6100: [[0.0003,0.0,0.0],
           [0.0008,0.0,0.0],
           [0.0004,0.003,0.0]],
    6179: [[0.0003,0.0005,0.0012],
           [0.0,0.0,0.0],
           [0.0005,0.002,0.0]],
    6347: [[0.0005,0.0006,0.0],
           [0.0,0.001,0.0013],
           [0.001,0.0,0.0]],
    6389: [[0.0003,0.0007,0.0],
           [0.0,0.005,0.015],
           [0.0,0.0035,0.0025]],
    680: [[0.0,0.000036,0.00021],
           [0.00095,0.0035,0.0],
           [0.0,0.00097,0.0014]],
    684: [[0.000015,0.0000113,0.00010925],
           [0.0007,0.0025,0.0],
           [0.0,0.0004,0.0]],
    731: [[0.0006,0.00,0.0],
           [0.0,0.00003,0.0],
           [0.0,0.0000045,0.0002]],
    767: [[0.00022,0.00008,0.0004],
           [0.0013,0.005,0.0],
           [0.0,0.00015,0.0]],
    1280: [[0.0,0.0005,0.0014],
           [0.0,0.0,0.0],
           [0.0,0.0004,0.0]],
    4021: [[0.0,0.00011,0.0006],
           [0.0014,0.011,0.0],
           [0.0,0.0004,0.0]],
    4028: [[0.0,0.00055,0.0017],
           [0.0,0.0,0.0],
           [0.0,0.0002,0.002]],
    4376:[[0.0,0.0,0.001],
           [0.0015,0.008,0.0],
           [0.0,0.0002,0.0]],
    4224:[[0.0,0.000045,0.0003],
           [0.00075,0.0,0.0],
           [0.0,0.0,0.0014]],
    6661: [[0.0005,0.0,0.0],
           [0,0.0,0],
           [0,0.000000032,0.000075]], # 0.001,0.0678
            } 
    
# total samples, thinning, burnin, switching
running_samples = {416:[[2400,5,200,200],[0,0,0,0],[4000,5,200,1000]],
                   31:[[2400,5,100,600],[2000,5,100,500],[0,0,0,0]],
                   127:[[2400,5,100,600],[3600,5,100,600],[3000,5,100,500]],
                   337:[[4000,5,1000,1000],[4000,5,500,500],[5000,5,1000,1000]],
                   377:[[4000,5,500,200],[0,0,0,0],[7000,5,100,7000]],
                   467:[[3000,5,100,600],[0,0,0,0],[6000,5,100,2000]],
                   618:[[0,0,0,0],[2400,5,100,600],[4000,5,100,1000]],
                   677:[[3250,5,100,650],[0,0,0,0],[7000,10,1000,1000]],
                   6013:[[4800,5,600,600],[3125,5,600,625],[3125,5,600,625]],
                   6216:[[5000,5.500,1000],[0,0,0,0],[0,0,0,0]],
                   6264:[[5000,5,500,1000],[0,0,0,0],[0,0,0,0]],
                   6689:[[5000,5,500,1000],[7000,5,1000,1000],[0,0,0,0]],
                   6683:[[5000,5,500,1000],[0,0,0,0],[4200,6,1000,600]],
                   679: [[3000,2,100,600],[0,0,0,0],[3750,4,100,625]],
                   6142: [[3000,5,100,600],[0,0,0,0],[0,0,0,0]],
                   6661: [[3000,5,100,1000],[0,0,0,0],[6000,2,100,1000]], ### N BADBADBAD
                   6549: [[3000,5,100,1000],[0,0,0,0],[3000,5,100,1000]],
                   6433: [[3000,5,100,1000],[0,0,0,0],[3600,5,600,600]],
                   6377: [[5000,5,100,1000],[0,0,0,0],[5000,5,100,1000]],
                   1052:[[3600,6,600,600],[3600,6,600,600],[3600,6,600,600]],
                   1378:[[3250,6,900,650],[3000,6,600,600],[1980,2,400,660]],
                   2380:[[3000,2,200,600],[0,0,0,0],[3600,2,800,600]],
                   4149: [[3600,3,100,600],[0,0,0,0],[0,0,0,0]],
                   4187: [[3600,3,100,600],[0,0,0,0],[5400,20,100,600]],
                   4199:[[3600,5,100,600],[3200,5,100,640],[3125,5,100,625]],
                   4212:[[3600,6,100,600],[0,0,0,0],[14000,6,100,1000]],
                   4272:[[3000,3,100,600],[3350,8,100,670],[3000,5,100,1000]],
                   4293:[[7000,5,100,1000],[3200,5,100,640],[6000,5,100,1000]],
                   4404:[[7000,5,100,1000],[3200,5,100,640],[3125,4,100,625]],
                   2187:[[3150,4,100,630],[3250,4,100,650],[6000,5,100,1000]],
                   2379: [[3870,4,100,645],[0,0,0,0],[3250,6,100,650]],
                   4036:[[3870,5,100,645],[3870,5,100,645],[3780,4,100,630]],
                   6039:[[3900,4,100,650],[3250,5,100,650],[3250,5,100,650]],
                   6100:[[5000,5,100,1000],[7000,5,100,1000],[3750,4,100,625]],
                   6179:[[10500,10,100,700],[0,0,0,0],[3900,4,100,650]],
                   6347:[[3250,5,100,650],[3870,5,100,645],[5000,5,100,1000]],
                   6389:[[3870,5,100,645],[3900,5,100,650],[3900,5,100,650]],
                   680:[[3150,5,100,630],[3150,5,100,630],[3150,5,100,630]],
                   684:[[4800,5,100,120],[3780,4,100,630],[3150,5,100,630]],
                   731:[[5000,5,100,1000],[7000,5,100,1000],[3150,5,100,630]],
                   767:[[2800,4,100,700],[3960,4,100,660],[6000,4,100,1000]],
                   1280:[[3250,4,100,650],[0,0,0,0],[6000,4,100,1000]],
                   4021:[[3900,5,100,650],[3900,5,100,650],[6000,4,100,1000]],
                   4028:[[4080,5,100,680],[0,0,0,0],[3780,5,100,630]],
                   4376:[[5000,5,100,1000],[3250,5,100,650],[5000,5,100,1000]],
                   4224:[[3200,4,100,640],[7000,6,100,1000],[7000,5,100,1000]],
                   }
    
for rid in [6661]: 
    print(rid)
    
    if rid in df5.RID.tolist():
        hyperparams = np.array(df5[df5['RID']==rid])[0]
        para_A = hyperparams[7:10]
        para_T = hyperparams[10:13]
        para_N = hyperparams[13:16]
        
        para_A[0] /= 10
        para_A[2] = (math.log(para_A[2])+35)/33.4
        para_T[0] /= 10
        para_N[0] /= 10
        
        para_AT = np.concatenate((para_A, para_T))
        abeta_lambda, tau_lambda, neuro_lambda = hyperparams[4:7]
        lambdas = [abeta_lambda, tau_lambda, neuro_lambda]
    else:
        print("hyperparameters have not been found yet")
        break
        
    
    PID = df.loc[df.loc[df.RID == rid].index] 
    PID_indexes = PID.index
    
    PTAU_PID = PTAU.loc[PID.index]
    AAgeP = []
    AAgeP_indexes = [] 
    for j in range(0,len(PID)): 
        l = PID_indexes[j] 
        agept = age[rid] + (df['datetimes'][l].year - age_time[rid].year) 
        if np.isnan(PTAU_PID[l]) == False:
            AAgeP.append(agept)
    
    
    ABETA_PID = ABETA.loc[PID.index]
    AAgeA = []
    for j in range(0,len(PID)):
        l = PID_indexes[j] 
        agea = age[rid] + (df['datetimes'][l].year - age_time[rid].year)
        if np.isnan(ABETA_PID[l]) == False:
            AAgeA.append(agea)
    
    
    N_PID = N.loc[PID.index]
    AAgeN = []
    for j in range(0,len(PID)):
        l = PID_indexes[j] 
        agen = age[rid] + (df['datetimes'][l].year - age_time[rid].year)
        if (np.isnan(N_PID[l]) == False):
            AAgeN.append(agen)
    
    ab = ABETA_PID.to_numpy()
    ab = ab[~np.isnan(ab)]
    pt = PTAU_PID.to_numpy()
    pt = pt[~np.isnan(pt)]
    nn = N_PID.to_numpy()
    nn = nn[~np.isnan(nn)]
    aab = AAgeA
    apt = AAgeP
    nnt = AAgeN
    
        
    def plot():
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        num=0

        t, a_traj = plot_para_est_obj_A(para_A, ab, aab, ct)
        t, t_traj = plot_para_est_obj_T(para_T, para_A, pt, apt, ct)
        t, n_traj = plot_para_est_obj_N(para_N, para_AT, nn, nnt, ct)
        # ax = axs[num]
        ax = axs
        ax.plot(aab, ab, '.r', label="abeta")
        ax.plot(apt, pt, '.b', label="tau")
        ax.plot(nnt, nn, '.y', label="neurodegen")
        ax.plot(t, a_traj, '-r')
        ax.plot(t, t_traj, '-b')
        ax.plot(t, n_traj, '-y')
        ax.set_title(rid)
        
    plot()
        
    ##############################
    
    # hyperparams
    bio, para = 2, np.array([1,2])
    MALA_ = True
    
    # set data, params, names
    opt_param_vals = np.array([para_A,para_T,para_N])
    all_param_names = np.array(all_param_names)
    if bio == 0:
        data = ab
        age_data = aab
        extra = None
        color = "red"
        bio_name = "Amyloid beta"
        param_name = "A"
    elif bio == 1:
        data = pt
        age_data = apt
        extra = opt_param_vals[0]
        color = "blue"
        bio_name = "Phosphorylated Tau"
        param_name = "T"
    else:
        data = nn
        age_data = nnt
        extra = np.concatenate([opt_param_vals[0],opt_param_vals[1]])
        color = "orange"
        bio_name = "Neurodegeneration"
        param_name = "N"
    
    # RUN MCMC
    num_samples = 630*6
    thinning = 4
    burn_in = 100
    switching = 630
    print("total samples, thinning, burnin, switching")
    print(num_samples, thinning, burn_in, switching)
    p = len(para)
    S = np.eye(p)
    scale = 0.5
    
    
    if MALA_:
        rid_epsilon = np.array(epsilons[rid])
        
        epsilon = rid_epsilon[bio][para]
        initial = opt_param_vals[bio][para]
        
        print('Starting MCMC on {}'.format(all_param_names[bio][para]))
        print("epsilon: ", epsilon, "initial: ", initial)
        print("bounds: ", bnds[bio*3+para])
        print("DATA: ", data, age_data)
        
        # I WANT TO BE ABLE TO PICK OFF OF WHERE I LEFT OFF
        current_iter = np.zeros(len(para)).astype(int)
        # initialize, just burn in
        full_MCMC = np.zeros((num_samples,len(para)))
        temp_MCMC = np.array([opt_param_vals[bio][para]]*switching)
        individual_counts = np.zeros(len(para)).astype(int)
        temp_opt_param_vals = opt_param_vals.copy()
        while (current_iter < np.ones(len(para)) * num_samples).all():
            for idx, coord in enumerate(para):
                print(coord, current_iter)
                initial = temp_MCMC[-1,idx]
                p = 1
                S = np.eye(p)
                epsilon = rid_epsilon[bio][coord]
                coord_MCMC, counts = MALA(switching, 0, epsilon, S,
                          bio, data, bnds[bio*3+coord].reshape(1,2), [coord], 
                          age_data, extra, opt_param_vals, scale, initial, p, df5, 
                          lambdas[bio], gradient, hessian, para_N
                          )
                individual_counts[idx] += counts
                current_iter[idx] += switching
                full_MCMC[current_iter[idx]-switching:current_iter[idx], idx] = coord_MCMC
                temp_MCMC[:,idx] = coord_MCMC
                temp_opt_param_vals[bio][coord] = coord_MCMC[-1]
        
        # MCMC_flattened = np.vstack([fit_MCMC, fit_MCMC2])
        MCMC_flattened = np.vstack([full_MCMC, full_MCMC])
        MCMC = MCMC_flattened.reshape((2,num_samples,len(para)))
        
        print("Shape of MCMC stack: ", MCMC.shape)
        
        for i, j in zip(np.arange(len(para)), para):
            ess = effective_n(MCMC[...,i], [all_param_names[bio][j]])
            print("Effective sample size for param {}: ".format(all_param_names[bio][j]), ess)
            
            autocorrelation_int = num_samples/ess[all_param_names[bio][j]]
            
            print("Integrated autocorrelation time: ", autocorrelation_int)
        
        # print("Acceptance rate of last MCMC chain: ", counts/num_samples)
        print("Acceptance rate of last MCMC chain: ", individual_counts/num_samples)
        
        
    for i, j in zip(np.arange(len(para)), para):
        plt.figure(figsize=(5,5))
        if bio == 0 and j == 2:
            plt.hist(np.exp(MCMC[0,...,i][burn_in::thinning]*33.4-35), bins=15, density=True, alpha=0.75)
            plt.axvline(x=math.exp(opt_param_vals[bio][j]*33.4-35), color='k', linestyle="--", linewidth=2, label='Optimized Parameter')
        elif j == 0:
            plt.hist(MCMC[0,...,i][burn_in::thinning]*10, bins=15, density=True, alpha=0.75)
            plt.axvline(x=opt_param_vals[bio][j]*10, color='k', linestyle="--", linewidth=2, label='Optimized Parameter')
        else:
            plt.hist(MCMC[0,...,i][burn_in::thinning], bins=15, density=True, alpha=0.75)
            plt.axvline(x=opt_param_vals[bio][j], color='k', linestyle="--", linewidth=2, label='Optimized Parameter')
        
        # plt.hist(MCMC[0,...,i][::thinning], bins=15, density=True, alpha=0.75)
        # plt.axvline(x=opt_param_vals[bio][j], color='k', linestyle="--", linewidth=2, label='Optimized Parameter')
        
        plt.xlabel('{}'.format(all_param_names[bio][j]))
        # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax = plt.gca()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))  # force scientific notation outside this range
        ax.xaxis.set_major_formatter(formatter)
        # plt.title("posterior of {}".format(all_param_names[bio][j]))
        plt.savefig("posteriors/{}_{}_{}.pdf".format(rid,bio_name[0],j))
        plt.show()
     
        # Plot trace in the second figure
        plt.figure(figsize=(4,4))
        plt.plot(MCMC[0,...,i][burn_in::thinning],'k')
        plt.xlabel('{}'.format(all_param_names[bio][j]))
        plt.ylabel("trace")
        plt.show()
    
        autocorrplot(MCMC[...,i][:,burn_in::thinning], max_lag=100, varnames=[all_param_names[bio][j]])
    
        # Monte Carlo propagation of uncertainty
        sampled_xs = np.random.choice(MCMC[0,...,i][burn_in::thinning], size=1000, replace=True)
        lower_bound = np.percentile(sampled_xs, 2.5)
        upper_bound = np.percentile(sampled_xs, 97.5)
        
        sampled_xs = sampled_xs[(sampled_xs >= lower_bound) & (sampled_xs <= upper_bound)]
    
        solutions = []
        for x in sampled_xs:
            temp = opt_param_vals[bio].copy()
            temp[j] = x
            if bio == 0:
                t, traj = plot_para_est_obj_A(temp, ab, aab, lambdas[bio])
            elif bio == 1:
                t, traj = plot_para_est_obj_T(temp, para_A, pt, apt, lambdas[bio])
            else:
                t, traj = plot_para_est_obj_N(temp, para_AT, nn, nnt, lambdas[bio])
            solutions.append(traj)
            
        # Compute mean and confidence intervals
        solutions = np.array(solutions)
        mean_sol = np.mean(solutions, axis=0)
        std_sol = np.std(solutions, axis=0)
        
        # Plot results
        plt.figure(figsize=(4,4))
        plt.plot(t, mean_sol, label="mean trajectory", color="k")
        plt.fill_between(t, np.min(solutions,axis=0), np.max(solutions,axis=0), color=color, alpha=0.3, label="95% Confidence Interval")
        plt.scatter(age_data, data, color=color, marker='x', label="biomarker data")
        plt.xlabel("age")
        plt.ylabel(bio_name)
        plt.legend()
        plt.show()

    
    # Monte Carlo propagation of uncertainty, joint
    total = MCMC[0][burn_in::thinning].shape[0]
    row_i = np.random.choice(total, size=min([1000,total]), replace=True)
    sampled_xs = MCMC[0][burn_in::thinning][row_i]
    lower_bound = np.percentile(sampled_xs, 2.5, axis=0)
    upper_bound = np.percentile(sampled_xs, 97.5, axis=0)
    
    sampled_xs = sampled_xs[np.all(sampled_xs >= lower_bound,axis=1) & np.all(sampled_xs <= upper_bound,axis=1)]

    solutions = []
    for x in sampled_xs:
        temp = opt_param_vals[bio].copy()
        for i, j in zip(np.arange(len(x)), para):
            temp[j] = x[i]
        if bio == 0:
            t, traj = plot_para_est_obj_A(temp, ab, aab, lambdas[bio])
        elif bio == 1:
            t, traj = plot_para_est_obj_T(temp, para_A, pt, apt, lambdas[bio])
        else:
            t, traj = plot_para_est_obj_N(temp, para_AT, nn, nnt, lambdas[bio])
        solutions.append(traj)
    
    # Compute mean and confidence intervals
    solutions = np.array(solutions)
    mean_sol = np.mean(solutions, axis=0)
    std_sol = np.std(solutions, axis=0)
    
    # Plot results
    plt.figure(figsize=(4,4))
    plt.plot(t, mean_sol, label="Mean Trajectory", color="k")
    plt.fill_between(t, np.min(solutions,axis=0), np.max(solutions,axis=0), color=color, alpha=0.3, label="95% Confidence Interval")
    plt.scatter(age_data, data, color=color, marker='x', label="Data")
    plt.xlabel("Age (Years)")
    plt.ylabel(bio_name)
    plt.legend()
    plt.savefig("uncertainties/{}_{}.pdf".format(rid,bio_name[0]))
    plt.show()
    
    
    ### calculate average confidence interval across data points
    intervals = np.max(solutions,axis=0)[np.floor(age_data).astype(int)] - np.min(solutions,axis=0)[np.floor(age_data).astype(int)] 
    print("AVERAGE CI ACROSS DATA PTS: ", np.mean(intervals))
