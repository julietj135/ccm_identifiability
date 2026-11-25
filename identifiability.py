#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:38:17 2024

@author: julietjiang
"""

'''
Identifiability analysis and posterior distribution sampling
'''

############ LOADING MODULES AND FUNCTIONS

import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os
from scipy.interpolate import interp1d
import math
from tqdm import tqdm
import random as r

# IMPORT DATA
from data import *
all_data = load_data()


# Source tables and parameter workbooks
df0 = all_data["df0"]          # ADNIMERGE clinical table
df = all_data["df"]            # PET and hippocampal measurements
df4 = all_data["df4"]          # Training parameter table
df5 = all_data["df5"]          # Identifiability and fitted parameters
df_fig = all_data["df_fig"]    # Identifiability colormap
uq_params = all_data["uq_params"]

# Subject IDs, diagnoses, and baseline information
RID = all_data["RID"]
RID_train = all_data["RID_train"]
DX = all_data["DX"]
DX_bl = all_data["DX_bl"]
age = all_data["age"]              # Baseline age by RID
age_time = all_data["age_time"]    # Earliest clinical exam date by RID
subjects = all_data["subjects"]

# Measurements and normalized biomarkers across all visits
hippocampus = all_data["hippocampus"]
ICV = all_data["ICV"]
raw_ratio = all_data["raw_ratio"]  # hippocampus / ICV
ratio = all_data["ratio"]          # Normalized and inverted ratio, same as N
N = all_data["N"]
ABETA = all_data["ABETA"]
PTAU = all_data["PTAU"]

# Quantiles used for normalization
r_lower, r_upper = all_data["r_lower"], all_data["r_upper"]
AB_lower, AB_upper = all_data["AB_lower"], all_data["AB_upper"]
T_upper = all_data["T_upper"]

# Summary matrices: row i corresponds to metric_rids[i]
metric_rids = all_data["metric_rids"]
metric_columns = all_data["metric_columns"]
abeta_data = all_data["abeta_data"]
tau_data = all_data["tau_data"]
neuro_data = all_data["neuro_data"]

# Each entry is a subject's number of identifiable parameters
cn_rid = all_data["cn_rid"]
mci_rid = all_data["mci_rid"]
ad_rid = all_data["ad_rid"]

#%%

############ PROFILE LIKELIHOOD

# choose unscaled model for profile likelihood search
import model as m
m.scale = False

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

    theta_opt = np.concatenate((theta_opt[0:theta_num], [x], theta_opt[theta_num:]), axis=0)
    return loss, theta_opt, ret

def get_one_endpoint(opt_param_vals, param_num, scan_bnds, bnds, step_num, loss_crit, loss_func, epsilon, log_bools):
    x_2 = opt_param_vals[param_num]
    x_1 = x_2
    theta_init = opt_param_vals
    theta_opt = opt_param_vals
    step = (scan_bnds[param_num][1]-scan_bnds[param_num][0])/step_num
    
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
                log_arr = np.logspace(math.log10(x_2-0.1*min([step,x_2])), math.log10(bound+2*step), num=step_num)
                id_curr = np.argmin(np.abs(log_arr-x_2))
                scaled_step = log_arr[id_curr+1] - log_arr[id_curr]
        else:
            scaled_step = step
            
        true_step = np.argmin([abs(step), abs(scaled_step)])
        
        loss = new_loss
        x_1 = x_2
        x_2 += [step,abs(scaled_step)*(abs(step)/step)][true_step]
        

def plot_profile(x_values, y_values, flags, inits, loss_crit, ep1s, ep2s, labels, title, error=False):
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
        
        axs[i].set_xlabel(labels[i], fontsize=20)
        axs[i].tick_params(axis='both', labelsize=16)
        axs[i].xaxis.get_offset_text().set_fontsize(16)
        axs[i].yaxis.get_offset_text().set_fontsize(16)
    
    axs[0].set_ylabel("Profile Likelihood", fontsize=16)
    
    plt.tight_layout(pad=1.12)
    # plt.savefig(title+".pdf")
    plt.show() 


bnds = np.array(((0,10), (0,1), (0,.2), 
                  (0,10), (0,1), (0,1), 
                  (0,10), (0,1), (0,1)))

all_param_names = [["KBETA","lambdaBETA","A_0"],["KPTAU",'lambdaPTAUBETA','lambdaPTAU'],["KN","lambdaNtp","lambdaN"]]
all_param_names = [[r'$K_{A_{\beta}}$',r'$\lambda_{A_{\beta}}$',r'$A_0$'],[r'$K_{\tau_{\rho}}$',r'$\lambda_{\tau_{\rho}A_{\beta}}$',r'$\lambda_{\tau_{\rho}}$'],[r'$K_N$',r'$\lambda_{N\tau_{\rho}}$',r'$\lambda_N$']]
iterations = 30


for rid in [4272]:  # change for specific subject
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
    abeta_lambda, tau_lambda, neuro_lambda = 1000, 1000, 1000
    identifiable = np.zeros((3,3))
    solve_params = False 
    plot_params = True
    find_identifiability = True 
    plot_uncertainty = False
    
    redo = False
    
    subject = all_data["subjects"][rid]
    has_fit = "hyperparams" in subject
    
    if has_fit and not redo:
        # Fit from the main identifiability workbook
        hp = subject["hyperparams"]  # pandas Series
    
        para_A = subject["para_A"].copy()
        para_T = subject["para_T"].copy()
        para_N = subject["para_N"].copy()
        identifiable = subject["identifiable"].copy()
    
        steps = hp.iloc[37:46].to_numpy(dtype=int).reshape(3, 3)
        scan_bnds = hp.iloc[19:37].to_numpy(dtype=float).reshape(9, 2)
        loss_A, loss_T, loss_N = hp.iloc[16:19].to_numpy(dtype=float)
        abeta_epsilon, tau_tol, neuro_epsilon = hp.iloc[1:4].to_numpy(dtype=float)
        abeta_lambda, tau_lambda, neuro_lambda = hp.iloc[4:7].to_numpy(dtype=float)
        epsilon = float(hp.iloc[55])
    
        solve_params = False
        find_identifiability = True
        plot_uncertainty = False
    
    elif not has_fit and rid in all_data["individual_params"]:
        # Fallback fit from the RID-specific workbook, already read by load_data()
        hp = all_data["individual_params"][rid][0]  # column labeled 0
    
        para_A = hp.iloc[7:10].to_numpy(dtype=float, copy=True)
        para_T = hp.iloc[10:13].to_numpy(dtype=float, copy=True)
        para_N = hp.iloc[13:16].to_numpy(dtype=float, copy=True)
        loss_A, loss_T, loss_N = hp.iloc[16:19].to_numpy(dtype=float)
        abeta_lambda, tau_lambda, neuro_lambda = hp.iloc[4:7].to_numpy(dtype=float)
    
        solve_params = False
        find_identifiability = True
        plot_uncertainty = False
    
    # Visit-level metadata
    PID = subject["PID"]
    PID_indexes = subject["PID_indexes"]
    clinical = subject["clinical"]
    dx_bl = subject["DX_bl"]       # This subject's baseline diagnosis
    visit_ages = subject["visit_ages"]
    
    # Full biomarker series, including visits with missing measurements
    ABETA_PID = subject["ABETA_PID"]
    PTAU_PID = subject["PTAU_PID"]
    N_PID = subject["N_PID"]
    
    # Available measurements paired with their corresponding visit ages
    ab, aab = subject["ab"], subject["aab"]   # Amyloid values and ages
    pt, apt = subject["pt"], subject["apt"]   # Tau values and ages
    nn, nnt = subject["nn"], subject["nnt"]   # Neurodegeneration values and ages
    
    # Fitted parameters and identifiability results
    hyperparams = subject["hyperparams"]
    identifiable = subject["identifiable"]
    derivatives = subject["derivatives"]
    n_id = subject["n_id"]
    para_A = subject["para_A"]
    para_T = subject["para_T"]
    para_N = subject["para_N"]
    para_AT = subject["para_AT"]
    
    print(ab,aab)
    
    # can re-optimize parameters
    if solve_params:        
        initials = (1, 0.12, 0.010)
        para_A, loss_A, opt_info = opt.fmin_l_bfgs_b(m.para_est_obj_A, x0=initials, args=(ab,aab,abeta_lambda), approx_grad=True, bounds=bnds[:3], epsilon=abeta_epsilon, maxiter=500)
        print(opt_info, loss_A)
        minloss = loss_A 
        grada = []
         
        for i in range(0,iterations): 
            r1 = r.uniform(.8*para_A[0],1.2*para_A[0]) 
            r2 = r.uniform(.8*para_A[1],1.2*para_A[1]) 
            r3 = r.uniform(.8*para_A[2],1.2*para_A[2]) 
            initials = [r1, r2, r3] 
            x_opt, loss_A, opt_info = opt.fmin_l_bfgs_b(m.para_est_obj_A, x0=initials, args=(ab,aab,abeta_lambda), approx_grad=True, bounds=bnds[:3], epsilon=abeta_epsilon, maxiter=500)
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
            try:
                res = opt.minimize(m.para_est_obj_T, initials, args=(para_A,pt,apt,tau_lambda), method='trust-constr', hess=None, hessp=None, bounds=bnds[3:6], tol=tau_tol, options={'initial_tr_radius': 0.01})
                loss_T = m.para_est_obj_T(res.x,para_A,pt,apt,tau_lambda)
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
        x_opt, y_opt, opt_info = opt.fmin_l_bfgs_b(m.para_est_obj_N, x0=initials, args=(para_AT,nn,nnt,neuro_lambda), approx_grad=True, bounds=bnds[6:], epsilon=neuro_epsilon, maxiter=500)
        loss_N = m.para_est_obj_N(x_opt,para_AT,nn,nnt,neuro_lambda)
        minloss = loss_N
        para_N = x_opt 
            
        for i in range(1,8):
            initials = [i,0.025,0.2]
            x_opt, y_opt, opt_info = opt.fmin_l_bfgs_b(m.para_est_obj_N, x0=initials, args=(para_AT,nn,nnt,neuro_lambda), approx_grad=True, bounds=bnds[6:], epsilon=neuro_epsilon, maxiter=500)
            print(x_opt,y_opt)
            print(opt_info)
            loss_N = m.para_est_obj_N(x_opt,para_AT,nn,nnt,neuro_lambda)
            if (loss_N < minloss and loss_N > 0):
                print("STORE")
                minloss = loss_N
                para_N = x_opt 
                gradn = opt_info["grad"]
                optinfo = opt_info
                
        loss_N = minloss
            
        print("final loss_N", loss_N)
        print("final para_N", para_N)
    
    para_AT = np.concatenate([para_A,para_T])
    
    
    def plot():
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        num=0

        t, a_traj = m.plot_para_est_obj_A(para_A, ab, aab)
        t, t_traj = m.plot_para_est_obj_T(para_T, para_A, pt, apt)
        t, n_traj = m.plot_para_est_obj_N(para_N, para_AT, nn, nnt)
        ax = axs
        ax.plot(aab, ab, '.r', label="abeta")
        ax.plot(apt, pt, '.b', label="tau")
        ax.plot(nnt, nn, '.y', label="neurodegen")
        ax.plot(t, a_traj, '-r')
        ax.plot(t, t_traj, '-b')
        ax.plot(t, n_traj, '-y')
        ax.set_title(rid)
        
    if plot_params:
        subject = all_data["subjects"][rid]
        fig, ax = m.plot_subject_trajectories(subject, subject["para_A"], subject["para_T"], subject["para_N"])
    
    def generate_synthetic(ab, aab, pt, apt, nn, nnt, NUM):
        t, a_traj = m.plot_para_est_obj_A(para_A, ab, aab)
        t, t_traj = m.plot_para_est_obj_T(para_T, para_A, pt, apt)
        t, n_traj = m.plot_para_est_obj_N(para_N, para_AT, nn, nnt)
        
        t_interp = np.linspace(0, 1, len(t))  
        interp_x = interp1d(t_interp, t, kind='cubic')
        interp_a = interp1d(t_interp, a_traj, kind='cubic')
        interp_t = interp1d(t_interp, t_traj, kind='cubic')
        interp_n = interp1d(t_interp, n_traj, kind='cubic')
        
        random_t = np.sort(np.random.rand(NUM))
        aab, apt, nnt = interp_x(random_t), interp_x(random_t), interp_x(random_t)
        ab, pt, nn = interp_a(random_t), interp_t(random_t), interp_n(random_t)
        
        return ab, aab, pt, apt, nn, nnt
    
    def matrix_rank(data):
        A_mat_T = np.vstack((data,(-data**2)))
        A_mat = A_mat_T.T
        
        ATA = np.matmul(A_mat_T,A_mat)
        
        print(ATA)
        print("rank: ", ATA.ndim)
        
        return ATA

    opt_param_vals = np.array([para_A,para_T,para_N])
    
    print(opt_param_vals)
    
    a_loss_crit = chi2.ppf(0.95, df=1) + m.para_est_obj_A(opt_param_vals[0], ab, aab, abeta_lambda)
    t_loss_crit = chi2.ppf(0.95, df=1) + m.para_est_obj_T(opt_param_vals[1], opt_param_vals[0], pt, apt, tau_lambda)
    n_loss_crit = chi2.ppf(0.95, df=1) + m.para_est_obj_N(opt_param_vals[2], np.concatenate([opt_param_vals[0],opt_param_vals[1]]), nn, nnt, neuro_lambda)
    
    
    print(a_loss_crit, t_loss_crit, n_loss_crit)
    
    # continue
    
    all_theta_opts = [[[],[],[]],[[],[],[]],[[],[],[]]]
    all_eps = [[],[]]
    
    # profile likelihood searching 
    if find_identifiability:
        bio_losses = [a_loss_crit, t_loss_crit, n_loss_crit]
        bio_objs = [lambda x: m.para_est_obj_A(x, ab, aab, abeta_lambda), lambda x: m.para_est_obj_T(x, para_A, pt, apt, tau_lambda), lambda x: m.para_est_obj_N(x, np.concatenate([para_A,para_T]), nn, nnt, neuro_lambda)]
        for bio in [0,1,2]:
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
            
            plot_profile(x_values, y_values, flags, opt_param_vals[0], bio_losses[bio], ep1s, ep2s, all_param_names[bio], title="profile_like_{}".format(["A","P","N"][bio]), error=False)
    
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

# choose scaled model for uncertainty quantification
import model as m
m.scale = True

from data import *
uq_params = load_uq_params()
all_data["uq_params"] = uq_params 

# parameters have been scaled to (0,1)
bnds = np.array(((0,1), (0,1), (0,1), 
                  (0,1), (0,1), (0,1), 
                  (0,1), (0,1), (0,1)))

all_param_names = [[r'$K_{A_{\beta}}$',r'$\lambda_{A_{\beta}}$',r'$A_0$'],[r'$K_{\tau_{\rho}}$',r'$\lambda_{\tau_{\rho}A_{\beta}}$',r'$\lambda_{\tau_{\rho}}$'],[r'$K_N$',r'$\lambda_{N\tau_{\rho}}$',r'$\lambda_N$']]

iterations = 30

def log_likelihood_MALA(x, biomarker, data, age_data, additional_params, lambda_bio):
    if biomarker == 0:
        loglik = m.para_est_obj_A(x, data, age_data, lambda_bio)
    elif biomarker == 1:
        loglik = m.para_est_obj_T(x, additional_params, data, age_data, lambda_bio)
    else:
        loglik = m.para_est_obj_N(x, additional_params, data, age_data, lambda_bio)

    return loglik/(-2)

def logpost(x, biomarker, data, 
            age_data, additional_params, 
            param_num, optimal_vars, scale, bnds, df5, lambda_bio):
    
    x = np.asarray(x, dtype=float).reshape(-1)
    center = optimal_vars[biomarker][param_num]

    new_params = optimal_vars[biomarker].copy()
    new_params[param_num] = x
    
    ##### Gaussian prior around optimal values
    logprior = -0.5 * np.sum((x - center) ** 2) / scale
    
    loglik = log_likelihood_MALA(new_params, biomarker, data, age_data, additional_params, lambda_bio)
        
    if not np.isfinite(loglik) or loglik > 0:
        return -np.inf
    
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
         age_data, additional_params, optimal_vars, scale, initial, p, df5, lambda_bio
         ):
    out = np.zeros((num_samples, p))
    params = np.array(initial).reshape((p,1))
    # epsilon = epsilon.reshape((p,1)) # coord MCMC
    epsilon = float(epsilon) # full MCMC
    
    A = cholesky(S, lower=True)
    S1 = np.linalg.inv(S)
    
    lgrad = S @ lgradient(params, biomarker, data, 
                age_data, additional_params, param_num, optimal_vars, scale, df5, lambda_bio)

    logp = logpost(params, biomarker, data, 
                age_data, additional_params, param_num, 
                optimal_vars, scale, bnds, df5, lambda_bio)

    # sigma2 = epsilon**2 / p**(1 / 3) # coord MCMC
    sigma2 = epsilon**2 # full MCMC
    sigma = np.sqrt(sigma2)
    
    counts = 0
    acceptance_rate = 0
    for rn in tqdm(range(burn_in + num_samples), desc="MCMC Sampling Progress"):
        new_params = params + sigma2 / 2 * lgrad + sigma * (A @ np.random.normal(size=p)[:, np.newaxis])

        logpnew = logpost(new_params, biomarker, data, 
                    age_data, additional_params, param_num, 
                    optimal_vars, scale, bnds, df5, lambda_bio)
        lgrad_new = S @ lgradient(new_params, biomarker, data, 
                    age_data, additional_params, param_num, optimal_vars, scale, df5, lambda_bio)
        
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

    # return out[out != 0], counts # coord MCMC
    return out, counts


def MALA_continue(num_samples, counts, epsilon, S,
         biomarker, data, bnds, param_num,
         age_data, additional_params, optimal_vars, scale, initial, p, df5, lambda_bio
         ):
    out = np.zeros((num_samples, p))
    params = np.array(initial).reshape((p,1))
    epsilon = epsilon.reshape((p,1))
    
    lgrad = S @ lgradient(params, biomarker, data, 
                age_data, additional_params, param_num, optimal_vars, scale, df5, lambda_bio)

    logp = logpost(params, biomarker, data, 
                age_data, additional_params, param_num, 
                optimal_vars, scale, bnds, df5, lambda_bio)
    
    A = cholesky(S, lower=True)
    S1 = np.linalg.inv(S)

    sigma2 = epsilon**2 / p**(1 / 3)
    sigma = np.sqrt(sigma2)
    
    for rn in tqdm(range(num_samples), desc="MCMC Sampling Progress"):
        new_params = params + sigma2 / 2 * lgrad + sigma * (A @ np.random.normal(size=p)[:, np.newaxis])

        logpnew = logpost(new_params, biomarker, data, 
                    age_data, additional_params, param_num, 
                    optimal_vars, scale, bnds, df5, lambda_bio)
        lgrad_new = S @ lgradient(new_params, biomarker, data, 
                    age_data, additional_params, param_num, optimal_vars, scale, df5, lambda_bio)
        
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
                        
            counts+=1
            out[rn, :] = params.flatten()

    return out[out != 0], counts

np.random.seed(42)
    
for rid in [4272]: 
    print(rid)
    
    subject = subjects[rid]
    
    # CHANGE ME, which biomarker
    bio = 1

    # Visit-level metadata
    PID = subject["PID"]
    PID_indexes = subject["PID_indexes"]
    clinical = subject["clinical"]
    dx_bl = subject["DX_bl"]       # This subject's baseline diagnosis
    visit_ages = subject["visit_ages"]
    
    # Full biomarker series, including visits with missing measurements
    ABETA_PID = subject["ABETA_PID"]
    PTAU_PID = subject["PTAU_PID"]
    N_PID = subject["N_PID"]
    
    # Available measurements paired with their corresponding visit ages
    ab, aab = subject["ab"], subject["aab"]   # Amyloid values and ages
    pt, apt = subject["pt"], subject["apt"]   # Tau values and ages
    nn, nnt = subject["nn"], subject["nnt"]   # Neurodegeneration values and ages
    
    # Fitted parameters and identifiability results
    hyperparams = subject["hyperparams"]
    identifiable = subject["identifiable"]
    derivatives = subject["derivatives"]
    n_id = subject["n_id"]
    para_A = subject["para_A"].copy()
    para_T = subject["para_T"].copy()
    para_N = subject["para_N"].copy()
    lambdas = list(hyperparams[4:7])
    
    # scaling parameters to be between 0 and 1
    para_A[0] /= 10
    para_T[0] /= 10
    para_N[0] /= 10
    para_A[2] = (math.log(para_A[2])+35)/33.4
    para_AT = np.concatenate((para_A, para_T))
    
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
        
    row = uq_params.loc[rid]

    sampling_cols = [f"{param_name}_{key}" for key in ("num_samples", "thinning", "burn_in", "switching")]
    num_samples, thinning, burn_in, switching = row[sampling_cols].to_numpy(dtype=int)
    
    epsilon_cols = [f"epsilon_{param_name}{i}" for i in range(3)]
    bio_epsilon = row[epsilon_cols].to_numpy(dtype=float)
    para = np.flatnonzero(bio_epsilon)
    epsilons = bio_epsilon[para]
        
    def plot():
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        num=0

        t, a_traj = m.plot_para_est_obj_A(para_A, ab, aab)
        t, t_traj = m.plot_para_est_obj_T(para_T, para_A, pt, apt)
        t, n_traj = m.plot_para_est_obj_N(para_N, para_AT, nn, nnt)
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
    
    MALA_ = True
    
    # RUN MCMC
    print("total samples, thinning, burnin, switching")
    print(num_samples, thinning, burn_in, switching)
    num_samples = 12000
    thinning = 5
    burn_in = 100
    # switching = 1000
    p = len(para)
    scale = 0.1
    
    if MALA_:
        initial = opt_param_vals[bio][para].copy()
        
        print('Starting MCMC on {}'.format(all_param_names[bio][para]))
        print("epsilon: ", epsilons, "initial: ", initial)
        print("bounds: ", bnds[bio*3+para])
        print("DATA: ", data, age_data)
        
        # COORD MCMC
        # current_iter = np.zeros(len(para)).astype(int)
        # full_MCMC = np.zeros((num_samples,len(para)))
        # temp_MCMC = np.array([opt_param_vals[bio][para]]*switching)
        # individual_counts = np.zeros(len(para)).astype(int)
        # temp_opt_param_vals = opt_param_vals.copy()
        # while (current_iter < np.ones(len(para)) * num_samples).all():
        #     for idx, coord in enumerate(para):
        #         print(coord, current_iter)
        #         initial = temp_MCMC[-1,idx]
        #         p = 1
        #         S = np.eye(p)
        #         epsilon = bio_epsilon[coord]
        #         coord_MCMC, counts = MALA(switching, 0, epsilon, S,
        #                   bio, data, bnds[bio*3+coord].reshape(1,2), [coord], 
        #                   age_data, extra, opt_param_vals, scale, initial, p, df5, lambdas[bio]
        #                   )
        #         individual_counts[idx] += counts
        #         current_iter[idx] += switching
        #         full_MCMC[current_iter[idx]-switching:current_iter[idx], idx] = coord_MCMC
        #         temp_MCMC[:,idx] = coord_MCMC
        #         temp_opt_param_vals[bio][coord] = coord_MCMC[-1]
        # print("Acceptance rate of last MCMC chain: ", individual_counts/num_samples)
                
        # FULL MCMC
        epsilon = 1.0
        S = np.diag(epsilons ** 2)
        
        full_MCMC, accepted = MALA(
            num_samples, 0, epsilon, S,
            bio, data, bnds[3 * bio + para], para,
            age_data, extra, opt_param_vals, scale, initial, p, df5, lambdas[bio],
        )
        
        print("Sampled parameter indices:", para)
        print("Joint acceptance rate:", accepted / num_samples)
        
        # MCMC = full_MCMC[None, :, :]  # (1 chain, num_samples, len(para))
        # joint_draws = full_MCMC[burn_in::thinning]
        
        MCMC_flattened = np.vstack([full_MCMC, full_MCMC])
        MCMC = MCMC_flattened.reshape((2,num_samples,len(para)))
        
        print("Shape of MCMC stack: ", MCMC.shape)
        
        for i, j in zip(np.arange(len(para)), para):
            ess = effective_n(MCMC[...,i], [all_param_names[bio][j]])
            print("Effective sample size for param {}: ".format(ess))
            
            autocorrelation_int = num_samples/ess[all_param_names[bio][j]]
            
            print("Integrated autocorrelation time: ", autocorrelation_int)
        
        
    for i, j in zip(np.arange(len(para)), para):
        plt.figure(figsize=(4,4))
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
        formatter.set_powerlimits((-2, 2))  # force scientific notation outside this range
        ax.xaxis.set_major_formatter(formatter)
        
        # for figures on illustrator
        plt.xlabel('{}'.format(all_param_names[bio][j]),fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        ax.xaxis.get_offset_text().set_fontsize(16)
        ax.yaxis.get_offset_text().set_fontsize(16)
        plt.tight_layout()
        
        # plt.savefig("posteriors/{}_{}_{}.pdf".format(rid,bio_name[0],j))
        plt.show()
     
        # plot trace in the second figure
        plt.figure(figsize=(4,4))
        plt.plot(MCMC[0,...,i][burn_in::thinning],'k')
        plt.xlabel('{}'.format(all_param_names[bio][j]))
        plt.ylabel("trace")
        plt.show()
    
        autocorrplot(MCMC[...,i][:,burn_in::thinning], max_lag=100, varnames=[all_param_names[bio][j]])
    
        # MC propagation of uncertainty
        sampled_xs = np.random.choice(MCMC[0,...,i][burn_in::thinning], size=1000, replace=True)
        lower_bound = np.percentile(sampled_xs, 2.5)
        upper_bound = np.percentile(sampled_xs, 97.5)
        
        sampled_xs = sampled_xs[(sampled_xs >= lower_bound) & (sampled_xs <= upper_bound)]
    
        solutions = []
        for x in sampled_xs:
            temp = opt_param_vals[bio].copy()
            temp[j] = x
            if bio == 0:
                t, traj = m.plot_para_est_obj_A(temp, ab, aab)
            elif bio == 1:
                t, traj = m.plot_para_est_obj_T(temp, para_A, pt, apt)
            else:
                t, traj = m.plot_para_est_obj_N(temp, para_AT, nn, nnt)
            solutions.append(traj)
            
        # compute mean and confidence intervals
        solutions = np.array(solutions)
        mean_sol = np.mean(solutions, axis=0)
        std_sol = np.std(solutions, axis=0)
        
        # plt.figure(figsize=(4,4))
        # plt.plot(t, mean_sol, label="mean trajectory", color="k")
        # plt.fill_between(t, np.min(solutions,axis=0), np.max(solutions,axis=0), color=color, alpha=0.3, label="95% Confidence Interval")
        # plt.scatter(age_data, data, color=color, marker='x', label="biomarker data")
        # plt.xlabel("age")
        # plt.ylabel(bio_name)
        # plt.legend()
        # plt.show()

    
    # MC propagation of uncertainty, JOINT
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
            t, traj = m.plot_para_est_obj_A(temp, ab, aab)
        elif bio == 1:
            t, traj = m.plot_para_est_obj_T(temp, para_A, pt, apt)
        else:
            t, traj = m.plot_para_est_obj_N(temp, para_AT, nn, nnt)
        solutions.append(traj)
    
    # compute mean and confidence intervals
    solutions = np.array(solutions)
    mean_sol = np.mean(solutions, axis=0)
    std_sol = np.std(solutions, axis=0)
    
    plt.figure(figsize=(4,4))
    plt.plot(t, mean_sol, label="Mean Trajectory", color="k")
    plt.fill_between(t, np.min(solutions,axis=0), np.max(solutions,axis=0), color=color, alpha=0.3, label="95% Confidence Interval")
    plt.scatter(age_data, data, color=color, marker='x', label="Data")
    plt.xlabel("Age (Years)")
    plt.ylabel(bio_name)
    plt.legend()
    
    ## for figures on illustrator
    # plt.plot(t, mean_sol, label="Mean Trajectory", color="k", linewidth=2)
    # plt.scatter(age_data, data, color=color, marker='x', label="Data", s=100)
    # plt.xlabel("Age (Years)",fontsize=16)
    # plt.ylabel(bio_name,fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.tight_layout()
    
    # plt.savefig("uncertainties/{}_{}.pdf".format(rid,bio_name[0]))
    plt.show()
    
    
    # calculate average confidence interval across data points
    intervals = np.max(solutions,axis=0)[np.floor(age_data).astype(int)] - np.min(solutions,axis=0)[np.floor(age_data).astype(int)] 
    print("AVERAGE CI ACROSS DATA PTS: ", np.mean(intervals))
    
    ###################################
    
    # UQ with perturbed non-identifiable params 
    # Sensitivity of the mean interval width to each fixed parameter
    n_trials = 10
    lower_perc, upper_perc = 0.5, 1.5
    fitted = opt_param_vals[bio].copy()
    sweep_values = {
        0: np.linspace(lower_perc * fitted[0], upper_perc * fitted[0], n_trials),
        # 1: np.linspace(lower_perc * fitted[1], upper_perc * fitted[1], n_trials),
        2: np.linspace(lower_perc * fitted[2], upper_perc * fitted[2], n_trials),
        # 2: np.linspace(0, 0.2, n_trials),
    }
    # sweep_values = {
    #     0: np.linspace(0, 1, n_trials),
    #     1: np.linspace(0, 1, n_trials),
    #     2: np.linspace(0, 1, n_trials),
    #     # 2: np.linspace(0, 0.2, n_trials),
    # }
    visit_idx = np.floor(age_data).astype(int)
    draws = np.unique(sampled_xs[:, 0])
    mean_widths = {}
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    # fig, axes = plt.subplots(1, 1, figsize=(3, 3), sharey=True)
    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    for ax, (j, values) in zip(axes, sweep_values.items()):
        mean_widths[j] = np.empty(len(values))
        for trial, value in enumerate(values):
            temp = fitted.copy()
            temp[j] = value
            at_visits = []
            for draw in draws:
                temp[1] = draw
                if bio == 0:
                    _, traj = m.plot_para_est_obj_A(temp, ab, aab)
                elif bio == 1:
                    _, traj = m.plot_para_est_obj_T(temp, para_A, pt, apt)
                else:
                    _, traj = m.plot_para_est_obj_N(temp, para_AT, nn, nnt)
                # at_visits.append(traj[visit_idx]) # at data points
                at_visits.append(traj[-1]) # at constraint only
            width_at_each_visit = np.max(at_visits, axis=0) - np.min(at_visits, axis=0)
            mean_widths[j][trial] = width_at_each_visit.mean()

        ax.plot(values, mean_widths[j], ".-", color=color)
        ax.axvline(fitted[j], color="k", linestyle="--", label="Fitted value")
        ax.set_xlabel(all_param_names[bio][j] + (" (scaled)" if j == 0 or bio == 0 else ""))

    axes[0].set_ylabel("Mean interval width at observed ages")
    fig.suptitle(f"RID {rid} — {bio_name}")
    fig.tight_layout()
    plt.show()
    
    

#%%

###############################

# logistic regressions
# 0 - mean data
# 1 - median data
# 2 - standard dev data
# 3 - range data
# 4 - max data
# 5 - min data
# 6 - mean age
# 7 - median age
# 8 - standard dev age
# 9 - range age
# 10 - max age
# 11 - min age
# 12 - derivative

import analysis as analysis

# Run one analysis
fig = analysis.plot_identifiability_by_diagnosis(data, output_dir="figures")

# Or run all nine
results = analysis.run_post_analysis(data, output_dir="figures")

