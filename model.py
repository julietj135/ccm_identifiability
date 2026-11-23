"""ADNI model for amyloid, tau, and neurodegeneration. Set scale before fitting or plotting."""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# True for scaled MCMC coordinates, False for unscaled fitted parameters.
scale = True


def _capacity(value):
    return 10 * value if scale else value


def _initial_amyloid(value):
    return math.exp(33.4 * value - 35) if scale else value


def Right_hand_side_A(t, u, x):
    ABETA = u[0]
    KBETA = x[0]
    lambdaBETA = x[1]
    a = lambdaBETA*ABETA*(_capacity(KBETA) - ABETA)
    return a


def Right_hand_side_T(t, u, x, para_A):
    ABETA = u[0]
    PTAU = u[1]
    KBETA = para_A[0]
    lambdaBETA = para_A[1]
    KPTAU = x[0]
    lambdaPTAUBETA = x[1]
    lambdaPTAU = x[2]
    a = lambdaBETA * ABETA * (_capacity(KBETA) - ABETA)
    t = lambdaPTAUBETA*ABETA + lambdaPTAU*PTAU*(_capacity(KPTAU) - PTAU)
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
    a = lambdaBETA * ABETA * (_capacity(KBETA) - ABETA)
    t = lambdaPTAUBETA * ABETA + lambdaPTAU * PTAU * (_capacity(KPTAU) - PTAU)
    n = lambdaNtp * PTAU + lambdaN * N * (_capacity(KN) - N)
    return np.array([a, t, n])


def para_est_obj_A(x, ABETA_data, AGE_dataA, abeta_lambda):
    TimeA = AGE_dataA
    initial = [_initial_amyloid(x[2])]
    t_span = (0,100)
    t = np.arange(0.0, 101, 1)

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
    initial = np.array([_initial_amyloid(para_A[2]), 0])
    t_span = np.array([0,100])
    t = np.arange(0.0, 101, 1)

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
    initial = np.array([_initial_amyloid(para_AT[2]), 0, 0])
    t_span = np.array([0,100])
    t = np.arange(0.0, 101, 1)

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


def plot_para_est_obj_A(x, ABETA_data, AGE_dataA):
    TimeA = AGE_dataA
    initial = [_initial_amyloid(x[2])]
    t_span = (0,100)
    t = np.arange(0.0, 101, 1)
    results = solve_ivp(Right_hand_side_A, t_span, initial, t_eval = t, args=(x,), rtol = 1e-5)
    TimeA = np.floor(TimeA).astype(int)
    return [t, results.y[0]]


def plot_para_est_obj_T(x, para_A, PTAU_data, AGE_dataP):
    TimeP = AGE_dataP
    initial = np.array([_initial_amyloid(para_A[2]), 0])
    t_span = np.array([0,100])
    t = np.arange(0.0, 101, 1)
    results = solve_ivp(Right_hand_side_T, t_span, initial, t_eval = t, args=(x, para_A,), rtol = 1e-5)
    TimeP = np.floor(TimeP).astype(int)
    return [t, results.y[1]]


def plot_para_est_obj_N(x, para_AT, N_data, AGE_dataN):
    TimeN = AGE_dataN
    initial = np.array([_initial_amyloid(para_AT[2]), 0, 0])
    t_span = np.array([0,100])
    t = np.arange(0.0, 101, 1)
    results = solve_ivp(Right_hand_side_N, t_span, initial, t_eval = t, args=(x, para_AT), rtol = 1e-5)
    TimeN = np.floor(TimeN).astype(int)
    return [t, results.y[2]]


def plot_subject_trajectories(subject, para_A, para_T, para_N):
    """Overlay the three trajectories on visits from load_data()["subjects"][rid].

    Supply parameters in this module's convention. Return the figure and axes.
    """
    para_AT = np.concatenate([para_A, para_T])
    fig, ax = plt.subplots(figsize=(5, 5))
    for values, ages, solve, params, color, label in (
        ("ab", "aab", plot_para_est_obj_A, (para_A,), "r", "abeta"),
        ("pt", "apt", plot_para_est_obj_T, (para_T, para_A), "b", "tau"),
        ("nn", "nnt", plot_para_est_obj_N, (para_N, para_AT), "y", "neurodegen"),
    ):
        t, trajectory = solve(*params, subject[values], subject[ages], 1)
        ax.plot(subject[ages], subject[values], "." + color, label=label)
        ax.plot(t, trajectory, "-" + color)
    ax.set_title(str(subject["PID"]["RID"].iloc[0]))
    return fig, ax
