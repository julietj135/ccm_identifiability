# -*- coding: utf-8 -*-

# MCMC diagnostics adapted from legacy PyMC3 for NumPy chain arrays
# see https://pysgmcmc.readthedocs.io/en/latest/index.html for more

import itertools
import matplotlib.pyplot as plt
import numpy as np

def effective_n(mtrace, varnames):
    R"""Returns estimate of the effective sample size of a set of traces.

    Parameters
    ----------
    mtrace : MultiTrace
        A MultiTrace object containing parallel traces (minimum 2)
        of one or more stochastic parameters.

    Returns
    -------
    n_eff : float
        Return the effective sample size, :math:`\hat{n}_{eff}`

    Notes
    -----
    The diagnostic is computed by:

    .. math:: \hat{n}_{eff} = \frac{mn}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}

    where :math:`\hat{\rho}_t` is the estimated autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` is negative.

    References
    ----------
    Gelman et al. (2014)"""

    if mtrace.shape[0] < 2:
        raise ValueError(
            'Calculation of effective sample size requires multiple chains '
            'of the same length.')

    def get_vhat(x):
        # number of chains is last dim (-1)
        # chain samples are second to last dim (-2)
        num_samples = x.shape[-2]

        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=-2), axis=-1, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=-2, ddof=1), axis=-1)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        return Vhat

    def get_neff(x, Vhat):
        num_chains = x.shape[-1]
        num_samples = x.shape[-2]

        negative_autocorr = False
        t = 1

        rho = np.ones(num_samples)
        # Iterate until the sum of consecutive estimates of autocorrelation is
        # negative
        while not negative_autocorr and (t < num_samples):

            variogram = np.mean((x[t:, :] - x[:-t, :])**2)
            rho[t] = 1. - variogram / (2. * Vhat)

            negative_autocorr = sum(rho[t - 1:t + 1]) < 0

            t += 1

        if t % 2:
            t -= 1

        return min(num_chains * num_samples,
                   int(num_chains * num_samples / (1. + 2 * rho[1:t-1].sum())))

    n_eff = {}
    for var in varnames:
        x = np.array(mtrace)

        # make sure to handle scalars correctly - add extra dim if needed
        if len(x.shape) == 2:
            is_scalar = True
            x = np.atleast_3d(mtrace)
        else:
            is_scalar = False

        # now we are going to transpose all dims - makes the loop below
        # easier by moving the axes of the variable to the front instead
        # of the chain and sample axes
        x = x.transpose()

        Vhat = get_vhat(x)

        # get an array the same shape as the var
        _n_eff = np.zeros(x.shape[:-2])

        # iterate over tuples of indices of the shape of var
        for tup in np.ndindex(*list(x.shape[:-2])):
            _n_eff[tup] = get_neff(x[tup], Vhat[tup])

        # we could be using np.squeeze here, but we don't want to squeeze
        # out dummy dimensions that a user inputs
        if is_scalar:
            n_eff[var] = _n_eff[0]
        else:
            # make sure to transpose the dims back
            n_eff[var] = np.transpose(_n_eff)

    return n_eff


def autocorrplot(trace, varnames, max_lag=50, burn=0, plot_transformed=False,
                 symmetric_plot=False, ax=None, figsize=None):
    """Bar plot of the autocorrelation function for a trace.

    Parameters
    ----------
    trace : result of MCMC run
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted.
        Vector-value stochastics are handled automatically.
    max_lag : int
        Maximum lag to calculate autocorrelation. Defaults to 100.
    burn : int
        Number of samples to discard from the beginning of the trace.
        Defaults to 0.
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).
    symmetric_plot : boolean
        Plot from either [0, +lag] or [-lag, lag]. Defaults to False, [-, +lag].
    ax : axes
        Matplotlib axes. Defaults to None.
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inches.
        Note this is not used if ax is supplied.

    Returns
    -------
    ax : matplotlib axes
    """
    # def _handle_array_varnames(varname):
    #     if trace[0].__class__ is np.ndarray:
    #         k = trace.shape[1]
    #         for i in range(k):
    #             yield varname + '_{0}'.format(i)
    #     else:
    #         yield varname

    # varnames = list(itertools.chain.from_iterable(map(_handle_array_varnames, varnames)))

    nchains = trace.shape[0]

    if figsize is None:
        figsize = (12, len(varnames) * 2)

    _, ax = plt.subplots(len(varnames),nchains)

    max_lag = min(len(trace[0]) - 1, max_lag)
    print("max_lag: ", max_lag)

    for i, v in enumerate(varnames):
        for j in range(nchains):
            try:
                d = np.squeeze(trace[j])
            except KeyError:
                k = int(v.split('_')[-1])
                v_use = '_'.join(v.split('_')[:-1])
                d = np.squeeze(trace[j])
            
            ax[j].acorr(d, detrend=plt.mlab.detrend_mean, maxlags=max_lag)

            if j == 0:
                ax[j].set_ylabel("correlation")

            if i == len(varnames) - 1:
                ax[j].set_xlabel("lag")

            ax[j].set_title(v)

            if not symmetric_plot:
                ax[j].set_xlim(0, max_lag)

            if nchains > 1:
                ax[j].set_title("{} chain {}".format(v, j + 1))
    
    # plt.savefig("{}_{}.png".format(varnames[0],trace.shape[-1]))

    return ax

def get_axis(ax, default_rows, default_columns, **default_kwargs):
    """Verifies the provided axis is of the correct shape, and creates one if needed.

    Args:
        ax: matplotlib axis or None
        default_rows: int, expected rows in axis
        default_columns: int, expected columns in axis
        **default_kwargs: keyword arguments to pass to plt.subplot

    Returns:
        axis, or raises an error
    """

    default_shape = (default_rows, default_columns)
    if ax is None:
        _, ax = plt.subplots(*default_shape, **default_kwargs)
    elif ax.shape != default_shape:
        raise ValueError('Subplots with shape %r required' % (default_shape,))
    return ax
