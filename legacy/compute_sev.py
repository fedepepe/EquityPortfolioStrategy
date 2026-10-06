import sys
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scipy import stats
from sklearn.model_selection import LeaveOneOut

from definitions import CorrelationMethods


def gau_ker(x):
    return 1. / np.sqrt(2 * np.pi) * np.exp(- 0.5 * pow(x, 2))


def kernel_density(x, x_pts, bw):
    """
    x:     vector of input samples
    x_pts: vector of values where the density is to be calculated
    bw:    kernel bandwidth """
    n_obs = len(x)
    fx_hat = np.nan * x_pts
    for i in np.arange(len(x_pts)):
        ker_x = 1. / bw * gau_ker((x_pts[i] - x) / bw)
        fx_hat[i] = 1. / n_obs * sum(ker_x)
    return fx_hat


def kernel_density_support(x, bw):
    n_obs = len(x)
    grid_size = int(2 ** np.ceil(np.log2(10. * n_obs)))
    cut = 3.0
    a = np.min(x) - cut * bw
    b = np.max(x) + cut * bw
    x_pts, dx = np.linspace(a, b, grid_size, retstep=True)
    return x_pts, dx


def nada_wats_estim(x, y, x_pts, bw):
    fx_hat, phi_hat = np.nan * x_pts, np.nan * x_pts
    n_obs, n_pts = len(x), len(x_pts)
    for i in np.arange(n_pts):
        ker_x = 1. / bw * gau_ker((x_pts[i] - x) / bw)
        fx_hat[i] = 1. / n_obs * sum(ker_x)
        phi_hat[i] = 1. / n_obs * sum(ker_x * y)

    fx_hat[fx_hat == 0] = sys.float_info.min  # Replace zeros with minimum float number

    # Conditional expectation of Y given X, i.e., regression function
    # a.k.a. Nadaraya-Watson estimator
    g_hat = phi_hat / fx_hat
    return g_hat, fx_hat


def get_bw_scott(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Get a robust estimate of sigma
        sig = np.nanmedian(np.abs(x - np.nanmedian(x))) / 0.6745

    if sig <= 0:
        sig = max(x) - min(x)

    if sig > 0:
        n = len(x)
        return 1.059 * sig * n ** (-0.2)
    else:
        return np.nan


def get_optimum_cv_bandwidth(x, bw_ref, n_bw_points=21):
    bw_vec = np.logspace(np.log10(bw_ref / 10.), np.log10(bw_ref * 10.), n_bw_points)
    n_obs = len(x)

    emp_risk = np.array([0.0] * len(bw_vec))
    emp_risk_1 = np.array([0.0] * len(bw_vec))
    emp_risk_2 = np.array([0.0] * len(bw_vec))

    i = 0
    for bw in bw_vec:
        x_pts, dx = kernel_density_support(x, bw)
        fx_hat = kernel_density(x, x_pts, bw)
        emp_risk_1[i] = sum(pow(fx_hat, 2)) * dx

        # Cross-validation to find optimal bandwidth
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(x):
            x_train, x_test = x[train_index], x[test_index]
            fx_hat = kernel_density(x_train, x_test, bw)
            emp_risk_2[i] = emp_risk_2[i] + fx_hat

        # Compute empirical risk
        emp_risk[i] = emp_risk_1[i] - 2. / n_obs * emp_risk_2[i]
        i += 1

    # Select bandwidth associated to minimum empirical risk
    bw_opt = bw_vec[emp_risk.argmin()]
    return bw_opt


def compute_sev(x: np.array, y: np.array,
                method: CorrelationMethods = CorrelationMethods.SEV, cv_opt_bw=False):
    if method == CorrelationMethods.LINEAR:
        return np.corrcoef(x, y)[0, 1]
    elif method == CorrelationMethods.SPEARMAN:
        return stats.spearmanr(x, y).correlation
    elif method == CorrelationMethods.SEV:
        bw_scott = get_bw_scott(x)  # Scott's Rule of Thumb

        if np.isnan(bw_scott):
            return np.nan

        if cv_opt_bw:
            bw = get_optimum_cv_bandwidth(x, bw_scott)
        else:
            bw = bw_scott

        # Compute the support where the density of x is to be estimated
        x_pts, dx = kernel_density_support(x, bw)

        # Compute an estimate of g(x) = E[Y | X = x] via kernel regression
        [g_hat, fx_hat] = nada_wats_estim(x, y, x_pts, bw)

        return float((sum(g_hat ** 2 * fx_hat) * dx - pow(y.mean(), 2)) / y.var(ddof=1))
    else:
        raise Exception('Correlation method not implemented.')


def compute_sev_inner_fun(x_df: pd.DataFrame,
                          y_df: pd.DataFrame,
                          method: CorrelationMethods = CorrelationMethods.SEV,
                          cv_opt_bw: bool = False):
    # Suppress warning message caused by NaNs in the dataframes for kernel regression
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        sev = compute_sev(x_df.values, y_df.values, method, cv_opt_bw)
    return sev


def compute_sev_multiproc(features_df, hsr_idx, method, cv_opt_bw):
    pandarallel.initialize()
    sev_df = features_df.parallel_apply(compute_sev_inner_fun, args=(hsr_idx, method, cv_opt_bw)).to_frame()
    return sev_df


def get_chunks(lst, n):
    # Yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
