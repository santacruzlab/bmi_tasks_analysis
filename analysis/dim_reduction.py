'''
Dimensionality reduction methods
'''
from sklearn.decomposition import PCA
import numpy as np
import trial_filter_functions, trial_proc_functions, trial_condition_functions

def reduce_dim_pca(Kyt, dims=slice(None, None)):
    '''
    Use PCA to sub-select dimensions of a (ndim, nsamples) array
    '''
    pca = PCA()
    pca.fit(Kyt.T)
    Kyt_pccoords = pca.transform(Kyt.T)
    Kyt_pccoords_ld = np.zeros_like(Kyt_pccoords)
    Kyt_pccoords_ld[:, dims] = Kyt_pccoords[:, dims]
    return pca.inverse_transform(Kyt_pccoords_ld).T

def logdet(x, bias=0, **kwargs):
    '''
    Return the log-determinant of the covariance matrix of x

    x: np.ndarray of shape (n_features, n_samples)
        Data samples to calculate the "volume" of

    Returns
    -------
    log det cov(x)
    '''
    if not isinstance(x, np.ndarray):
        x = np.hstack(x)
    return np.linalg.slogdet(np.cov(x, bias=bias, **kwargs))[1]

def tracevol(x, bias=0, **kwargs):
    '''
    Return the log-determinant of the covariance matrix of x

    x: np.ndarray of shape (n_features, n_samples)
        Data samples to calculate the "volume" of

    Returns
    -------
    log det cov(x)
    '''
    if not isinstance(x, np.ndarray):
        x = np.hstack(x)
    c = np.cov(x, bias=bias, **kwargs)
    return np.trace(c)

def Kyt_pca_ratio(col):
    Kyt_data = col.proc_trials(trial_filter_fn=trial_filter_functions.target_in_trial, trial_proc_fn=trial_proc_functions.Kyt_during_target_state, data_comb_fn=np.hstack)

    expl_var_ratio = []
    expl_var = []
    for Kyt in Kyt_data:
        pca = PCA()
        pca.fit(Kyt[4:8,:].T)
        expl_var_ratio.append(pca.explained_variance_ratio_)
        expl_var.append(pca.explained_variance_)
    return np.vstack(expl_var), np.vstack(expl_var_ratio)

