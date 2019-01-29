#!/usr/bin/python
'''
Statistics functions 
'''
import numpy as np
from scipy.stats import pearsonr
from scipy import stats


def pcc(xdata, ydata, zdata):
    '''
    Calculate partial correlation between x and y, removing the effects of z
    '''
    r_xy = stats.pearsonr(xdata, ydata)[0]
    r_xz = stats.pearsonr(xdata, zdata)[0]
    r_yz = stats.pearsonr(ydata, zdata)[0]
    return _pcc(r_xy, r_xz, r_yz)
    # return (r_xy - r_xz*r_yz)/(np.sqrt(1-r_xz**2) * np.sqrt(1-r_yz**2))

def _pcc(r_xy, r_xz, r_yz):
    return (r_xy - r_xz*r_yz)/(np.sqrt(1-r_xz**2) * np.sqrt(1-r_yz**2))


def pearsonr(x, y=None, weights=None):
    '''
    weighted correlation
    '''
    if y == None:
        y = np.arange(len(x))
    x = np.array(x)
    y = np.array(y)
    assert len(x) == len(y)
    N = len(x)
    if weights == None:
        weights = np.ones(len(x))*1./N
    else:
        weights = np.array(weights)

    weights /= np.sum(weights)

    inds = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[inds]
    y = y[inds]
    weights = weights[inds]
    
    mean_x = np.sum(x*weights)
    mean_y = np.sum(y*weights)
    cov_est = sum( weights*(x-mean_x)*(y-mean_y) )
    std_x = np.sqrt( np.sum( weights*(x-mean_x)**2 ) )
    std_y = np.sqrt( np.sum( weights*(y-mean_y)**2 ) )
    r = cov_est/(std_x*std_y)

    # calculate p value
    df = N-2
    t = r*np.sqrt( float(df)/(1-r**2) )
    
    p = 2*(stats.t.cdf( -abs(t), df))

    return r, p 

def circvar(x, **kwargs):
    '''
    Variance of a circular variable calculated using the Berens method
    '''
    return 1-np.sqrt(np.mean(np.sin(x), **kwargs)**2 + np.mean(np.cos(x), **kwargs)**2)

def kruskal(x, y, display=False):
    '''
    Same as standard Kruskal-Wallis test implemented by scipy.stats, but intervenes on the printing/display if desired
    '''
    from scipy.stats import kruskal as kw
    h, p = kw(x, y)
    if display:
        print 'p=%g (h=%g)' % (p, h)
    return h, p

def trig_(data, inds, DT=None, window_before=30, window_after=30):
    '''
    Extract data portions around specified 'trigger' indices
    '''
    if DT is not None:
        inds = (inds/DT).astype(int)

    if np.ndim(data) == 1:
        data = data.reshape(1, -1)
    trig_data = []
    
    for ind in inds:
        try:
            data_ind = data[:, ind-window_before:ind+window_after].T
            if data_ind.shape[0] == window_before + window_after:
                trig_data.append(data_ind)  
            else:
                print data_ind.shape  
        except:
            pass
    return np.dstack(trig_data)

def plot_trig_(trig_data, window=None, ax=None, error_fn='var', **kwargs):
    '''
    Standard plotting methods for "triggered" data created by trig_ function above.
    '''
    if callable(error_fn):
        pass
    elif error_fn == 'var':
        error_fn = np.var
    elif error_fn == 'std':
        error_fn = np.std
    elif error_fn == 'sem':
        error_fn = stats.sem
    elif error_fn == None:
        pass
    else:
        raise ValueError('Unrecognized error_fn: %r' % error_fn)


    if ax == None:
        import plotutil
        plt.figure()
        axes = plotutil.subplots(1, 1, hold=True)
        ax = axes[0,0]

    mean_ = np.mean(trig_data, axis=2).ravel()
    
    if window == None:
        window = np.arange(len(mean_))

    if error_fn is not None:
        import plotutil
        err_ = error_fn(trig_data, axis=2).ravel()
        plotutil.error_line(ax, window, mean_, err_, **kwargs)
    else:
        ax.plot(window, mean_, **kwargs)
    return ax

def mlr(obs, predictors):
    '''
    Determine the r^2 by multiple linear regression
    '''
    C = np.linalg.pinv(np.mat(predictors)) * np.mat(obs).reshape(-1,1)
    pred_obs = np.array(np.mat(predictors) * C).ravel()
    return pearsonr(np.array(obs).ravel(), pred_obs)
