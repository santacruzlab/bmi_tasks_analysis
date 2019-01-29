import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

def exp_curve_fit(xdata, ydata):
    assert len(xdata) == len(ydata)
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(exp_func, xdata, ydata, p0=[0, 0.001, np.mean(ydata)], maxfev=10000)
    
    fn_mdl = lambda x: exp_func(x, *popt)
    ydata_mdl = exp_func(xdata, *popt)
    SS_res = np.sum((ydata - ydata_mdl)**2)
    SS_tot = np.sum((ydata - np.mean(ydata))**2)
    R2 = 1 - SS_res/SS_tot
    print R2
    a, b, c = popt
    print 'exp fit', pearsonr(np.log((ydata - c)/a), -b*xdata)
    return ydata_mdl, R2, popt, fn_mdl

def plot_exp_fit_curve(xdata, ydata=None, ax=None, p_corner='top_right', text_kwargs=dict(), **kwargs):
    if ax == None:
        plt.figure()
        ax = plt.subplot(111)

    if ydata == None:
        ydata = xdata
        xdata = np.arange(len(ydata))

    curve, R2, _, _ = exp_curve_fit(xdata, ydata)
    r, p = pearsonr(curve, ydata)
    # print pearsonr(curve, ydata)
    ax.plot(xdata, curve, **kwargs)

    import plotutil
    plotutil.write_corner_text(ax, p_corner, r'$r^2=%0.3g$%s' % (R2, plotutil.p_val_to_asterisks(p)), **text_kwargs)