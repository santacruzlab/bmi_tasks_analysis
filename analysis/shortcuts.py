#!/usr/bin/python
'''
'''
import numpy as np
from analysis import trial_filter_functions, trial_condition_functions, trial_proc_functions

def comb(x):
    x = np.hstack(x)
    x = x[(x['ME'] < np.inf) * (x['NPL'] < np.inf)]
    x = np.array(filter(lambda y: not (y == np.inf), x))
    return x

def task_axis_perf(col, sl=(None,None), **kwargs):
    '''
    Calculate task-axis average performance for a collection for a reaching task
    '''
    perf = col.proc_trials(trial_filter_fn=trial_filter_functions.rewarded_trial,
        trial_proc_fn=trial_proc_functions.get_trial_task_axis_perf,
        data_comb_fn=np.hstack, **kwargs)

    keys = perf[0].dtype.fields.keys()
    dtype = np.dtype([(x, 'f8') for x in keys])
    mean_perf = []
    for p in perf:
        mean_p = np.zeros((1,), dtype=dtype)
        for key in keys:
            mean_p[key] = np.mean(p[key][sl])
        mean_perf.append(mean_p)
    return np.hstack(mean_perf)

def n_trials(col, **kwargs):
    n_trials = col.proc_trials(trial_filter_fn=trial_filter_functions.target_in_trial,
        trial_proc_fn=lambda x, y: 1,
        data_comb_fn=np.sum, **kwargs)
    return np.array(n_trials, dtype=np.float64)

def mean_reach_time(te, cond=None, **kwargs):
    return te.proc(filt='rewarded_trial', proc='get_trial_task_axis_perf', cond=cond, comb=lambda x: np.mean(np.hstack(x)['reach_time']), **kwargs)

def reach_times(te, **kwargs):
    return te.proc(filt='rewarded_trial', proc='get_trial_task_axis_perf', cond=None, comb=np.hstack, **kwargs)[0]['reach_time']