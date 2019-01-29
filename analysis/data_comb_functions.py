'''
Functions for combining data
'''
import numpy as np
from collections import defaultdict

def uneven_vstack(data_ls):
    '''
    Apply np.vstack to a list of data where the rows are not all the same lengths. 
    "Missing" data gets replaced with NaNs
    '''
    l = max(map(lambda x: x.shape[-1], data_ls))
    data_ls = [x.reshape(1,-1) if np.ndim(x) == 1 else x for x in data_ls]
    n_pts = sum(map(len, data_ls))
    data_arr = np.ones([n_pts, l]) * np.nan
    data_pt_exts = []
    for k, data_pt in enumerate(data_ls):
        data_pt_ext = np.ones([data_pt.shape[0], l]) * np.nan 
        data_pt_ext[:,:data_pt.shape[1]] = data_pt
        data_pt_exts.append(data_pt_ext)
        # data_arr[k, :len(data_pt)] = data_pt
    data_arr = np.vstack(data_pt_exts)
    return data_arr

def combine_across_blocks(data, comb=lambda x: x):
    '''
    Combine output from TaskEntryCollection.proc_trials across days

    Parameters
    ----------
    data: list of dictionaries
        Each dictionary is the output of one set of blocks (see TaskEntryCollection.proc_trials for details)
    comb: callable with 1 arg, optional, default = "no op"
        Function to use to combine the different 'values' corresponding to the same dictionary key

    Returns
    -------
    data3: dict
        Keys are the union of all the keys in each dictionary in 'data'. 
        Values for repeated keys are combined using the 'comb' input.
    '''
    data2 = defaultdict(list)
    for d in data:
        for key in d:
            data2[key].append(d[key])

    data3 = dict()
    for key in data2:
        data3[key] = comb(data2[key])
    return data3