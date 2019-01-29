#!/usr/bin/python
'''
Methods for analyzing decoders
'''
from db.tracker import models
import datetime
import numpy as np
from db import dbfunctions as dbfn


def calc_cursor_null_movement(kf, spike_counts, ctrl_inds=slice(3,6), data_inds=slice(1,None)):
    '''
    Calculate the magnitude of all the stuff cancelled out in the null spce of the Kalman gain
    '''
    K_null = kf.get_K_null()
    null_spike_counts = K_null*spike_counts
    F, K = kf.get_sskf()
    K = np.mat(K)
    null_movement = np.abs(K) * np.abs(null_spike_counts)
    return np.array(map(np.linalg.norm, null_movement[ctrl_inds, data_inds].T))

def calc_cursor_null_movement_bmi3d(te, **kwargs):
    kf = te.decoder.filt
    spike_counts = te.get_spike_counts()
    return calc_cursor_null_movement(kf, spike_counts, **kwargs)

def get_seed_block(te, max_depth=np.inf):
    '''
    Determine how much time was spent on CLDA for a given decoder on a particular day
    '''
    decoder_record = te.decoder_record
    from db.tracker import models 
    depth = 0
    total_time = 0
    while depth < max_depth:
        rec = models.TaskEntry.objects.using(te.record._state.db).get(id=decoder_record.entry_id)
        t = datetime.datetime.strptime(rec.offline_report()['Runtime'], r'%H:%M:%S') - datetime.datetime.strptime('', '')
        total_time += t.seconds
        try:
            new_decoder_record = dbfn.TaskEntry(rec, dbname=te.record._state.db).decoder_record
            if new_decoder_record == decoder_record:
                print 'circle'
                break
            else:
                decoder_record = new_decoder_record

            if decoder_record.entry_id == None:
                break
        except:
            break
            
        depth += 1
    return rec

def crop_decoder_bands(decoder, bands_to_keep):
    '''
    For LFP KFDecoders, modify the decoder object to use only the subset of frequency bands specified
    '''
    n_channels = len(decoder.extractor_kwargs['channels'])
    bands = decoder.extractor_kwargs['bands']
    full_bands = []
    for x in bands:
        full_bands += [x] * n_channels
    obs_inds_to_keep, = np.nonzero([x in bands_to_keep for x in full_bands])
    C = decoder.filt.C[obs_inds_to_keep, :]
    Q = decoder.filt.Q[np.ix_(obs_inds_to_keep, obs_inds_to_keep)]
    C_xpose_Q_inv = C.T * Q.I
    C_xpose_Q_inv_C = C.T * Q.I * C
    decoder.filt.C = C
    decoder.filt.Q = Q
    decoder.filt.C_xpose_Q_inv = C_xpose_Q_inv
    decoder.filt.C_xpose_Q_inv_C = C_xpose_Q_inv_C
    return decoder

def _subband_obs_inds(decoder, bands_to_keep):
    n_channels = len(decoder.extractor_kwargs['channels'])
    bands = decoder.extractor_kwargs['bands']
    full_bands = []
    for x in bands:
        full_bands += [x] * n_channels
    obs_inds_to_keep, = np.nonzero([x in bands_to_keep for x in full_bands]) 
    return obs_inds_to_keep   

def band_contribution(te):
    decoder = te.decoder_record.load()
    decoder_0to40 = crop_decoder_bands(te.decoder_record.load(), [(0, 10), (10, 20), (20, 30), (30, 40)])
    decoder_40to80 = crop_decoder_bands(te.decoder_record.load(), [(40, 50), (50, 60), (60, 70), (70, 80)])
    decoder_80to150 = crop_decoder_bands(te.decoder_record.load(), [(80, 90), (90, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150)])

    _, _, epochs = te.get_matching_state_transition_seq(('target', 'hold'))
    F_0to40, K_0to40 = decoder_0to40.get_sskf()
    F_40to80, K_40to80 = decoder_40to80.get_sskf()
    F_80to150, K_80to150 = decoder_40to80.get_sskf()


    for st, end in epochs:
        st /= 6
        end /= 6
        n = end - st

def decoder_similarity(params, te, var_inds=[3,5], verbose=False):
    '''
    Parameters
    -----------
    params: record array
        One row of the 'clda' table
    te: dbfn.TaskEntry instance
        Block with the "true" decoder
    '''
    if isinstance(params, kfdecoder.KFDecoder):
        dec = params
    elif hasattr(params, 'dtype'):
        try:
            C = params['kf_C']
            Q = params['kf_Q']
        except:
            C = params['kf.C']
            Q = params['kf.Q']        
        kf = kfdecoder.KalmanFilter(A=te.decoder.filt.A, W=te.decoder.filt.W, C=C, Q=Q, is_stochastic=te.decoder.filt.is_stochastic)
        dec = te.decoder
        dec.filt = dec.kf = kf
    else:
        raise Exception 

    dec.filt._init_state()
    spike_counts = te.get_spike_counts()

    F, K = dec.filt.get_sskf()
    F = np.mat(F)
    K = np.mat(K)
    
    x = np.mat(te.hdf.root.task[5::6]['decoder_state'][:,:,0]).T
    Ku_rec = np.array(x[:,1:] - F*x[:,:-1])
    Ku = np.array(K*np.mat(spike_counts[:,1:]))

    inds = te.label_trying()[:-1]
    inds[:100] = False

    if verbose:
        print "final CLDA", np.var(Ku[var_inds, :], axis=1)
        print "seed      ", np.var(Ku_rec[var_inds, :], axis=1)
        print 

    r_vals = []
    for var_ind in var_inds:
        r, p = pearsonr(Ku_rec[var_ind, inds], Ku[var_ind, inds])
        r_vals.append(r)
    return tuple(r_vals)

def decoder_similarity_full(params, te, var_inds=[3,5], verbose=False):
    '''
    Parameters
    -----------
    params: record array
        One row of the 'clda' table
    te: dbfn.TaskEntry instance
        Block with the "true" decoder
    '''
    if isinstance(params, kfdecoder.KFDecoder):
        dec = params
    elif hasattr(params, 'dtype'):
        try:
            C = params['kf_C']
            Q = params['kf_Q']
        except:
            C = params['kf.C']
            Q = params['kf.Q']        
        kf = kfdecoder.KalmanFilter(A=te.decoder.filt.A, W=te.decoder.filt.W, C=C, Q=Q, is_stochastic=te.decoder.filt.is_stochastic)
        dec = te.decoder
        dec.filt = dec.kf = kf
    else:
        raise Exception 

    dec.filt._init_state()
    spike_counts = te.get_spike_counts()

    F, K = dec.filt.get_sskf()
    F = np.mat(F)
    K = np.mat(K)
    
    x = te.hdf.root.task[5::6]['decoder_state'][:,:,0].T
    x_seed = np.array(dec.decode(spike_counts).T)

    inds = te.label_trying()
    inds[:100] = False

    r_vals = []
    for var_ind in var_inds:
        r, p = pearsonr(x[var_ind, inds], x_seed[var_ind, inds])
        r_vals.append(r)
    return tuple(r_vals)

# def decoder_similarity_full(params, task_entries, var_inds=[3,5], verbose=False):
#     data = []
#     data_seed = []
#     for te in task_entries:
#         x, x_seed = run_seed_decoder_full(params, te)
#         inds = te.label_trying()
#         inds[:100] = False        
#         data.append(x[var_inds, inds])
#         data_seed.append(x_seed[var_inds, inds])

#     data = np.hstack(data)
#     data_seed = np.hstack(data_seed)

#     r_vals = []
#     for var_ind in var_inds:
#         r, p = pearsonr(x[var_ind, inds], x_seed[var_ind, inds])
#         r_vals.append(r)
#     return tuple(r_vals)



def run_seed_decoder_full(params, te, var_inds=[3,5], verbose=False):
    '''
    Parameters
    -----------
    params: record array
        One row of the 'clda' table
    te: dbfn.TaskEntry instance
        Block with the "true" decoder
    '''
    if isinstance(params, kfdecoder.KFDecoder):
        dec = params
    elif hasattr(params, 'dtype'):
        try:
            C = params['kf_C']
            Q = params['kf_Q']
        except:
            C = params['kf.C']
            Q = params['kf.Q']        
        kf = kfdecoder.KalmanFilter(A=te.decoder.filt.A, W=te.decoder.filt.W, C=C, Q=Q, is_stochastic=te.decoder.filt.is_stochastic)
        dec = te.decoder
        dec.filt = dec.kf = kf
    else:
        raise Exception 

    dec.filt._init_state()
    spike_counts = te.get_spike_counts()
    
    x = te.hdf.root.task[5::6]['decoder_state'][:,:,0].T
    x_seed = np.array(dec.decode(spike_counts).T)
    return x, x_seed


def tentacle_task_relevant_joint_similarity_full(params, te, var_inds=[3,5], verbose=False):
    if isinstance(params, kfdecoder.KFDecoder):
        dec = params
    elif hasattr(params, 'dtype'):
        try:
            C = params['kf_C']
            Q = params['kf_Q']
        except:
            C = params['kf.C']
            Q = params['kf.Q']        
        kf = kfdecoder.KalmanFilter(A=te.decoder.filt.A, W=te.decoder.filt.W, C=C, Q=Q, is_stochastic=te.decoder.filt.is_stochastic)
        dec = te.decoder
        dec.filt = dec.kf = kf
    else:
        raise Exception 

    dec.filt._init_state()
    spike_counts = te.get_spike_counts()

    F, K = dec.filt.get_sskf()
    F = np.mat(F)
    K = np.mat(K)
    
    x = te.hdf.root.task[5::6]['decoder_state'][:,:,0].T
    x_seed = np.array(dec.decode(spike_counts).T)

    inds = te.label_trying()
    inds[:100] = False

    ### Calculate the endpoint velocity state
    joint_pos = te.hdf.root.task[5::6]['joint_angles']
    
    T = x_seed.shape[1]
    endpt_vel_seed = np.zeros([2, T])
    endpt_vel = np.zeros([2, T])
    for k in range(T):
        theta = joint_pos[k]
        J = te.kin_chain.jacobian(theta)
        endpt_vel_seed[:,k] = np.dot(J, x_seed[4:8, k])
        endpt_vel[:,k] = np.dot(J, x[4:8, k])

    r_vals = []
    for var_ind in [0, 1]:
        r, p = pearsonr(endpt_vel[var_ind, inds], endpt_vel_seed[var_ind, inds])
        r_vals.append(r)
    
    return tuple(r_vals)


