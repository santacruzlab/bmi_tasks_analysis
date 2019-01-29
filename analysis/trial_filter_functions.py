'''
Collection of functions to filter trials by the state transition messages during the trial
'''
import numpy as np

def default(trial_msgs): 
    return True

def rewarded_trial(te, trial_msgs):
    return trial_msgs[-1]['msg']  == 'reward'

def rewarded_and_unassisted_trial(te, trial_msgs):
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    except IndexError:
        return False

    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return trial_msgs[-1]['msg']  == 'reward' and np.all(te.hdf.root.task[st:end]['assist_level'] == 0)

def target_in_trial(te, trial_msgs):
    return trial_msgs[-1]['msg'] in ['hold_penalty', 'reward'] and (1 in trial_msgs['target_index'][1:])

def hit_targets(target_list, tol=0.1):
    '''
    Returns a function which can be used to filter for specific reach targets. 
    Only to be used on trials with a single target in the sequence (or if it's used on a multi-target trial, only the first target is examined)
    '''
    if np.ndim(target_list) == 1:
        target_list = target_list.reshape(1, -1)

    def fn(te, trial_msgs):
        try:
            end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
            end = trial_msgs[end_msg_ind]['time']
            target = te.hdf.root.task[end]['target']
            return (min(map(np.linalg.norm, target_list - target)) < tol)
        except:
            return False
    return fn

def reach_origin(target_list, tol=0.1):
    '''
    Returns a function which can be used to filter for specific reach targets. 
    Only to be used on trials with a single target in the sequence (or if it's used on a multi-target trial, only the first target is examined)
    '''
    if np.ndim(target_list) == 1:
        target_list = target_list.reshape(1, -1)

    def fn(te, trial_msgs):
        try:
            end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
            end = trial_msgs[end_msg_ind]['time']
            target = te.hdf.root.task[end]['target'][[0,2]]
            st = trial_msgs[end_msg_ind-1]['time']
            origin = te.reach_origin[st]
            return (min(map(np.linalg.norm, target_list - origin)) < tol)
        except:
            return False
    1
    return fn	
