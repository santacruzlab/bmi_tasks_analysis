'''
Trial processing functions
'''
import numpy as np
import performance_metrics

def default(te, trial_msgs): 
    return 1

def get_trial_task_axis_perf(te, trial_msgs, **kwargs):
    ## Get the boundaries
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
        end = trial_msgs[end_msg_ind]['time']
        st = trial_msgs[end_msg_ind-1]['time']
    except:
        print trial_msgs
        raise Exception
    
    origin = te.reach_origin[end][[0,2]]
    target = te.hdf.root.task[end]['target'][[0,2]]

    traj = te.hdf.root.task[st:end]['cursor'][:,[0,2]]
    trial_perf = performance_metrics.get_task_axis_error_measures(traj, origin=origin, target=target, return_type='recarray', eff_target_size=te.target_radius-te.cursor_radius, **kwargs)
    if trial_perf['NPL'] == np.inf:
        print origin, target

    return trial_perf

def joint_space_path_length(te, trial_msgs):
    sl = _Nth_target_state_slice(trial_msgs, target_index=1)
    joint_angles_diff = np.diff(te.hdf.root.task[sl]['joint_angles'], axis=0)
    total_path_length = np.sum(np.abs(joint_angles_diff), axis=0)
    return total_path_length

def _Nth_target_state_slice(trial_msgs, target_index=1):
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
        end = trial_msgs[end_msg_ind]['time']
        st = trial_msgs[end_msg_ind-1]['time']
    except:
        print trial_msgs
        import traceback
        traceback.print_exc()
        raise Exception
    return slice(st, end)    

def average_spike_rate_during_target_state(te, trial_msgs, target_index=1):
    sl = _Nth_target_state_slice(trial_msgs, target_index=target_index)
    return np.mean(te.hdf.root.task[sl]['spike_counts'], axis=0)

def Ku_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return te.get_ctrl_vecs()[:, st/6:end/6]

def w_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    ds = 60 * max(te.decoder.binlen, 1./60) 
    us = max(1./60 / te.decoder.binlen, 1)
    if us >= 1 and ds <= 1:
        sl = slice(st*us, end*us, None)
    else:
        sl = slice(st/ds, end/ds)
    return np.array(te.get_BMI_motor_commands()[:, sl])

def inds_of_target_state(te, trial_msgs, target_index=1):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == target_index))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return st, end


def Kyt_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    ds = 60 * max(te.decoder.binlen, 1./60) 
    us = max(1./60 / te.decoder.binlen, 1)
    if us >= 1 and ds <= 1:
        sl = slice(st*us, end*us, None)
    else:
        sl = slice(st/ds, end/ds)
    return np.array(te.Kyt[:, sl])

def mean_Kyt_during_target_state(te, trial_msgs):
    Kyt = Kyt_during_target_state(te, trial_msgs)
    return np.mean(Kyt, axis=1)

def w_KF_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    ds = 60 * max(te.decoder.binlen, 1./60) 
    us = max(1./60 / te.decoder.binlen, 1)
    if us >= 1 and ds <= 1:
        sl = slice(st*us, end*us, None)
    else:
        sl = slice(st/ds, end/ds)
    return np.array(te.get_KF_active_BMI_motor_commands()[:, sl])

def xt_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    ds = 60 * max(te.decoder.binlen, 1./60)
    xt = te.get_decoder_state()
    us = max(1./60 / te.decoder.binlen, 1)
    if us >= 1 and ds <= 1:
        sl = slice(st*us, end*us, None)
    else:
        sl = slice(st/ds, end/ds)
    return xt[:, sl]


def ctrl_angle_during_target_state(te, trial_msgs):
    xt = xt_during_target_state(te, trial_msgs)
    wt = w_during_target_state(te, trial_msgs)
    vt = xt[[3,5], :]
    vct = wt[[3,5], :]
    from utils_ import geometry
    return geometry.angle(vt, vct)

def Kyt_coherence_angle_during_target_state(te, trial_msgs):
    Kyt = Kyt_during_target_state(te, trial_msgs)
    v_prev = Kyt[[3,5], :-1]
    v = Kyt[[3,5], 1:]    
    from utils_ import geometry
    return geometry.angle(v, v_prev)

def w_coherence_angle_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    ds = 60 * max(te.decoder.binlen, 1./60) 
    us = max(1./60 / te.decoder.binlen, 1)
    if us >= 1 and ds <= 1:
        sl = slice(st*us, end*us, None)
    else:
        sl = slice(st/ds-1, end/ds)
    wt = np.array(te.get_BMI_motor_commands()[:, sl])

    v_prev = wt[[3,5], :-1]
    v = wt[[3,5], 1:]    
    from utils_ import geometry
    return (np.pi - geometry.angle(v, v_prev))/np.pi

def Kyt_vel_mag_during_target_state(te, trial_msgs):
    Kyt = Kyt_during_target_state(te, trial_msgs)
    return np.array(map(np.linalg.norm, Kyt[[3,5], :].T))

def w_vel_mag_during_target_state(te, trial_msgs):
    wt = w_during_target_state(te, trial_msgs)
    return np.array(map(np.linalg.norm, wt[[3,5], :].T))

def decoded_speed_during_target_state(te, trial_msgs):
    dec_state = xt_during_target_state(te, trial_msgs)
    return np.array(map(np.linalg.norm, np.array(dec_state)[3:6, :].T))
    



def spike_counts_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return te.hdf.root.task[st:end]['spike_counts']

def spike_counts_during_target_state_offset(te, trial_msgs, start_offset=30, end_offset=0):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return te.hdf.root.task[st + start_offset:end - end_offset]['spike_counts']

def spike_counts_during_target_state_first_half(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return te.hdf.root.task[st:(st+end)/2]['spike_counts']

def spike_counts_during_target_state_second_half(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return te.hdf.root.task[(st+end)/2:end]['spike_counts']

def mean_w_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return np.mean(np.array(te.get_BMI_motor_commands()[:, st/6:end/6]), axis=1)

def total_null_movement_during_target_state(te, trial_msgs):    
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    task_rel_movement = te.task_relevant_movement
    return np.sum(np.diff(task_rel_movement[st/6:end/6], axis=1))

def total_task_movement_during_target_state(te, trial_msgs):    
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    task_rel_movement = te.task_relevant_movement
    return np.sum(task_rel_movement[st/6:end/6, 0])

def endpt_cmd_during_target_state(te, trial_msgs):
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    except:
        print trial_msgs
        raise Exception
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']

    joint_pos = te.hdf.root.task[st:end:6]['joint_angles']
    Kyt = te.get_ctrl_vecs()[:, st/6:end/6]
    endpt_cmd = []
    for k in range(len(joint_pos)-2):
        theta = joint_pos[k]
        J = te.kin_chain.jacobian(theta)
        endpt_cmd_k = np.dot(J, Kyt[4:8, k])
        endpt_cmd.append(endpt_cmd_k)
    endpt_cmd = np.vstack(endpt_cmd).T
    return endpt_cmd

def endpt_cmd_during_target_state(te, trial_msgs):
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    except:
        print trial_msgs
        raise Exception
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']

    joint_pos = te.hdf.root.task[st:end:6]['joint_angles']
    w_t = np.mat(te.get_BMI_motor_commands()[:, st/6:end/6])
    # print w_t.shape
    endpt_cmd = []
    for k in range(len(joint_pos)-2):
        theta = joint_pos[k]
        J = te.kin_chain.jacobian(theta)
        endpt_cmd_k = np.dot(J, w_t[4:8, k])
        endpt_cmd.append(endpt_cmd_k)
    endpt_cmd = np.hstack(endpt_cmd)
    return endpt_cmd

def endpt_null_during_target_state(te, trial_msgs):
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    except:
        print trial_msgs
        raise Exception
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']

    joint_pos = te.hdf.root.task[st:end:6]['joint_angles']
    w_t = np.mat(te.get_BMI_motor_commands()[:, st/6:end/6])
    # print w_t.shape
    endpt_cmd = []
    for k in range(len(joint_pos)-2):
        theta = joint_pos[k]
        J = np.mat(te.kin_chain.jacobian(theta))
        J_null = np.eye(J.shape[1]) - np.linalg.pinv(J) * J
        endpt_cmd_k = J_null * w_t[4:8, k]
        endpt_cmd.append(endpt_cmd_k)
    endpt_cmd = np.hstack(endpt_cmd)
    return endpt_cmd

def joint_vel_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return te.hdf.root.task[st:end:6]['decoder_state'][:,4:8,0].T

def mean_Ku_during_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return np.mean(te.get_ctrl_vecs()[:, st/6:end/6], axis=1)


def Ku_during_beginning_of_target_state(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return te.get_ctrl_vecs()[:, st/6+2:st/6 + 10]

def extract_cursor_trajectory(te, trial_msgs):
    '''
    '''
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
        end = trial_msgs[end_msg_ind]['time']
        st = trial_msgs[end_msg_ind-1]['time']
    except:
        print trial_msgs    

    cursor_traj = te.hdf.root.task[st:end]['cursor']
    return cursor_traj

def extract_joint_trajectory(te, trial_msgs):
    '''
    '''
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
        end = trial_msgs[end_msg_ind]['time']
        st = trial_msgs[end_msg_ind-1]['time']
    except:
        print trial_msgs    

    joint_traj = te.hdf.root.task[st:end]['joint_angles']
    return joint_traj

def extract_joint_trajectory_correlations(te, trial_msgs):
    joint_traj = extract_joint_trajectory(te, trial_msgs)[::6]
    return np.corrcoef(joint_traj.T)

def get_joint_config_starting(te, trial_msgs):
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    except:
        print trial_msgs
        raise Exception
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    return te.hdf.root.task[st]['joint_angles']

def get_joint_config_entering_target(te, trial_msgs):
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    except:
        print trial_msgs
        raise Exception
    end = trial_msgs[end_msg_ind]['time']
    return te.hdf.root.task[end]['joint_angles']

def get_cursor_pos_entering_target(te, trial_msgs):
    try:
        end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    except:
        print trial_msgs
        raise Exception
    end = trial_msgs[end_msg_ind]['time']
    return te.hdf.root.task[end]['cursor']    

