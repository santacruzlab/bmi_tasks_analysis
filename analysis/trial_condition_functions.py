'''
Trial categorization functions
'''
import numpy as np

def default(te, trial_msgs): 
    return 0

def get_visibility_from_trial(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']

    visible = te.hdf.root.task[st]['arm_visible'][0]
    return visible

def get_starting_config_and_visibility_from_trial(te, trial_msgs):
    premove_msg = trial_msgs[trial_msgs['msg'] == 'premove']
    premove_time = premove_msg[0]['time']
    starting_config = te.hdf.root.task[premove_time]['decoder_state'].ravel()
    n_iter = 7
    k = 0
    while not np.all(starting_config[4:8] == 0) and k < n_iter:
        starting_config = te.hdf.root.task[premove_time]['decoder_state'].ravel()        
        premove_time += 1
        k += 1
        
    starting_config = tuple(starting_config[:4])
    visible = te.hdf.root.task[premove_time]['arm_visible'][0]
    return starting_config, visible

def get_starting_config_from_trial(te, trial_msgs, debug=False):
    premove_msg = trial_msgs[trial_msgs['msg'] == 'premove']
    premove_time = max(premove_msg[0]['time'], 1)
    starting_config = te.hdf.root.task[premove_time]['decoder_state'].ravel()
    n_iter = 12
    k = -5
    while not np.all(starting_config[4:8] == 0) and k < n_iter:
        starting_config = te.hdf.root.task[premove_time]['decoder_state'].ravel()        
        premove_time += 1
        k += 1
        if debug: print premove_time
        
    starting_config = tuple(starting_config[:4])
    starting_config = tuple([np.arctan2(np.sin(angle), np.cos(angle)) for angle in starting_config])
    return starting_config

def get_starting_elbow_angle(te, trial_msgs):
    starting_config = get_starting_config_from_trial(te, trial_msgs)
    el_angle = np.round(np.sum(starting_config[:3]) * 180/np.pi, decimals=1)
    origin = te.kin_chain.endpoint_pos(starting_config)
    origin_angle = np.round(np.arctan2(origin[-1], origin[0]) * 180/np.pi, decimals=2)
    return origin_angle, el_angle

def get_starting_elbow_angle_and_visibility(te, trial_msgs):
    visibility = get_visibility_from_trial(te, trial_msgs)
    origin_angle, el_angle = get_starting_elbow_angle(te, trial_msgs)
    return origin_angle, el_angle, visibility


starting_config = get_starting_config_from_trial    

def get_end_type_and_visibility(te, task_msgs):
    visibility = get_visibility_from_trial(te, task_msgs)
    end_type = task_msgs[-1]['msg']
    return (visibility, end_type)

def get_reach_target(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    target = tuple(te.hdf.root.task[end]['target'])
    return target

def reach_target_angle(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    target = tuple(te.hdf.root.task[end]['target'])
    angle = np.arctan2(target[-1], target[0]) * 180/np.pi
    return np.round(angle, decimals=2)

def reach_target_angle2(te, trial_msgs):
    '''
    Same as reach-target-angle, but angles bounded between [0, 360] degrees instead of [-180,180] degrees
    '''
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    target = tuple(te.hdf.root.task[end]['target'])
    angle = np.round(np.arctan2(target[-1], target[0]) * 180/np.pi, decimals=2)
    if angle < 0:
        angle += 360
    return angle #np.round(angle, decimals=2)


def get_reach_origin(te, trial_msgs):
    end_msg_ind = np.nonzero(np.logical_and(trial_msgs['msg'] == 'hold', trial_msgs['target_index'] == 1))[0][0]
    end = trial_msgs[end_msg_ind]['time']
    st = trial_msgs[end_msg_ind-1]['time']
    target = tuple(te.hdf.root.task[end]['target'])
    origin = tuple(te.reach_origin[st])
    return origin

def dist_to_target(te, trial_msgs):
    targ = get_reach_target(te, trial_msgs)
    return np.round(np.linalg.norm(targ), decimals=2)
