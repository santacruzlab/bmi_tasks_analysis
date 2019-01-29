
import numpy as np


def inst_direction_error(move_vec, cursor_rad, target_vec, target_rad):
    '''
    Return measure of angular error for a movement vector based on assumption that
    movement goal is to get the cursor to the closest location where it is entirely inside
    the target. Only works for 2D
    '''

    rad_diff = target_rad - cursor_rad
    if np.linalg.norm(target_vec - move_vec) <= rad_diff:
        error = 0.0
    else:
        mag_targ_vec = np.linalg.norm(target_vec)
        alpha = np.abs(np.arctan(rad_diff/mag_targ_vec))

        theta = np.abs(np.arctan2(move_vec[1], move_vec[0]) - np.arctan2(target_vec[1], target_vec[0]))
        if theta > (np.pi):
            theta = np.pi*2 - theta

        if theta < alpha:
            error = 0.0
        else:
            error = np.abs(theta - alpha)

    return error

def angular_error(hdf):

    cursor = hdf.root.task[:]['cursor'][:,[0,2]]
    cursor_vecs = np.diff(cursor, axis=0)
    target = hdf.root.task[:]['target'][:,[0,2]]
    target_vecs = (target - cursor)[1:]
    cursor_rad = hdf.root.task.attrs.cursor_radius
    target_rad = hdf.root.task.attrs.target_radius

    states = hdf.root.task_msgs[:]
    times = [(state['time']-1, states[i+1]['time']-1) for i, state in enumerate(states) if state['msg'] in ['target', 'hold']]
    mask = np.zeros(len(cursor_vecs))
    mask = mask>0

    for t in times:
        mask[t[0]:t[1]] = True

    mcursor_vecs = cursor_vecs[mask]
    mtarget_vecs = target_vecs[mask]

    errs = np.zeros(len(mcursor_vecs))

    for i in range(len(mcursor_vecs)):
        errs[i] = inst_direction_error(mcursor_vecs[i], cursor_rad, mtarget_vecs[i], target_rad)
    errs = errs[1:]

    delta = np.zeros(len(cursor_vecs)-1)
    for i in range(len(delta)):
        ang = np.abs(np.arctan2(cursor_vecs[i,1],cursor_vecs[i,0]) - np.arctan2(cursor_vecs[i+1,1],cursor_vecs[i+1,0]))
        if ang > np.pi:
            ang = np.pi*2 - ang
        delta[i] = ang
    delta = delta[mask[1:]]

    speed = np.zeros(len(cursor_vecs))
    for i in range(len(speed)):
        speed[i] = np.linalg.norm(cursor_vecs[i])
    speed = speed[1:]
    speed = speed[mask[1:]]

    return errs, delta, speed

