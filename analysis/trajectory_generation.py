'''
Functions for generating artificial trajectories in arm/joint space
'''
import numpy as np
from itertools import izip

def min_jerk_movement(vel, time, starting_pos, ending_pos, t0, D):
    '''
    Generate the trajectory of a "minimum jerk" point-to-point movement.
    The trajectory for this type of movement has a closed form when 
    the movement starts and begins at rest. See Flash and Hogan, J. Neurosci 1985
    for details

    Parameters
    ----------
    vel: np.array of shape (T, 6)
        Array in which to insert the trajectory
    time: np.array of shape (T,)
        Array of time (in seconds) corresponding to each row of the vel array
    starting_pos: np.array of shape (3,)
    ending_pos: np.array of shape (3,)
    t0: float
        Time marking the beginning of the movement
    D: float
        Duration of the movement

    Returns
    -------
    vel: np.array of shape (T, 6)
        Points to the same array as the 'vel' input argument
    '''
    A = ending_pos - starting_pos
    thisrng, = np.nonzero(np.logical_and(time >= t0, time <= t0+D))
    t = time[thisrng]
    nt = (t-t0)/D
    vel[thisrng, 0:3] = starting_pos + np.outer(A, 6*nt**5 - 15*nt**4 + 10*nt**3).T
    vel[thisrng, 3:6] = np.outer(A, 1./D * (30 * nt**4 -60 * nt**3 + 30 * nt**2)).T
    return vel

def sim_endpt_traj(targets, durations, DT=0.1, inter_target_delay=0.3):
    '''
    Parameters
    ----------
    targets: np.array of shape (N, 3)
        Array of 3D targets
    durations: np.array of shape (N,)
        Durations of each reach
    DT: float, optional, default = 0.1
        Amount of time in between each discrete sample, in seconds
    inter_target_delay: float, optional, default=0.3
        Amount of time in between each "trial", i.e. how long to wait at each target

    Returns
    -------
    kin: np.array of shape (T, 6)
        Position and velocity trajectories
    '''

    n_targets = len(targets - 1)
    total_time = n_targets*inter_target_delay + sum(durations[:n_targets])
    T = total_time / DT
    time = np.arange(T) * DT
    kin = np.zeros([T, 6])
    t0 = 0
    for starting_pos, ending_pos, D in izip(targets[:-1], targets[1:], durations):
        kin = min_jerk_movement(kin, time, starting_pos, ending_pos, t0, D)
        # Fill in the positions for the inter_target_delay period
        thisrng, = np.nonzero(np.logical_and(time >= (t0 + D), time <= (t0+D+inter_target_delay)))
        kin[thisrng,0:3] = ending_pos

        # Advance the t_0 to be the start of the next trial
        t0 += D + inter_target_delay

    
    return kin

def rec_to_normal(arr):
    '''
    Helper function to convert a numpy record array to a normal array
    '''
    new_dtype = (np.float64, len(arr.dtype))
    return arr.view(new_dtype).reshape(-1, len(arr.dtype))

def sim_joint_traj(targets, durations, link_lengths, shoulder_anchor, DT=0.1, inter_target_delay=0.3):
    '''
    Parameters
    ----------
    targets: np.array of shape (N, 3)
        Array of 3D targets
    durations: np.array of shape (N,)
        Durations of each reach
    link_lengths: iterable,
        List (or array) of link lengths of the arm
    shoulder_anchor: np.array of shape (3,)
        Spatial location of most proximal joint in space
    DT: float, optional, default = 0.1
        Amount of time in between each discrete sample, in seconds
    inter_target_delay: float, optional, default=0.3
        Amount of time in between each "trial", i.e. how long to wait at each target

    Returns
    -------
    vel: np.array of shape (T, 3)
        Velocity trajectories
    '''    
    kin = sim_endpt_traj(targets, durations)
    
    # calculate joint velocities
    from riglib.stereo_opengl import ik
    joint_angles, joint_vel = ik.inv_kin_2D(kin[:,0:3] - shoulder_anchor, 
        link_lengths[1], link_lengths[0], kin[:,3:6])
    
    joint_kin = np.hstack([rec_to_normal(joint_angles), rec_to_normal(joint_vel)])
    return joint_kin