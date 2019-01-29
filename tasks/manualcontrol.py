"""
NOTE: This module has been deprecated as of 10/12/13!

This module contains generators and tasks related to manual control experiments.
"""

from __future__ import division
import numpy as np
from collections import OrderedDict
import time

from riglib import reward
from riglib.experiment import traits, Sequence

from riglib.stereo_opengl.window import Window, FPScontrol
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex
from riglib.stereo_opengl.ik import RobotArm

import math

TexPlane = type("TexPlane", (Plane, TexModel), {})

############################################################
# Generators used in the tasks below
###########################################################

def rand_target_sequence_3d(length, boundaries=(-18,18,-10,10,-15,15), distance=10):
    '''

    Generates a sequence of 3D target pairs.

    Parameters
    ----------
    length : int
        The number of target pairs in the sequence.
    boundaries: 6 element Tuple
        The limits of the allowed target locations (-x, x, -y, y, -z, z)
    distance : float
        The distance in cm between the targets in a pair.

    Returns
    -------
    pairs : [n x 3 x 2] array of pairs of target locations


    '''

    # Choose a random sequence of points at least "distance" from the edge of
    # the allowed area
    pts = np.random.rand(length, 3)*((boundaries[1]-boundaries[0]-2*distance),
        (boundaries[3]-boundaries[2]-2*distance),
        (boundaries[5]-boundaries[4]-2*distance))
    pts = pts+(boundaries[0]+distance,boundaries[2]+distance,
        boundaries[4]+distance)

    # Choose a random sequence of points on the surface of a sphere of radius
    # "distance"
    theta = np.random.rand(length)*2*np.pi
    phi = np.arccos(2*np.random.rand(length) - 1)
    x = distance*np.cos(theta)*np.sin(phi)
    y = distance*np.sin(theta)*np.sin(phi)
    z = distance*np.cos(theta)

    # Shift points to correct position relative to first sequence and join two
    # sequences together in a trial x coordinate x start/end point array
    pts2 = np.array([x,y,z]).transpose([1,0]) + pts
    pairs = np.array([pts, pts2]).transpose([1,2,0])
    copy = pairs[0:length//2,:,:].copy()

    # Swap start and endpoint for first half of the pairs
    pairs[0:length//2,:,0] = copy[:,:,1]
    pairs[0:length//2,:,1] = copy[:,:,0]

    # Shuffle list of pairs
    np.random.shuffle(pairs)

    return pairs

def rand_target_sequence_2d(length, boundaries=(-18,18,-12,12), distance=10):
    '''

    Generates a sequence of 2D (x and z) target pairs.

    Parameters
    ----------
    length : int
        The number of target pairs in the sequence.
    boundaries: 6 element Tuple
        The limits of the allowed target locations (-x, x, -z, z)
    distance : float
        The distance in cm between the targets in a pair.

    Returns
    -------
    pairs : [n x 2 x 2] array of pairs of target locations


    '''

    # Choose a random sequence of points at least "distance" from the boundaries
    pts = np.random.rand(length, 3)*((boundaries[1]-boundaries[0]-2*distance),
        0, (boundaries[3]-boundaries[2]-2*distance))
    pts = pts+(boundaries[0]+distance, 0, boundaries[2]+distance)

    # Choose a random sequence of points on the edge of a circle of radius
    # "distance"
    theta = np.random.rand(length)*2*np.pi
    x = distance*np.cos(theta)
    z = distance*np.sin(theta)

    # Shift points to correct position relative to first sequence and join two
    # sequences together in a trial x coordinate x start/end point array
    pts2 = np.array([x,np.zeros(length),z]).transpose([1,0]) + pts
    pairs = np.array([pts, pts2]).transpose([1,2,0])
    copy = pairs[0:length//2,:,:].copy()

    # Swap start and endpoint for first half of the pairs
    pairs[0:length//2,:,0] = copy[:,:,1]
    pairs[0:length//2,:,1] = copy[:,:,0]

    # Shuffle list of pairs
    np.random.shuffle(pairs)

    return pairs

def rand_target_sequence_2d_centerout(length, boundaries=(-18,18,-12,12),
    distance=10):
    '''

    Generates a sequence of 2D (x and z) target pairs with the first target
    always at the origin.

    Parameters
    ----------
    length : int
        The number of target pairs in the sequence.
    boundaries: 6 element Tuple
        The limits of the allowed target locations (-x, x, -z, z)
    distance : float
        The distance in cm between the targets in a pair.

    Returns
    -------
    pairs : [n x 2 x 2] array of pairs of target locations


    '''

    # Create list of origin targets
    pts1 = np.zeros([length,3])

    # Choose a random sequence of points on the edge of a circle of radius 
    # "distance"
    theta = np.random.rand(length)*2*np.pi
    x = distance*np.cos(theta)
    z = distance*np.sin(theta)
    
    # Join start and end points together in a [trial x coordinate x start/end]
    # array (fill in zeros for endpoint y values)
    pts2 = np.array([x,np.zeros(length),z]).transpose([1,0])
    pairs = np.array([pts1, pts2]).transpose([1,2,0])
    
    return pairs

def rand_target_sequence_2d_partial_centerout(length, boundaries=(-18,18,-12,12),distance=10,perc_z=20):
    '''
    PK
    Generates a sequence of 2D (x and z) target pairs with the first target
    always at the origin.

    Parameters
    ----------
    perc_z : float
        The percentage of the z axis to be used for targets 
    length : int
        The number of target pairs in the sequence.
    boundaries: 6 element Tuple
        The limits of the allowed target locations (-x, x, -z, z)
    distance : float
        The distance in cm between the targets in a pair.
    

    Returns
    -------
    pairs : [n x 2 x 2] array of pairs of target locations


    '''
    # Need to get perc_z from settable traits
    # perc_z = traits.Float(0.1, desc="Percent of Y axis that targets move along")
    perc_z=float(10)
    # Create list of origin targets
    pts1 = np.zeros([length,3])

    # Choose a random sequence of points on the edge of a circle of radius 
    # "distance"
    
    #Added PK -- to confine z value according to entered boundaries: 
    theta_max = math.asin(boundaries[3]/distance*(perc_z)/float(100))
    theta = (np.random.rand(length)-0.5)*2*theta_max
    
    #theta = np.random.rand(length)*2*np.pi
    
    x = distance*np.cos(theta)*(np.ones(length)*-1)**np.random.randint(1,3,length)
    z = distance*np.sin(theta)
    
    # Join start and end points together in a [trial x coordinate x start/end]
    # array (fill in zeros for endpoint y values)
    pts2 = np.array([x,np.zeros(length),z]).transpose([1,0])

    pairs = np.array([pts1, pts2]).transpose([1,2,0])
    
    return pairs

def rand_multi_sequence_2d_centeroutback(length, boundaries=(-18,18,-12,12), distance=8):
    '''

    Generates a sequence of 2D (x and z) center-edge-center target triplets.

    Parameters
    ----------
    length : int
        The number of target pairs in the sequence.
    boundaries: 6 element Tuple
        The limits of the allowed target locations (-x, x, -z, z)
    distance : float
        The distance in cm between consecutive targets.

    Returns
    -------
    targs : [length x 2 x 3] array of pairs of target locations


    '''

    # Create list of origin targets
    pts1 = np.zeros([length,3])

    # Choose a random sequence of points on the edge of a circle of radius 
    # "distance"
    theta = np.random.rand(length)*2*np.pi
    x = distance*np.cos(theta)
    z = distance*np.sin(theta)
    
    # Join start and end points together in a [trial x coordinate x start/end]
    # array (fill in zeros for endpoint y values)
    pts2 = np.array([x,np.zeros(length),z]).transpose([1,0])
    targs = np.array([pts1, pts2, pts1]).transpose([1,2,0])
    
    return targs

def rand_multi_sequence_2d_centerout2step(length, boundaries=(-20,20,-12,12), distance=10):
    '''

    Generates a sequence of 2D (x and z) center-edge-far edge target triplets.

    Parameters
    ----------
    length : int
        The number of target pairs in the sequence.
    boundaries: 6 element Tuple
        The limits of the allowed target locations (-x, x, -z, z)
    distance : float
        The distance in cm between consecutive targets.

    Returns
    -------
    targs : [length x 3 x 3] array of pairs of target locations


    '''

    # Create list of origin targets
    pts1 = np.zeros([length,3])

    # Choose a random sequence of points on the edge of a circle of radius 
    # "distance", and matching set with radius distance*2
    theta = np.random.rand(length*10)*2*np.pi
    x1 = distance*np.cos(theta)
    z1 = distance*np.sin(theta)
    x2 = distance*2*np.cos(theta)
    z2 = distance*2*np.sin(theta)

    mask = np.logical_and(np.logical_and(x2>=boundaries[0],x2<=boundaries[1]),np.logical_and(z2>=boundaries[2],z2<=boundaries[3]))
    
    # Join start and end points together in a [trial x coordinate x start/end]
    # array (fill in zeros for endpoint y values)
    pts2 = np.array([x1[mask][:length], np.zeros(length), z1[mask][:length]]).transpose([1,0])
    pts3 = np.array([x2[mask][:length], np.zeros(length), z2[mask][:length]]).transpose([1,0])
    targs = np.array([pts1, pts2, pts3]).transpose([1,2,0])
    
    return targs

class TestBoundary(Window):
    '''
    A very simple task that displays a marker at the specified screen locations.
    Useful for determining reasonable boundary values for targets.
    '''

    status = dict(
        wait = dict(stop=None)
        )
        
    state = "wait"

    boundaries = traits.Tuple((-18,18,-10,10,-12,12), desc="x,y,z boundaries to display")
    
    def __init__(self, **kwargs):
        super(TestBoundary, self).__init__(**kwargs)
        # Create a small sphere for each of the 6 boundary marks
        self.xmin = Sphere(radius=.1, color=(.5,0,.5,1))
        self.add_model(self.xmin)
        self.xmax = Sphere(radius=.1, color=(.5,0,.5,1))
        self.add_model(self.xmax)
        self.ymin = Sphere(radius=.1, color=(.5,0,.5,1))
        self.add_model(self.ymin)
        self.ymax = Sphere(radius=.1, color=(.5,0,.5,1))
        self.add_model(self.ymax)
        self.zmin = Sphere(radius=.1, color=(.5,0,.5,1))
        self.add_model(self.zmin)
        self.zmax = Sphere(radius=.1, color=(.5,0,.5,1))
        self.add_model(self.zmax)
        
    def _start_wait(self):
        self.xmin.translate(self.boundaries[0], 0, 0, reset=True)
        self.xmin.attach()
        self.xmax.translate(self.boundaries[1], 0, 0, reset=True)
        self.xmax.attach()
        self.ymin.translate(0, self.boundaries[2], 0, reset=True)
        self.ymin.attach()
        self.ymax.translate(0, self.boundaries[3], 0, reset=True)
        self.ymax.attach()
        self.zmin.translate(0, 0, self.boundaries[4], reset=True)
        self.zmin.attach()
        self.zmax.translate(0, 0, self.boundaries[5], reset=True)
        self.zmax.attach()
        self.requeue()
        
    def _while_wait(self):
        self.draw_world()

class MovementTraining(Window):
    status = dict(
        wait = dict(stop=None, move_start="movement"),
        movement = dict(move_end="reward", move_stop="wait", stop=None),
        reward = dict(reward_end="wait")
    )
    log_exclude = set((("wait", "move_start"), ("movement", "move_stop")))

    #initial state
    state = "wait"

    path = [[0,0,0]]
    speed = 0
    frame_offset = 2
    over = 0
    inside = 0

    #settable traits
    movement_distance = traits.Float(1, desc="Minimum movement distance to trigger reward")
    speed_range = traits.Tuple(20, 30, desc="Range of movement speed in cm/s to trigger reward")
    reward_time = traits.Float(14)

    #initialize
    def __init__(self, **kwargs):
        super(MovementTraining, self).__init__(**kwargs)
        self.cursor = Sphere(radius=.5, color=(.5,0,.5,1))
        self.add_model(self.cursor)

    def update_cursor(self):
        #get data from 13th marker on motion tracker- take average of all data points since last poll
        pt = self.motiondata.get()
        if len(pt) > 0:
            pt = pt[:, 14, :]
            # NOTE!!! The marker on the hand was changed from #0 to #14 on
            # 5/19/13 after LED #0 broke. All data files saved before this date
            # have LED #0 controlling the cursor.
            pt = pt[~np.isnan(pt).any(1)]        
        if len(pt) > 0:
            pt = pt.mean(0)
            self.path.append(pt)
            #ignore y direction
            t = pt*.25
            t[1] = 0
            #move cursor to marker location
            self.cursor.translate(*t[:3],reset=True)
        else:
            self.path.append(self.path[-1])
        if len(self.path) > self.frame_offset:
            self.path.pop(0)
            d = np.sqrt((self.path[-1][0]-self.path[0][0])**2 + (self.path[-1][1]-self.path[0][1])**2 + (self.path[-1][2]-self.path[0][2])**2)
            self.speed = d/(self.frame_offset/60)
            if self.speed>self.speed_range[0]:
                self.over += 1
            if self.speed_range[0]<self.speed<self.speed_range[1]:
                self.inside += 1
        #write to screen
        self.draw_world()
    
    def _start_wait(self):
        self.over = 0
        self.inside = 0

    def _while_wait(self):
        self.update_cursor()

    def _while_movement(self):
        self.update_cursor()

    def _while_reward(self):
        self.update_cursor()

    def _test_move_start(self, ts):
        return self.over > self.frame_offset

    def _test_move_end(self, ts):
        return ts > self.movement_distance/self.speed_range[0]

    def _test_move_stop(self, ts):
        return self.inside > self.frame_offset
        
    def _test_reward_end(self, ts):
        return ts > self.reward_time

class FixationTraining(Window):
    status = dict(
        wait = dict(start_trial="reward", stop=None),
        reward = dict(reward_end="wait")
    )

    #initial state
    state = "wait"

    #settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")

    #initialize and create fixation point object
    def __init__(self, **kwargs):
        super(FixationTraining, self).__init__(**kwargs)
        self.fixation_point = Sphere(radius=.1, color=(1,0,0,1))
        #keep fixation point hidden for now
        #self.add_model(self.fixation_point)

    def _get_renderer(self):
        return stereo.MirrorDisplay(self.window_size, self.fov, 1, 1024, self.screen_dist, self.iod)
    
    def _test_reward_end(self, ts):
        return ts>self.reward_time

    def _while_wait(self):
        self.draw_world()
    def _while_reward(self):
        self.draw_world()

class TargetCapture(Sequence, FixationTraining):
    status = dict(
        wait = dict(start_trial="origin", stop=None),
        origin = dict(enter_target="origin_hold", stop=None),
        origin_hold = dict(leave_early="hold_penalty", hold="reward"),
        reward = dict(reward_end="target_change"),
        hold_penalty = dict(penalty_end="pre_target_change"),
        pre_target_change = dict(tried_enough = 'target_change', not_tried_enough='wait'),
        target_change = dict(target_change_end='wait')
    )

    #create settable traits
    origin_size = traits.Float(1, desc="Radius of origin targets") #add error if target is smaller than cursor
    origin_hold_time = traits.Float(2, desc="Length of hold required at origin")
    hold_penalty_time = traits.Float(3, desc="Length of penalty time for target hold error")
    exit_radius = 1.5 #Multiplier for the actual radius which is considered 'exiting' the target
    
    no_data_count = 0
    tries = 0
    scale_factor = 3.5 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)
    cursor_radius = .5

    def __init__(self, *args, **kwargs):
        super(TargetCapture, self).__init__(*args, **kwargs)
        self.origin_target = Sphere(radius=self.origin_size, color=(1,0,0,.5))
        self.add_model(self.origin_target)
        self.cursor = Sphere(radius=self.cursor_radius, color=(.5,0,.5,1))
        self.add_model(self.cursor)
    
    def _start_wait(self):
        super(TargetCapture, self)._start_wait()
        #set target color
        self.origin_target.color = (1,0,0,.5)
        #hide target from previous trial
        self.show_origin(False)

    def show_origin(self, show=False):
        if show:
            self.origin_target.attach()
        else:
            self.origin_target.detach()
        self.requeue()

    def _start_origin(self):
        if self.tries == 0:
            #retrieve location of next origin target
            o = self.next_trial.T[0]
            #move target to correct location
            self.origin_target.translate(*o, reset=True) 
        #make visible
        self.show_origin(True)

    def _end_origin_hold(self):
        #change target color
        self.origin_target.color = (0,1,0,0.5)

    def _start_hold_penalty(self):
        self.tries += 1
        #hide target
        self.show_origin(False)
        
    def _start_target_change(self):
        self.tries = 0
    
    def _test_target_change_end(self, ts):
        return True

    def _test_enter_target(self, ts):
        #get the current cursor location and target location, return true if center of cursor is inside target (has to be close enough to center to be fully inside)
        c = self.cursor.xfm.move
        t = self.origin_target.xfm.move
        d = np.sqrt((c[0]-t[0])**2 + (c[1]-t[1])**2 + (c[2]-t[2])**2)
        return d <= self.origin_target.radius - self.cursor.radius

    def _test_leave_early(self, ts):
        c = self.cursor.xfm.move
        t = self.origin_target.xfm.move
        d = np.sqrt((c[0]-t[0])**2 + (c[1]-t[1])**2 + (c[2]-t[2])**2)
        rad = self.origin_target.radius - self.cursor.radius
        return d > rad * self.exit_radius

    def _test_hold(self, ts):
        return ts>=self.origin_hold_time

    def _test_penalty_end(self, ts):
        if self.state=="fixation_penalty":
            return ts>self.fixation_penalty_time
        else:
            return ts>self.hold_penalty_time
            
    def _test_tried_enough(self, ts):
        return self.tries == 3

        
    def _test_not_tried_enough(self, ts):
        return self.tries != 3
    
    def _update(self, pt):
        if len(pt) > 0:
            self.cursor.translate(*pt[:3],reset=True)
        #if no data has come in for at least 3 frames, hide cursor
        elif self.no_data_count > 2:
            self.no_data_count += 1
            self.cursor.detach()
            self.requeue()
        else:
            self.no_data_count +=1

    def update_cursor(self):
        #get data from 1st marker on motion tracker- take average of all data points since last poll
        pt = self.motiondata.get()
        if len(pt) > 0:
            pt = pt[:, 14, :]
            # NOTE!!! The marker on the hand was changed from #0 to #14 on
            # 5/19/13 after LED #0 broke. All data files saved before this date
            # have LED #0 controlling the cursor.
            conds = pt[:, 3]
            inds = np.nonzero((conds>=0) & (conds!=4))
            if len(inds[0]) > 0:
                pt = pt[inds[0],:3]
                #convert units from mm to cm and scale to desired amount
                pt = pt.mean(0) * .1 * self.scale_factor
                #ignore y direction
                pt[1] = 0
                #move cursor to marker location
                self._update(pt)
            else:
                self.no_data_count += 1
        else:
            self.no_data_count +=1
        #write to screen
        self.draw_world()

    def calc_trial_num(self):
        '''Calculates the current trial count'''
        trialtimes = [state[1] for state in self.state_log if state[0] in ['reward', 'timeout_penalty', 'hold_penalty']]
        return len(trialtimes)

    def calc_rewards_per_min(self, window):
        '''Calculates the Rewards/min for the most recent window of specified number of seconds in the past'''
        rewardtimes = np.array([state[1] for state in self.state_log if state[0]=='reward'])
        if (self.get_time() - self.task_start_time) < window:
            divideby = (self.get_time() - self.task_start_time)/60.0
        else:
            divideby = window/60.0
        return np.sum(rewardtimes >= (self.get_time() - window))/divideby

    def calc_success_rate(self, window):
        '''Calculates the rewarded trials/initiated trials for the most recent window of specified length in sec'''
        trialtimes = np.array([state[1] for state in self.state_log if state[0] in ['reward', 'timeout_penalty', 'hold_penalty']])
        rewardtimes = np.array([state[1] for state in self.state_log if state[0]=='reward'])
        if len(trialtimes) == 0:
            return 0.0
        else:
            return float(np.sum(rewardtimes >= (self.get_time() - window)))/np.sum(trialtimes >= (self.get_time() - window))

    def update_report_stats(self):
        '''Function to update any relevant report stats for the task. Values are saved in self.reportstats,
        an ordered dictionary. Keys are strings that will be displayed as the label for the stat in the web interface,
        values can be numbers or strings. Called every time task state changes.'''
        super(TargetCapture, self).update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_rewards_per_min(120),decimals=2)
        self.reportstats['Success rate'] = str(np.round(self.calc_success_rate(120)*100.0,decimals=2)) + '%'
        
    def _while_wait(self):
        self.update_cursor()
    def _while_origin(self):
        self.update_cursor()
    def _while_origin_hold(self):
        self.update_cursor()
    def _while_fixation_penalty(self):
        self.update_cursor()
    def _while_hold_penalty(self):
        self.update_cursor()
    def _while_reward(self):
        self.update_cursor()
    def _while_pre_target_change(self):
        self.update_cursor()
    def _while_target_change(self):
        self.update_cursor()

class TargetDirection(TargetCapture):
    #only works in 2D!
    status = dict(
        wait = dict(start_trial="origin", stop=None),
        origin = dict(enter_target="origin_hold", leave_zone="direction_penalty", stop=None),
        origin_hold = dict(leave_early="hold_penalty", hold="reward"),
        reward = dict(reward_end="target_change"),
        hold_penalty = dict(penalty_end="pre_target_change"),
        direction_penalty = dict(penalty_end="pre_target_change"),
        pre_target_change = dict(tried_enough = 'target_change', not_tried_enough='wait'),
        target_change = dict(target_change_end='wait')
    )
        
    direction_penalty_time = traits.Float(1, desc="Length of penalty time for moving cursor in wrong direction")
    angle = traits.Float(90, desc="Angular size of movement zone in degrees")
    angular_limit = []
    target_angle = []
    starting_point = []
    start_distance = 5 #how far from start position cursor has to be before it will start checking the angle

    def __init__(self, *args, **kwargs):
        super(TargetDirection, self).__init__(*args, **kwargs)

    def _start_origin(self):
        super(TargetDirection, self)._start_origin()
        #get initial cursor position
        c = self.cursor.xfm.move
        #save it
        self.starting_point = c.copy()
        #get target position
        t = self.origin_target.xfm.move
        #vector between start position and target
        v = t-c
        #angle of that vector
        self.target_angle = np.arctan2(v[2],v[0])
        #get components of vector in target direction with magnitude = to target radius
        x = self.origin_target.radius*np.cos(self.target_angle)
        y = self.origin_target.radius*np.sin(self.target_angle)
        #get components of vector from starting position to edge of target
        x_edge = v[0]+y
        y_edge = v[2]+x
        #angle between edge vector and target vector
        diff = np.arctan2(y_edge,x_edge) - self.target_angle
        if diff<np.pi*(-1):
            diff += 2*np.pi
        if diff>np.pi:
            diff -= 2*np.pi
        #make sure self.anglular_limit is big enough to at least cover the entire target, otherwise leave it as user inputed value
        if np.absolute(diff)>self.angle*np.pi/180/2:
            self.angular_limit = np.absolute(diff)
        else:
            self.angular_limit = np.absolute(self.angle*np.pi/180/2)

    def _start_direction_penalty(self):
        #hide target
        self.origin_target.detach()
        self.requeue()
        self.tries += 1

    def _test_leave_zone(self, ts):
        #get current cursor position and vector from starting point to current position
        c = self.cursor.xfm.move
        v = c - self.starting_point
        #angle of the vector
        cursor_angle = np.arctan2(v[2],v[0])
        #angular difference between cursor direction and target direction from starting point
        angle_diff = cursor_angle - self.target_angle
        if angle_diff<np.pi*(-1):
            angle_diff += 2*np.pi
        if angle_diff>np.pi:
            angle_diff -= 2*np.pi
        #distance from starting point to current position
        d = np.sqrt((v[0])**2 + (v[2])**2)
        #true if cursor is at least start.distance away from starting point and outside angular boundaries
        return d>(self.start_distance) and np.absolute(angle_diff)>self.angular_limit

    def _test_penalty_end(self, ts):
        if self.state=="direction_penalty":
            return ts>self.direction_penalty_time
        if self.state=="fixation_penalty":
            return ts>self.fixation_penalty_time
        else:
            return ts>self.hold_penalty_time

    def _while_direction_penalty(self):
        self.update_cursor()

class ManualControl(TargetCapture):
    status = dict(
        wait = dict(start_trial="origin", stop=None),
        origin = dict(enter_target="origin_hold", stop=None),
        origin_hold = dict(leave_early="hold_penalty", hold="terminus"),
        terminus = dict(timeout="timeout_penalty", enter_target="terminus_hold", stop=None),
        timeout_penalty = dict(penalty_end="pre_target_change"),
        terminus_hold = dict(leave_early="hold_penalty", hold="reward"),
        reward = dict(reward_end="target_change"),
        hold_penalty = dict(penalty_end="pre_target_change"),
        pre_target_change = dict(tried_enough = 'target_change', not_tried_enough='wait'),
        target_change = dict(target_change_end='wait')
    )

    #create settable traits
    terminus_size = traits.Float(1, desc="Radius of terminus targets")
    terminus_hold_time = traits.Float(2, desc="Length of hold required at terminus")
    timeout_time = traits.Float(10, desc="Time allowed to go between origin and terminus")
    timeout_penalty_time = traits.Float(3, desc="Length of penalty time for timeout error")
    
    #create fixation point, targets, cursor objects, initialize
    def __init__(self, *args, **kwargs):
        # Add the target and cursor locations to the task data to be saved to
        # file
        self.dtype = [('target', 'f', (3,)), ('cursor', 'f', (3,))]
        super(ManualControl, self).__init__(*args, **kwargs)
        self.terminus_target = Sphere(radius=self.terminus_size, color=(1,0,0,.5))
        self.add_model(self.terminus_target)
        # Initialize target location variables
        self.location = np.array([0,0,0])
        self.target_xz = np.array([0,0])
        
    def _start_wait(self):
        super(ManualControl, self)._start_wait()
        #set target colors
        self.terminus_target.color = (1,0,0,.5)
        #hide targets from previous trial
        self.show_terminus(False)

    def _start_origin(self):
        if self.tries == 0:
            #retrieve location of next terminus target
            t = self.next_trial.T[1]
            #move target to correct location
            self.terminus_target.translate(*t, reset=True)
        super(ManualControl, self)._start_origin()

    def _start_origin_hold(self):
        #make terminus target visible
        self.show_terminus(True)
        
    def show_terminus(self, show=False):
        if show:
            self.terminus_target.attach()
        else:
            self.terminus_target.detach()
        self.requeue()

    def _start_terminus(self):
        self.show_origin(False)
    
    def _end_terminus_hold(self):
        self.terminus_target.color = (0,1,0,0.5)

    def _start_hold_penalty(self):
        #hide targets
        super(ManualControl, self)._start_hold_penalty()
        self.show_terminus(False)
    
    def _start_timeout_penalty(self):
        #hide targets and fixation point
        self.tries += 1
        self.show_terminus(False)
    
    def _start_reward(self):
        pass

    def _test_enter_target(self, ts):
        #get the current cursor location and target location, return true if center of cursor is inside target (has to be close enough to center to be fully inside)
        if self.state=="origin":
            c = self.cursor.xfm.move
            t = self.origin_target.xfm.move
            d = np.sqrt((c[0]-t[0])**2 + (c[1]-t[1])**2 + (c[2]-t[2])**2)
            return d <= self.origin_target.radius - self.cursor.radius
        if self.state=="terminus":
            c = self.cursor.xfm.move
            t = self.terminus_target.xfm.move
            d = np.sqrt((c[0]-t[0])**2 + (c[1]-t[1])**2 + (c[2]-t[2])**2)
            return d <= self.terminus_target.radius - self.cursor.radius
        
    def _test_leave_early(self, ts):
        if self.state=="origin_hold":
            c = self.cursor.xfm.move
            t = self.origin_target.xfm.move
            d = np.sqrt((c[0]-t[0])**2 + (c[1]-t[1])**2 + (c[2]-t[2])**2)
            rad = self.origin_target.radius - self.cursor.radius
            return d > rad * self.exit_radius
        if self.state=="terminus_hold":
            c = self.cursor.xfm.move
            t = self.terminus_target.xfm.move
            d = np.sqrt((c[0]-t[0])**2 + (c[1]-t[1])**2 + (c[2]-t[2])**2)
            rad = self.terminus_target.radius - self.cursor.radius
            return d > rad * self.exit_radius

    def _test_hold(self, ts):
        if self.state=="origin_hold":
            return ts>=self.origin_hold_time
        else:
            return ts>=self.terminus_hold_time

    def _test_timeout(self, ts):
        return ts>self.timeout_time

    def _test_penalty_end(self, ts):
        if self.state=="timeout_penalty":
            return ts>self.timeout_penalty_time
        if self.state=="fixation_penalty":
            return ts>self.fixation_penalty_time
        else:
            return ts>self.hold_penalty_time
    
    def _while_terminus(self):
        self.update_cursor()
    def _while_terminus_hold(self):
        self.update_cursor()
    def _while_timeout_penalty(self):
        self.update_cursor()

    def update_target_location(self):
        # Determine the task target for assist/decoder adaptation purposes (convert
        # units from cm to mm for decoder)
        # TODO - decide what to do with y location, target_xz ignores it!
        if self.state=='origin' or self.state=='origin_hold':
            self.location = 10*self.origin_target.xfm.move
            self.target_xz = np.array([self.location[0], self.location[2]])
        elif self.state=='terminus' or self.state=='terminus_hold':
            self.location = 10*self.terminus_target.xfm.move
            self.target_xz = np.array([self.location[0], self.location[2]])
        self.task_data['target'] = self.location[:3]

    def update_cursor(self):
        self.update_target_location()
        super(ManualControl, self).update_cursor()
        self.task_data['cursor'] = self.cursor.xfm.move.copy()

class Test(Sequence, FixationTraining):
    def __init__(self, *args, **kwargs):
        # Add the target and cursor locations to the task data to be saved to
        # file
        super(Test, self).__init__(*args, **kwargs)

class ManualControl2(ManualControl):
    status = dict(
        wait = dict(start_trial="origin", stop=None),
        origin = dict(enter_target="origin_hold", stop=None),
        origin_hold = dict(leave_early="hold_penalty", hold="terminus"),
        terminus = dict(timeout="timeout_penalty", enter_target="terminus_hold", stop=None),
        timeout_penalty = dict(penalty_end="pre_target_change"),
        terminus_hold = dict(leave_early="hold_penalty", hold="terminus2"),
        terminus2 = dict(timeout="timeout_penalty", enter_target="terminus2_hold", stop=None),
        terminus2_hold = dict(leave_early="hold_penalty", hold="reward"),
        reward = dict(reward_end="target_change"),
        hold_penalty = dict(penalty_end="pre_target_change"),
        pre_target_change = dict(tried_enough = 'target_change', not_tried_enough='wait'),
        target_change = dict(target_change_end='wait')
    )

    scale_factor=2
    cursor_radius = .4

    def __init__(self, *args, **kwargs):
        # Add the 2nd terminus target
        super(ManualControl2, self).__init__(*args, **kwargs)
        self.terminus2_target = Sphere(radius=self.terminus_size, color=(1,0,0,.5))
        self.add_model(self.terminus2_target)

    def _start_wait(self):
        #set target colors
        self.terminus2_target.color = (1,0,0,.5)
        #hide targets from previous trial
        self.terminus2_target.detach()
        super(ManualControl2, self)._start_wait()

    def _test_enter_target(self, ts):
        #get the current cursor location and target location, return true if center of cursor is inside target (has to be close enough to center to be fully inside)
        if self.state=="terminus2":
            c = self.cursor.xfm.move
            t = self.terminus2_target.xfm.move
            d = np.sqrt((c[0]-t[0])**2 + (c[1]-t[1])**2 + (c[2]-t[2])**2)
            return d <= self.terminus_target.radius - self.cursor.radius
        else:
            return super(ManualControl2, self)._test_enter_target(ts)
    
    def _start_origin(self):
        if self.tries == 0:
            #retrieve location of next terminus target
            t2 = self.next_trial.T[2]
            #move target to correct location
            self.terminus2_target.translate(*t2, reset=True)
        super(ManualControl2, self)._start_origin()

    def _start_terminus_hold(self):
        self.terminus2_target.color = (1,0,0,0.5)
        self.terminus2_target.attach()
        self.requeue()

    def _start_timeout_penalty(self):
        #hide targets and fixation point
        self.terminus2_target.detach()
        super(ManualControl2, self)._start_timeout_penalty()

    def _start_terminus2(self):
        self.terminus_target.detach()
        self.requeue()

    def _test_leave_early(self, ts):
        if self.state=="terminus2_hold":
            c = self.cursor.xfm.move
            t = self.terminus2_target.xfm.move
            d = np.sqrt((c[0]-t[0])**2 + (c[1]-t[1])**2 + (c[2]-t[2])**2)
            rad = self.terminus_target.radius - self.cursor.radius
            return d > rad * self.exit_radius
        else:
            return super(ManualControl2, self)._test_leave_early(ts)

    def _while_terminus2(self):
        self.update_cursor()

    def _while_terminus2_hold(self):
        self.update_cursor()

    def _end_terminus2_hold(self):
        self.terminus2_target.color = (0,1,0,0.5)

    def _start_hold_penalty(self):
        self.terminus2_target.detach()
        super(ManualControl2, self)._start_hold_penalty()
    
    def _start_timeout_penalty(self):
        self.terminus2_target.detach()
        super(ManualControl2, self)._start_timeout_penalty()

    def update_target_location(self):
        # Determine the task target for assist/decoder adaptation purposes (convert
        # units from cm to mm for decoder)
        # TODO - decide what to do with y location, target_xz ignores it!
        if self.state=='terminus2' or self.state=='terminus2_hold':
            self.location = 10*self.terminus2_target.xfm.move
            self.target_xz = np.array([self.location[0], self.location[2]])
        super(ManualControl2, self).update_target_location()

class JoystickControl(ManualControl):
    #create settable traits
    joystick_method = traits.Float(1,desc="3: Max velocity, 2: Non-linear, 1: Normal velocity, 0: Position control")
    joystick_speed = traits.Float(50,desc="scaling factor for speed of cursor (veloc. control only)")
    joystick_max_speed = traits.Float(10, desc="For ''Max velocity control (opt. 3)'' determine max velocity")
    
    move_dist = traits.Float(.1,desc="min distance for movement")
    move_reward_time = traits.Float(0,desc="Length of Juice Reward for Moving")
    
    origin_reward_time = traits.Float(0.1,desc="Length of Juice Reward Getting Origin Target")
    
    enough_tries = traits.Float(1,desc="Number of Tries Before Target Switches")
    targ_repeats = traits.Float(0,desc="Number of Times to Repeat a Target")


    #Track performance traits inherited from 'experiment.py'
    ntrials = 0 #Number of trials
    nrewards = 0 #Number of rewards
    reward_len = 0 #Length of reward 
    
    status = dict(
        wait = dict(start_trial="origin", stop=None),
        origin = dict(enter_target="origin_hold", stop=None),
        #origin_hold = dict(leave_early="hold_penalty", hold="origin_reward"), #Removed Origin reward
        #origin_reward = dict(origin_reward_end="terminus"),
        origin_hold = dict(leave_early="hold_penalty", hold="terminus"),
        terminus = dict(timeout="timeout_penalty", enter_target="terminus_hold", movement="move_reward", stop=None),
        timeout_penalty = dict(penalty_end="pre_target_change"),
        move_reward = dict(move_reward_end="terminus"),
        terminus_hold = dict(leave_early="hold_penalty", hold="reward"),
        reward = dict(reward_end="pre_target_change"),
        hold_penalty = dict(penalty_end="pre_target_change"),
        pre_target_change = dict(tried_enough = 'target_change', not_tried_enough='wait'),
        target_change = dict(target_change_end='wait')
    )

    def __init__(self, *args, **kwargs):
        super(JoystickControl, self).__init__(*args, **kwargs)
        self.last_pt=np.zeros([3]) #Just store x and z axis
        self.current_pt=np.zeros([3])
        self.last_rew_pt=np.zeros([3])
        self.targ_repeats_cnt=0

        #Create flag for if origin has been counted:
        self.orig_cnt_flag = 0

    def _start_wait(self):
        super(JoystickControl, self)._start_wait()

    def _start_origin(self):
        self.orig_cnt_flag = 0 #New origin, hasn't been tracked yet
        if self.tries == 0:

            #retrieve location of next terminus target
            t = self.next_trial.T[1]
            self.terminus_target.translate(*t, reset=True)

            #Turn self.tries flag to 1 (counting starting at 1 
            #since self.tries = 0 is also a target switching flag)
            self.tries = 1
        super(JoystickControl, self)._start_origin()

    def _while_terminus(self):
        #Count trial if he moves outside radius of origin + 1
        if self.orig_cnt_flag == 0: 
            if np.sqrt(sum(self.current_pt**2)) > (self.origin_target.radius + 1):
                self.orig_cnt_flag = 1
                self.ntrials +=1
        super(JoystickControl, self)._while_terminus()

    def _test_tried_enough(self, ts):

        #If tried enough on one iteration, increase counter
        if self.tries>self.enough_tries:
            self.targ_repeats_cnt +=1
            self.tries = 1 #Reset self_tries to 1 (not zero)
        
        #Test counter for if finished
        return self.targ_repeats_cnt>self.targ_repeats

    def _test_not_tried_enough(self, ts):
        return self.targ_repeats_cnt<=self.targ_repeats
       
    #Copied below function to turn on reward from features.py
    def _start_origin_reward(self):
        if self.reward is not None:
            self.reward.reward(self.origin_reward_time*1000.)

    def _while_origin_reward(self):
        self.update_cursor()
            
    def _test_origin_reward_end(self,ts):
        return ts > self.origin_reward_time

    def _start_target_change(self):
        self.tries = 0
        self.targ_repeats_cnt = 0

    def _test_target_change_end(self,ts):
        return True

    def _start_reward(self):
        self.targ_repeats_cnt = self.targ_repeats_cnt +1
        self.nrewards += 1
        pass
            
    def _start_move_reward(self):
        if self.reward is not None:
            self.reward.reward(self.move_reward_time*1000.)

    def _while_move_reward(self):
        self.update_cursor()
    
    def _test_move_reward_end(self,ts):
        if self.move_reward_time>0:
           self.last_rew_pt =  self.current_pt.copy()
           return ts > self.move_reward_time
        else:
           return False

    def _test_movement_pk(self,ts):
        if self.move_reward_time>0: 
           d = (sum ( (self.current_pt - self.last_rew_pt) **2) )**.5
           return d > self.move_dist
        else: 
           return False

    def update_cursor(self):
        '''
        Joystick params is an object with the following defined: 
        joystick_params.method : either "position" or "velocity"
        joystick_params.cursor_speed : if velocity, used for scaling
        '''
        
        #get data from joystick
        pt = self.joystick.get()

        if len(pt) > 0:
            pt = pt[-1][0]
            

            pt[0]=1-pt[0]; #Switch L / R axes
            #tmp = np.array([(pt[0]-0.503), 0, 0.507-pt[1]])
            #print tmp
            #print pt
            calib = [0.497,0.517] 
            '''Sometimes zero point is subject to drift
            #this is the value of the incoming joystick when at 'rest' '''

            if self.joystick_method==0:
                pos = np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                pos[0] = pos[0]*36
                pos[2] = pos[2]*24
                self.current_pt = pos
                
            elif self.joystick_method==1:
                vel=np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                epsilon = 2*(10**-2) #Define epsilon to Ensure micromotion / noise in joystick doesn't cause drift
                if sum((vel)**2) > epsilon:
                    self.current_pt=self.last_pt+self.joystick_speed*vel*(1/60) #60 Hz update rate, dt = 1/60
                else:
                    self.current_pt = self.last_pt
                
                #Store Current Pt/Time as last_pt for next iteration
                self.last_pt=self.current_pt.copy()

            elif self.joystick_method==2:
                '''Use a non-linear mapping of joystick position to 
                cursor position to give him better control of the joystick 
                in close proximity to the origin'''
                joystick_input = np.array([pt[0]-0.003, 0, 1.007-pt[1]])
                cursor_output = np.zeros([3])
                #Empirically determined function for control:
                cursor_output[0] = np.array(-5*np.log((1-joystick_input[0])/joystick_input[0]))
                cursor_output[2] = np.array(-3*np.log((1-joystick_input[2])/joystick_input[2]))
                self.current_pt = cursor_output
        
            elif self.joystick_method==3: 
                '''Position control, except also tracking velocity and restricting 
                position movement if velocity is too high'''
                pos = np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                pos[0] = pos[0]*36
                pos[2] = pos[2]*24
                
                vel = (pos - self.last_pt)*(60)/self.joystick_speed
                speed = (sum(vel**2))**0.5
                #print speed
                if speed > self.joystick_max_speed:
                    alph = self.joystick_max_speed * (1/60) / (speed*self.joystick_speed)
                    self.current_pt = self.last_pt + (pos - self.last_pt)*alph 
                else:
                    self.current_pt = pos
                    
            #Keep cursor in boundaries of screen
            bound = np.array([18, 0, 12])
            tmp = self.current_pt + bound
            for i in np.arange(0,len(tmp)):
                if tmp[i] > 2*bound[i]:
                    tmp[i]=2*(bound[i]-1)
                    #print tmp-bound
                elif tmp[i] < 0:
                    tmp[i]=1
                    #print tmp-bound
            self.current_pt = tmp - bound

            #Update screen
            self._update(np.array(self.current_pt))

            #Update 'last_pt' for next iteration
            self.last_pt=self.current_pt.copy()

        #write to screen
        self.draw_world()    

        #Save Cursor position to HDF
        self.task_data['cursor']=np.array(self.current_pt) 

        #Save Target position to HDF (super form )
        self.update_target_location()


if __name__ == "__main__":
    from riglib.experiment import make
    from riglib.experiment.generate import runseq
    from riglib.experiment.features import SimulatedEyeData, MotionData, RewardSystem
    seq = rand_target_sequence()
    Exp = make(ManualControl, (SimulatedEyeData, MotionData, RewardSystem))
    exp = Exp(runseq(Exp, seq), fixations=[(0,0)])
    exp.run()
