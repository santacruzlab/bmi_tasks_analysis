'''
Base tasks for generic point-to-point reaching
'''
from __future__ import division
import numpy as np
from collections import OrderedDict
import time

from riglib import reward
from riglib.experiment import traits, Sequence

from riglib.stereo_opengl.window import Window, FPScontrol, WindowDispl2D
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere, Cube
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from plantlist import plantlist

from riglib.stereo_opengl import ik

import math
import traceback

####### CONSTANTS
sec_per_min = 60.0
RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)
GOLD = (1., 0.843, 0., 0.5)
mm_per_cm = 1./10

class CircularTarget(object):
    def __init__(self, target_radius=2, target_color=(1, 0, 0, .5), starting_pos=np.zeros(3)):
        self.target_radius = target_radius
        self.target_color = target_color
        self.position = starting_pos
        self.int_position = starting_pos
        self._pickle_init()

    def _pickle_init(self):
        self.sphere = Sphere(radius=self.target_radius, color=self.target_color)
        self.graphics_models = [self.sphere]
        self.sphere.translate(*self.position)

    def move_to_position(self, new_pos):
        self.int_position = new_pos
        self.drive_to_new_pos()

    def drive_to_new_pos(self):
        raise NotImplementedError 

class VirtualCircularTarget(CircularTarget):
    def drive_to_new_pos(self):
        self.position = self.int_position
        self.sphere.translate(*self.position, reset=True)

    def hide(self):
        self.sphere.detach()

    def show(self):
        self.sphere.attach()

    def cue_trial_start(self):
        self.sphere.color = RED
        self.show()

    def cue_trial_end_success(self):
        self.sphere.color = GREEN

    def cue_trial_end_failure(self):
        self.sphere.color = RED
        self.hide()
        # self.sphere.color = GREEN
    def turn_yellow(self):
        self.sphere.color = GOLD

    def idle(self):
        self.sphere.color = RED
        self.hide()


class RectangularTarget(object):
    def __init__(self, target_width=4, target_height=4, target_color=(1, 0, 0, .5), starting_pos=np.zeros(3)):
        self.target_width = target_width
        self.target_height = target_height
        self.target_color = target_color
        self.default_target_color = tuple(self.target_color)
        self.position = starting_pos
        self.int_position = starting_pos
        self._pickle_init()

    def _pickle_init(self):
        self.cube = Cube(side_len=self.target_width, color=self.target_color)
        self.graphics_models = [self.cube]
        self.cube.translate(*self.position)

    def move_to_position(self, new_pos):
        self.int_position = new_pos
        self.drive_to_new_pos()

    def drive_to_new_pos(self):
        raise NotImplementedError 


class VirtualRectangularTarget(RectangularTarget):
    def drive_to_new_pos(self):
        self.position = self.int_position
        self.cube.translate(*self.position, reset=True)

    def hide(self):
        self.cube.detach()

    def show(self):
        self.cube.attach()

    def cue_trial_start(self):
        self.cube.color = RED
        self.show()

    def cue_trial_end_success(self):
        self.cube.color = GREEN

    def cue_trial_end_failure(self):
        self.cube.color = RED
        self.hide()

    def idle(self):
        self.cube.color = RED
        self.hide()

    def pt_inside(self, pt):
        '''
        Test if a point is inside the target
        '''
        pos = self.cube.xfm.move
        # TODO this currently assumes that the cube doesn't rotate
        # print (pt[0] - pos[0]), (pt[2] - pos[2])
        return (np.abs(pt[0] - pos[0]) < self.target_width/2) and (np.abs(pt[2] - pos[2]) < self.target_height/2)

    def reset(self):
        self.cube.color = self.default_target_color

    def get_position(self):
        return self.cube.xfm.move


class ManualControlMulti(Sequence, Window):
    '''
    This is an improved version of the original manual control tasks that includes the functionality
    of ManualControl, ManualControl2, and TargetCapture all in a single task. This task doesn't
    assume anything about the trial structure of the task and allows a trial to consist of a sequence
    of any number of sequential targets that must be captured before the reward is triggered. The number
    of targets per trial is determined by the structure of the target sequence used.
    '''

    background = (0,0,0,1)
    
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    cursor_color = (.5,0,.5,1)

    #plant_type_options = plantlist.keys()
    #plant_type = traits.Enum(*plantlist)
    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())
    starting_pos = (5, 0, 5)
    window_size = traits.Tuple((1920*2, 1080), descr='window size')

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    trial_end_states = ['reward', 'timeout_penalty', 'hold_penalty']

    #initial state
    state = "wait"

    #create settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")
    target_radius = traits.Float(2, desc="Radius of targets in cm")
    
    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')
    session_length = traits.Float(0, desc="Time until task automatically stops. Length of 0 means no auto stop.")
    marker_num = traits.Int(14, desc="The index of the motiontracker marker to use for cursor position")
    # NOTE!!! The marker on the hand was changed from #0 to #14 on
    # 5/19/13 after LED #0 broke. All data files saved before this date
    # have LED #0 controlling the cursor.

    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    target_color = (1,0,0,.5)
    target_index = -1 # Helper variable to keep track of which target to display within a trial
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.
    
    cursor_visible = False # Determines when to hide the cursor.
    no_data_count = 0 # Counter for number of missing data frames in a row
    scale_factor = 3.0 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)

    limit2d = 1

    sequence_generators = ['point_to_point_3D', 'centerout_3D', 'centerout_3D_cube', 'centerout_2D_discrete', 'centerout_2D_discrete_upper','centerout_2D_discrete_rot', 'centerout_2D_discrete_multiring',
        'centerout_2D_discrete_randorder', 'centeroutback_2D', 'centeroutback_2D_farcatch', 'centeroutback_2D_farcatch_discrete',
        'outcenterout_2D_discrete', 'outcenter_2D_discrete', 'rand_target_sequence_3d', 'rand_target_sequence_2d', 'rand_target_sequence_2d_centerout',
        'rand_target_sequence_2d_partial_centerout', 'rand_multi_sequence_2d_centerout2step', 'rand_pt_to_pt',
        'centerout_2D_eyetracker']
    is_bmi_seed = True
    
    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        # Initialize the plant
        if not hasattr(self, 'plant'):
            self.plant = plantlist[self.plant_type]
        self.plant_vis_prev = True

        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=RED)
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=RED)

            self.targets = [target1, target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
        
        # Initialize target location variable
        self.target_location = np.array([0, 0, 0])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

    def init(self):
        self.add_dtype('target', 'f8', (3,))
        self.add_dtype('target_index', 'i', (1,))
        super(ManualControlMulti, self).init()

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['target'] = self.target_location.copy()
        self.task_data['target_index'] = self.target_index

        ## Run graphics commands to show/hide the plant if the visibility has changed
        if self.plant_type != 'CursorPlant':
            if self.plant_visible != self.plant_vis_prev:
                self.plant_vis_prev = self.plant_visible
                self.plant.set_visibility(self.plant_visible)
                # self.show_object(self.plant, show=self.plant_visible)

        self.move_plant()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(ManualControlMulti, self)._cycle()
        
    def move_plant(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        #get data from motion tracker- take average of all data points since last poll
        pt = self.motiondata.get()
        if len(pt) > 0:
            pt = pt[:, self.marker_num, :]
            conds = pt[:, 3]
            inds = np.nonzero((conds>=0) & (conds!=4))[0]
            if len(inds) > 0:
                pt = pt[inds,:3]
                #scale actual movement to desired amount of screen movement
                pt = pt.mean(0) * self.scale_factor
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: pt[1] = 0
                pt[1] = pt[1]*2
                # Return cursor location
                self.no_data_count = 0
                pt = pt * mm_per_cm #self.convert_to_cm(pt)
            else: #if no usable data
                self.no_data_count += 1
                pt = None
        else: #if no new data
            self.no_data_count +=1
            pt = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available
        if pt is not None:
            self.plant.set_endpoint_pos(pt)

    def run(self):
        '''
        See experiment.Experiment.run for documentation. 
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.plant.start()
        try:
            super(ManualControlMulti, self).run()
        finally:
            self.plant.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_cursor_visibility(self):
        ''' Update cursor visible flag to hide cursor if there has been no good data for more than 3 frames in a row'''
        prev = self.cursor_visible
        if self.no_data_count < 3:
            self.cursor_visible = True
            if prev != self.cursor_visible:
                self.show_object(self.cursor, show=True)
        else:
            self.cursor_visible = False
            if prev != self.cursor_visible:
                self.show_object(self.cursor, show=False)

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super(ManualControlMulti, self).update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120), decimals=2)

    #### TEST FUNCTIONS ####
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius - self.cursor_radius)
        
    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius - self.cursor_radius
        return d > rad

    def _test_hold_complete(self, ts):
        return ts>=self.hold_time

    def _test_timeout(self, ts):
        return ts>self.timeout_time

    def _test_timeout_penalty_end(self, ts):
        return ts>self.timeout_penalty_time

    def _test_hold_penalty_end(self, ts):
        return ts>self.hold_penalty_time

    def _test_trial_complete(self, ts):
        return self.target_index==self.chain_length-1

    def _test_trial_incomplete(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts)

    def _test_trial_abort(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries==self.max_attempts)

    def _test_reward_end(self, ts):
        return ts>self.reward_time

    def _test_stop(self, ts):
        if self.session_length > 0 and (self.get_time() - self.task_start_time) > self.session_length:
            self.end_task()
        return self.stop

    #### STATE FUNCTIONS ####
    def _parse_next_trial(self):
        self.targs = self.next_trial

    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets:
            target.hide()

        #get target locations for this trial
        self._parse_next_trial()
        # if self.plant_type != 'CursorPlant' and np.random.rand() < self.plant_hide_rate:
        #     self.plant_visible = False
        # else:
        #     self.plant_visible = True
        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial

    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index % 2]
        self.target_location = self.targs[self.target_index]
        target.move_to_position(self.target_location)
        target.cue_trial_start()

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx % 2]
            target.move_to_position(self.targs[idx])
    
    def _end_hold(self):
        # change current target color to green
        self.targets[self.target_index % 2].cue_trial_end_success()

    def _start_hold_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1
    
    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_targ_transition(self):
        #hide targets
        for target in self.targets:
            target.hide()

    def _start_reward(self):
        super(ManualControlMulti, self)._start_reward()
        self.targets[self.target_index % 2].show()

    #### Generator functions ####
    @staticmethod
    def point_to_point_3D(length=2000, boundaries=(-18,18,-10,10,-15,15), distance=10, chain_length=2):
        '''
        Generates sets of randomly located targets. Each trial will have the number of targets specified in the chain_length
        argument. 
        '''

        #inner_bounds = 
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

    @staticmethod
    def centerout_3D(length=1000, boundaries=(-18,18,-10,10,-15,15),distance=8):
        # Choose a random sequence of points on the surface of a sphere of radius
        # "distance"
        theta = np.random.rand(length)*2*np.pi
        phi = np.arccos(2*np.random.rand(length) - 1)
        x = distance*np.cos(theta)*np.sin(phi)
        y = distance*np.sin(theta)*np.sin(phi)
        z = distance*np.cos(theta)

        pairs = np.zeros([length,2,3])
        pairs[:,1,0] = x
        pairs[:,1,1] = y
        pairs[:,1,2] = z

        return pairs

    @staticmethod
    def centerout_3D_cube(length=1000, edge_length=8):
        '''
        Choose a random sequence of points on the surface of a sphere of radius
        "distance"
        '''
        coord = [-float(edge_length)/2, float(edge_length)/2]
        from itertools import product
        target_locs = [(x, y, z) for x, y, z in product(coord, coord, coord)]
        
        n_corners_in_cube = 8
        pairs = np.zeros([length, 2, 3])

        for k in range(length):
            pairs[k, 0, :] = np.zeros(3)
            pairs[k, 1, :] = target_locs[np.random.randint(0, n_corners_in_cube)]

        print pairs.shape
        return pairs

    @staticmethod
    def centerout_2D_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
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
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs

    @staticmethod
    def centerout_2D_eyetracker(nblocks=100, ntargets=4, boundaries=(-18,18,-12,12), distance=10):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin. The order required by the eye-tracker calibration
        is Center, Left, Right, Up, Down. The sequence generator therefore
        displays 4- trial sequences in the following order:
        Center-Left (C-L), C-R, C-U, C-D.

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
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''

        # Create a LRUD (Left Right Up Down) sequence of points on the edge of
        # a circle of radius "distance"

        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets) # ntargets = 4 --> shape of a +
            temp2 = np.array([temp[0],temp[2],temp[1],temp[3]]) # Left Right Up Down
            theta = theta + [temp2]
        theta = np.hstack(theta)

        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)

        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs


    @staticmethod
    def centerout_2D_discrete_upper(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        '''Same as centerout_2D_discrete, but rotates position of targets by 'rotate_deg'.
           For example, if you wanted only 1 target, but sometimes wanted it at pi and sometiems at 3pi/2,
             you could rotate it by 90 degrees'''
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, np.pi, np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs
    
    @staticmethod
    def centerout_2D_discrete_rot(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10,rotate_deg=0):
        '''Same as centerout_2D_discrete, but rotates position of targets by 'rotate_deg'.
           For example, if you wanted only 1 target, but sometimes wanted it at pi and sometiems at 3pi/2,
             you could rotate it by 90 degrees'''
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            temp = temp + (rotate_deg)*(2*np.pi/360)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs

    @staticmethod
    def centerout_2D_discrete_multiring(n_blocks=100, n_angles=8, boundaries=(-18,18,-12,12),
        distance=10, n_rings=2):
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
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations

        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        target_set = []
        angles = np.arange(0, 2*np.pi, 2*np.pi/n_angles)
        distances = np.arange(0, distance + 1, float(distance)/n_rings)[1:]
        for angle in angles:
            for dist in distances:
                targ = np.array([np.cos(angle), 0, np.sin(angle)]) * dist
                target_set.append(targ)

        target_set = np.vstack(target_set)
        n_targets = len(target_set)
        

        periph_target_list = []
        for k in range(n_blocks):
            target_inds = np.arange(n_targets)
            np.random.shuffle(target_inds)
            periph_target_list.append(target_set[target_inds])

        periph_target_list = np.vstack(periph_target_list)

        
        pairs = np.zeros([len(periph_target_list), 2, 3])
        pairs[:,1,:] = periph_target_list#np.vstack([x, y, z]).T
        
        return pairs

    @staticmethod
    def centerout_2D_discrete_randorder(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):                                                                 
        '''                                                                           
                                                                                      
        Generates a sequence of 2D (x and z) target pairs with the first target       
        always at the origin, totally randomized instead of block randomized.         
                                                                                      
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
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations         
                                                                                      
                                                                                      
        '''                                                                           
                                                                                      
        # Choose a random sequence of points on the edge of a circle of radius        
        # "distance"                                                                  
                                                                                      
        theta = []                                                                    
        for i in range(nblocks):                                                      
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)                            
            np.random.shuffle(temp)                                                   
            theta = theta + [temp]                                                    
        theta = np.hstack(theta)                                                      
        np.random.shuffle(theta)                                                      
                                                                                      
                                                                                      
        x = distance*np.cos(theta)                                                    
        y = np.zeros(len(theta))                                                      
        z = distance*np.sin(theta)                                                    
                                                                                      
        pairs = np.zeros([len(theta), 2, 3])                                          
        pairs[:,1,:] = np.vstack([x, y, z]).T                                         
                                                                                      
        return pairs

    @staticmethod
    def centeroutback_2D(length, boundaries=(-18,18,-12,12), distance=8):
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
        targs : [length x 3 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        theta = np.random.rand(length)*2*np.pi
        x = distance*np.cos(theta)
        z = distance*np.sin(theta)
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([length, 3, 3])
        targs[:,1,0] = x
        targs[:,1,2] = z
        
        return targs

    @staticmethod
    def centeroutback_2D_farcatch(length, boundaries=(-18,18,-12,12), distance=8, catchrate=.1):
        '''

        Generates a sequence of 2D (x and z) center-edge-center target triplets, with occasional
        center-edge-far edge catch trials thrown in.

        Parameters
        ----------
        length : int
            The number of target sets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.
        catchrate: float
            The percent of trials that are far target catch trials. If distance*2

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance" and a corresponding set of points on circle of raidus 2*distance
        theta = np.random.rand(length)*2*np.pi
        x = distance*np.cos(theta)
        z = distance*np.sin(theta)

        x2 = 2*x
        x2[np.nonzero((x2<boundaries[0]) | (x2>boundaries[1]))] = np.nan

        z2 = 2*z
        z2[np.nonzero((z2<boundaries[2]) | (z2>boundaries[3]))] = np.nan

        outertargs = np.zeros([length, 3])
        outertargs[:,0] = x2
        outertargs[:,2] = z2
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([length, 3, 3])
        targs[:,1,0] = x
        targs[:,1,2] = z

        # shuffle order of indices and select specified percent to use for catch trials
        numcatch = int(length*catchrate)
        shuffinds = np.array(range(length))
        np.random.shuffle(shuffinds)
        replace = shuffinds[:numcatch]
        count = numcatch

        while np.any(np.isnan(outertargs[list(replace)])):
            replace = replace[~np.isnan(np.sum(outertargs[list(replace)],axis=1))]
            diff = numcatch - len(replace)
            new = shuffinds[count:count+diff]
            replace = np.concatenate((replace, new))
            count += diff

        targs[list(replace),2,:] = outertargs[list(replace)]

        return targs

    @staticmethod
    def centeroutback_2D_farcatch_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=8, catchrate=0):
        '''

        Generates a sequence of 2D (x and z) center-edge-center target triplets, with occasional
        center-edge-far edge catch trials thrown in.

        Parameters
        ----------
        length : int
            The number of target sets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.
        catchrate: float
            The percent of trials that are far target catch trials. If distance*2

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations


        '''

       # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)

        x = distance*np.cos(theta)
        z = distance*np.sin(theta)

        # Choose a corresponding set of points on circle of radius 2*distance. Mark any points that
        # are outside specified boundaries with nans

        x2 = 2*x
        x2[np.nonzero((x2<boundaries[0]) | (x2>boundaries[1]))] = np.nan

        z2 = 2*z
        z2[np.nonzero((z2<boundaries[2]) | (z2>boundaries[3]))] = np.nan

        outertargs = np.zeros([nblocks*ntargets, 3])
        outertargs[:,0] = x2
        outertargs[:,2] = z2
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([nblocks*ntargets, 3, 3])
        targs[:,1,0] = x
        targs[:,1,2] = z

        # shuffle order of indices and select specified percent to use for catch trials
        numcatch = int(nblocks*ntargets*catchrate)
        shuffinds = np.array(range(nblocks*ntargets))
        np.random.shuffle(shuffinds)
        replace = shuffinds[:numcatch]
        count = numcatch

        while np.any(np.isnan(outertargs[list(replace)])):
            replace = replace[~np.isnan(np.sum(outertargs[list(replace)],axis=1))]
            diff = numcatch - len(replace)
            new = shuffinds[count:count+diff]
            replace = np.concatenate((replace, new))
            count += diff

        targs[list(replace),2,:] = outertargs[list(replace)]

        return targs

    @staticmethod
    def outcenterout_2D_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=8):
        '''

        Generates a sequence of 2D (x and z) center-edge-center target triplets, with occasional
        center-edge-far edge catch trials thrown in.

        Parameters
        ----------
        length : int
            The number of target sets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.
        catchrate: float
            The percent of trials that are far target catch trials. If distance*2

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations


        '''

       # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)

        x = distance*np.cos(theta)
        z = distance*np.sin(theta)
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([nblocks*ntargets, 3, 3])
        targs[:,0,0] = x
        targs[:,2,0] = x
        targs[:,0,2] = z
        targs[:,2,2] = z

        return targs

    @staticmethod
    def outcenter_2D_discrete(nblocks=100, ntargets=4, boundaries=(-18,18,-12,12),
        distance=8, startangle=np.pi/4):
        '''

        Generates a sequence of 2D (x and z) center-edge-center target triplets, with occasional
        center-edge-far edge catch trials thrown in.

        Parameters
        ----------
        length : int
            The number of target sets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.
        catchrate: float
            The percent of trials that are far target catch trials. If distance*2

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations
        '''

       # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(startangle, startangle+(2*np.pi), 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)

        x = distance*np.cos(theta)
        z = distance*np.sin(theta)
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([nblocks*ntargets, 2, 3])
        targs[:,0,0] = x
        targs[:,0,2] = z

        return targs

    @staticmethod
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

    @staticmethod
    def rand_pt_to_pt(length=100, boundaries=(-18,18,-12,12), buf=2, seq_len=2):
        '''
        Generates sequences of random postiions in the XZ plane

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
        list
            Each element of the list is an array of shape (seq_len, 3) indicating the target 
            positions to be acquired for the trial.
        '''
        xmin, xmax, zmin, zmax = boundaries
        L = length*seq_len
        pts = np.vstack([np.random.uniform(xmin+buf, xmax-buf, L),
            np.zeros(L), np.random.uniform(zmin+buf, zmax-buf, L)]).T
        targ_seqs = []
        for k in range(length):
            targ_seqs.append(pts[k*seq_len:(k+1)*seq_len])
        return targ_seqs

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


class MovementTrainingMulti(ManualControlMulti):
    '''
    Motion-tracker based task to train subjects to move in the depth dimension
    '''
    threshold_speed = traits.Float(1,desc='how far in depth he has to move')
    prev_y_positions = np.zeros(60)

    status = dict(
        wait = dict(moved='reward', stop=None),
        reward = dict(reward_end='wait'))

    sequence_generators = ['depth_trainer']

    def move_plant(self):
        super(MovementTrainingMulti, self).move_plant()
        self.prev_y_positions[1:] = self.prev_y_positions[:-1]
        self.prev_y_positions[0] = self.plant.get_endpoint_pos()[1]

    def _test_moved(self, ts):
        nback = 6
        speed = np.abs(self.prev_y_positions[0] - self.prev_y_positions[nback])/(nback/60.)
        return speed > self.threshold_speed

    ##### Generator functions
    @staticmethod
    def depth_trainer(length=1000,boundaries=(-18,18,-10,10,-15,15),distance=8):
        pairs = np.zeros([length,2,3])
        pairs[:length/2,1,1] = distance
        pairs[length/2:,1,1] = -distance
        np.random.shuffle(pairs)
        return pairs



class JoystickTentacle(ManualControlMulti):
    joystick_gain = .1

    def move_plant(self):
        #get data from phidget
        pt = self.dualjoystick.get()
        jts = self.plant.get_intrinsic_coordinates()

        if len(pt) > 0:
            pt = pt[-1][0]
            calib = [0.493, 0.494, 0.5, 0.498] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 

            vel=np.array(pt - calib)

            epsilon = 2*(10**-2) #Define epsilon to stabilize cursor movement
            if sum((vel)**2) > epsilon:
                self.plant.set_intrinsic_coordinates(jts+vel*self.joystick_gain) #60 Hz update rate, dt = 1/60


class JoystickMulti(ManualControlMulti):

    # #Settable Traits
    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    random_rewards = traits.Float(0,desc="Add randomness to reward, 1: yes, 0: no")
    joystick_speed = traits.Float(20, desc="Radius of cursor")

    is_bmi_seed = True
    
    def __init__(self, *args, **kwargs):
        super(JoystickMulti, self).__init__(*args, **kwargs)
        self.current_pt=np.zeros([3]) #keep track of current pt
        self.last_pt=np.zeros([3]) #keep track of last pt to calc. velocity
        #self.plant_visible = False

    def update_report_stats(self):
        super(JoystickMulti, self).update_report_stats()
        start_time = self.state_log[0][1]
        rewardtimes=np.array([state[1] for state in self.state_log if state[0]=='reward'])
        if len(rewardtimes):
            rt = rewardtimes[-1]-start_time
        else:
            rt= np.float64("0.0")

        sec = str(np.int(np.mod(rt,60)))
        if len(sec) < 2:
            sec = '0'+sec
        self.reportstats['Time Of Last Reward'] = str(np.int(np.floor(rt/60))) + ':' + sec
    
    def _test_trial_complete(self, ts):
        if self.target_index==self.chain_length-1 :
            if self.random_rewards:
                if not self.rand_reward_set_flag: #reward time has not been set for this iteration
                    self.reward_time = np.max([2*(np.random.rand()-0.5) + self.reward_time_base, self.reward_time_base/2]) #set randomly with min of base / 2
                    self.rand_reward_set_flag =1;
                    #print self.reward_time, self.rand_reward_set_flag
            return self.target_index==self.chain_length-1
        
    def _test_reward_end(self, ts):
        #When finished reward, reset flag. 
        if self.random_rewards:
            if ts > self.reward_time:
                self.rand_reward_set_flag = 0;
                #print self.reward_time, self.rand_reward_set_flag, ts
        return ts > self.reward_time
    def _cycle(self):
        super(JoystickMulti, self)._cycle()

    def move_plant(self):
        ''' Returns the 3D coordinates of the cursor. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        #get data from phidget
        pt = self.joystick.get()

        if len(pt) > 0:
            pt = pt[-1][0]
            pt[0]=1-pt[0]; #Switch L / R axes
            calib = [0.497,0.517] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
            # calib = [ 0.487,  0.   ]

            #if self.joystick_method==0:                
            if self.joystick_method==0:
                pos = np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                pos[0] = pos[0]*36
                pos[2] = pos[2]*24
                self.current_pt = pos

            elif self.joystick_method==1:
                vel=np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                epsilon = 2*(10**-2) #Define epsilon to stabilize cursor movement
                if sum((vel)**2) > epsilon:
                    self.current_pt=self.last_pt+self.joystick_speed*vel*(1/60) #60 Hz update rate, dt = 1/60
                else:
                    self.current_pt = self.last_pt

                if self.current_pt[0] < -25: self.current_pt[0] = -25
                if self.current_pt[0] > 25: self.current_pt[0] = 25
                if self.current_pt[-1] < -14: self.current_pt[-1] = -14
                if self.current_pt[-1] > 14: self.current_pt[-1] = 14

            self.plant.set_endpoint_pos(self.current_pt)
            self.last_pt = self.current_pt.copy()

class JoystickMultiWithStressTrials(JoystickMulti):
    '''
    Manual Control multi task with blocks of stress trials
    '''
    audio_cue = traits.Bool(True, desc='ON or OFF for audio cue on stress trials')
    stress_timeout_time = traits.Float(1.5, desc="Time allowed to go between targets on stress trials")

    def __init__(self, *args, **kwargs):
        super(JoystickMultiWithStressTrials, self).__init__(*args, **kwargs)
        self.original_timeout_time = self.timeout_time

    def init(self):
        self.add_dtype('stress_trial', 'i', (1,))
        super(JoystickMultiWithStressTrials, self).init()

    def _start_wait(self):
        super(JoystickMultiWithStressTrials, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        self.stress_trial = 0     # indicator for stress trial
        #hide targets
        for target in self.targets:
            target.hide()

        #get target locations for this trial
        self._parse_next_trial()
        # if self.plant_type != 'CursorPlant' and np.random.rand() < self.plant_hide_rate:
        #     self.plant_visible = False
        # else:
        #     self.plant_visible = True
        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial

        #if (self.calc_state_occurrences('reward') < 11):
        if self.calc_trial_num() < 11:
            self.stress_trial = 0
            self.timeout_time = self.original_timeout_time
        #if (11 <= self.calc_state_occurrences('reward') < 31):
        if (11 <= self.calc_trial_num() < 41):
            self.stress_trial = 1
            self.timeout_time = self.stress_timeout_time
        else:
            self.stress_trial = 0
            self.timeout_time = self.original_timeout_time
        #if self.calc_trial_num() > 110:  
        #    if self.calc_trial_num() % 20 < 10:
        #        self.timeout_time = self.stress_timeout_time
        #        self.stress_trial = 1

        self.task_data['stress_trial'] = self.stress_trial
        
        self.requeue()

class LeakyIntegratorVelocityJoystick(JoystickMulti):
    joystick_method = 1
    bias_angle = traits.Float(0, desc="Angle to bias cursor velocity, in degrees")
    bias_gain = traits.Float(1, desc="Gain of directional velocity bias")
    alpha = traits.Float(0.5, desc="Velocity memory factory (between 0 and 1)")

    prev_vel = np.zeros(3)

    def move_plant(self):
        ''' Returns the 3D coordinates of the cursor. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        #get data from phidget
        pt = self.joystick.get()

        if len(pt) > 0:
            pt = pt[-1][0]
            pt[0] = 1-pt[0]; #Switch L / R axes
            calib = [0.497,0.517] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 

            if 1:
                joystick_vel = np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])

                epsilon = 2*(10**-2) #Define epsilon to stabilize cursor movement
                if sum((joystick_vel)**2) < epsilon:
                    joystick_vel = np.zeros(3)

                vel = self.joystick_speed*joystick_vel + self.alpha*self.prev_vel
                self.current_pt = self.last_pt + vel*(1/60) #60 Hz update rate, dt = 1/60

                if self.current_pt[0] < -25: self.current_pt[0] = -25
                if self.current_pt[0] > 25: self.current_pt[0] = 25
                if self.current_pt[-1] < -14: self.current_pt[-1] = -14
                if self.current_pt[-1] > 14: self.current_pt[-1] = 14

                self.prev_vel = vel

            self.plant.set_endpoint_pos(self.current_pt)
            self.last_pt = self.current_pt.copy()    

class JoystickMulti_Directed(JoystickMulti):
    
    #Settable Traits
    allowed_zone_deg = traits.Float(90,desc="Angle of allowed movement from center to target")

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", exit_allowed_zone="timeout_penalty",stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
        )

    def __init__(self, *args, **kwargs):
        super(JoystickMulti_Directed, self).__init__(*args, **kwargs)
        self.target_theta = 0;

    def _start_target(self):
        super(JoystickMulti_Directed, self)._start_target()
        if self.target_index%2==1:
            targ = self.target2.xfm.move.copy()
            targ[0]=-1*targ[0]
            self.target_theta = self.get_theta(targ)
            self.theta_tol = ((self.allowed_zone_deg)*np.pi/180)
            print 'start_target',self.target_theta, self.theta_tol
    
    def _test_exit_allowed_zone(self,ts):
        exit_zone = False
        if self.target_index%2==1: #Target, not origin. 
            # Test if self.current_pt is within zone
            pt = self.current_pt.copy()
            pt[0] = -1*pt[0]
            theta = self.get_theta(pt)

            if np.sqrt(pt[0]**2 + pt[2]**2) > self.target_radius:
                print 'theta: ',theta
                d = theta-self.target_theta
                diff = np.min(abs(np.array([d+2*np.pi, d-2*np.pi, d])))
                print 'diff: ',d+2*np.pi, d-2*np.pi, d, diff
                if diff > self.theta_tol:
                    exit_zone = True
        return exit_zone

    def get_theta(self, pt):
        
        theta = math.atan2(pt[2],pt[0])
        if theta < 0:
            theta = theta + 2*np.pi
        return theta

       
class JoystickMulti_plusMove(JoystickMulti):

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", move="move_reward", stop=None),
        move_reward = dict(move_reward_end="target"),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
        )

    #Settable Traits
    move_reward_time = traits.Float(1,desc="Reward Time for moving Joystick")
    #move_dist = traits.Float(5,desc="Minimum Distance to move on screen for Reward")
    move_dist_x = traits.Float(5,desc="Minimum Distance to move on screen for Reward")
    move_dist_y = traits.Float(5,desc="Minimum Distance to move on screen for Reward")

    def __init__(self, *args, **kwargs):
        super(JoystickMulti_plusMove, self).__init__(*args, **kwargs)
        self.last_rew_pt = np.zeros([3])

    def _start_move_reward(self):
        #if self.reward is not None:
        #    self.reward.reward(self.move_reward_time*1000.)
        self._start_reward()
  
    def _test_move_reward_end(self,ts):
        if self.move_reward_time>0:
           self.last_rew_pt =  self.current_pt.copy()
           if ts > self.move_reward_time: #to compensate for adding +1 to target index for moving. 
                self.target_index += -1
           return ts > self.move_reward_time
        else:
           return False

    def _end_move_reward(self):
        self._end_reward()

    def _test_move(self,ts):
        if self.move_reward_time>0: 
            d_x =  ( (self.current_pt[0] - self.last_rew_pt[0]) **2)  **.5
            d_y =  ( (self.current_pt[2] - self.last_rew_pt[2]) **2)  **.5

            if d_x > self.move_dist_x or d_y > self.move_dist_y:
                r=1
            else:
                r=0
            return r

        else: 
           return False

class JoystickMove(JoystickMulti_plusMove):
    status = dict(
        wait = dict(start_trial = "show_cursor", stop=None),
        show_cursor = dict(move = "move_reward", stop = None),
        move_reward = dict(move_reward_end="show_cursor"),
        )
        
    def __init__(self, *args, **kwargs):
        super(JoystickMove, self).__init__(*args, **kwargs)
        self.last_rew_pt = np.zeros([3])

    def _start_show_cursor(self):
        self.move_plant()


class TestGraphics(Sequence, Window):
    status = dict(
        wait = dict(stop=None),
    )

    #initial state
    state = "wait"
    target_radius = 2.
    
    #create targets, cursor objects, initialize
    def __init__(self, *args, **kwargs):
        # Add the target and cursor locations to the task data to be saved to
        # file
        super(TestGraphics, self).__init__(*args, **kwargs)
        self.dtype = [('target', 'f', (3,)), ('cursor', 'f', (3,)), (('target_index', 'i', (1,)))]
        self.target1 = Sphere(radius=self.target_radius, color=(1,0,0,.5))
        self.add_model(self.target1)
        self.target2 = Sphere(radius=self.target_radius, color=(1,0,0,.5))
        self.add_model(self.target2)
        
        # Initialize target location variable
        self.target_location = np.array([0,0,0])

    ##### HELPER AND UPDATE FUNCTIONS ####

    def _get_renderer(self):
        return stereo.MirrorDisplay(self.window_size, self.fov, 1, 1024, self.screen_dist, self.iod)

    #### STATE FUNCTIONS ####
    def _while_wait(self):
        self.target1.translate(0, 0, 0, reset=True)
        self.target1.attach()
        self.requeue()
        self.draw_world()

