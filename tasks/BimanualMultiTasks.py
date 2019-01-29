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
import os

import math
import traceback

####### CONSTANTS
sec_per_min = 60.0
RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)
GOLD = (1., 0.843, 0., 0.5)
PURPLE = (.5,0,.5,0.5)
YELLOW = (1,0.65,0,1)
mm_per_cm = 1./10

from target_graphics import *


class ManualControlMulti(Sequence, Window):
    '''
    This is an improved version of the original manual control tasks that includes the functionality
    of ManualControl, ManualControl2, and TargetCapture all in a single task. This task doesn't
    assume anything about the trial structure of the task and allows a trial to consist of a sequence
    of any number of sequential targets that must be captured before the reward is triggered. The number
    of targets per trial is determined by the structure of the target sequence used.
    '''

    background = (0,0,0,1)
    # cursor_color = YELLOW
    # _target_color = YELLOW

    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())

    starting_pos = (5, 0, 5)

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition", stop=None),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", stop=None),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", stop=None),
        hold_penalty = dict(hold_penalty_end="targ_transition", stop=None),
        reward = dict(reward_end="wait")
    )
    trial_end_states = ['reward', 'timeout_penalty', 'hold_penalty']

    #initial state
    state = "wait"

    # target_color = YELLOW
    target_index = -1 # Helper variable to keep track of which target to display within a trial
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.
    
    cursor_visible = False # Determines when to hide the cursor.
    no_data_count = 0 # Counter for number of missing data frames in a row
    scale_factor = 2 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)

    limit2d = traits.Int(1, desc='Specifies whether the task will be constrained to 2 dimensions or 3')
    righthand = traits.Int(1, desc='Specifies whether the hand controlling the plant is the right hand or left')

    sequence_generators = ['centerout_2D_discrete', 'point_to_point_3D', 'centerout_3D', 'centerout_3D_cube', 'centerout_2D_discrete_upper','centerout_2D_discrete_rot', 'centerout_2D_discrete_multiring',
        'centerout_2D_discrete_randorder', 'centeroutback_2D', 'centeroutback_2D_farcatch', 'centeroutback_2D_farcatch_discrete',
        'outcenterout_2D_discrete', 'outcenter_2D_discrete', 'rand_target_sequence_3d', 'rand_target_sequence_2d', 'rand_target_sequence_2d_centerout',
        'rand_target_sequence_2d_partial_centerout', 'rand_multi_sequence_2d_centerout2step', 'rand_pt_to_pt',
        'centerout_2D_discrete_far', 'centeroutback_2D_v2','centerout_2D_discrete_eyetracker_calibration','centerout_2D_bimanual_congruent', 'centerout_3D_2t_cong', 'inout_plane_3D', 'rect_plane_3D']
    is_bmi_seed = True
    


    # Runtime settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")
    target_radius = traits.Float(4, desc="Radius of targets in cm")
    
    hold_time = traits.Float(.1, desc="Length of hold required at targets")
    instr_delay = traits.Float(.1, desc="Length of instructed delay period (and center hold)")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(20, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')
    # session_length = traits.Float(0, desc="Time until task automatically stops. Length of 0 means no auto stop.")
    marker_num = traits.Int(13, desc="The index of the motiontracker marker to use for cursor position")
    # NOTE!!! The marker on the hand was changed from #0 to #14 on
    # 5/19/13 after LED #0 broke. All data files saved before this date
    # have LED #0 controlling the cursor.
    # NOTE!!! Changed again from #14 to #13 by Tanner after initial recalibration for bimanual capture space on 6/21/16

    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')    
    plant_type_options = plantlist.keys()
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=plantlist.keys())
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc="Radius of cursor")

    

    
    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)

        if self.righthand==1:
            self.cursor_color = PURPLE
            self._target_color = PURPLE
            self.target_color = PURPLE
            self.targ_shift = np.array([8,10,0])
        else:
            self.cursor_color = YELLOW
            self._target_color = YELLOW
            self.target_color = YELLOW
            self.targ_shift = np.array([-8,10,0])

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
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)

            self.targets = [target1, target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)

        
        # Initialize target location variable
        self.target_location = self.targ_shift

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

        self.lasttwo = np.zeros([2,3])
        self.isstart=1

        # Instantiate 3D cue (cube) if reaching in 3D
        if self.limit2d==0:
                self.threed_cue = VirtualRectangularTarget(target_width=10, target_color=[1,1,1,0.25])
                for model in self.threed_cue.graphics_models:
                    self.add_model(model)
                self.threed_cue.move_to_position(np.array([0, -10, 0])+self.targ_shift)




        # self.threed_cue = plantlist[self.threed_cue_type]
        # if hasattr(self.threed_cue, 'graphics_models'):
        #     for model in self.threed_cue.graphics_models:
        #         self.add_model(model)


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

        self.move_arm_motiontracker()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(ManualControlMulti, self)._cycle()

        # Blink 3d cue to maintain rendering order
        if self.limit2d==0:
            self.threed_cue.hide()
            self.threed_cue.show()
        
    def move_arm_motiontracker(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        #get data from motion tracker- take average of all data points since last poll
        pt = self.motiondata.get()
        # reorder = pt
        # reorder[0]=1-pt[0]
        # reorder[1]=-pt[2]
        # reorder[2]=pt[1]
        # pt = reorder

        if len(pt) > 0:
            pt = pt[:, self.marker_num, :]
            conds = pt[:, 3]
            inds = np.nonzero((conds>=0) & (conds!=4))[0]
            if len(inds) > 0:
                pt = pt[inds,:3]
                # Flip L/R axis
                pt[0,0]=-200-pt[0,0]
                # Reorder y and z axes
                # temp = np.copy(pt[1])
                # pt[1]=pt[2]
                pt[0,2]=-900+1.5*pt[0,1]
                #scale actual movement to desired amount of screen movement
                pt = pt.mean(0) * self.scale_factor
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: pt[1] = 0
                #pt[1] = pt[1]*2
                # Return cursor location
                self.no_data_count = 0
                pt = pt * mm_per_cm #self.convert_to_cm(pt)
                #print pt
            else: #if no usable data
                self.no_data_count += 1
                pt = None
        else: #if no new data
            self.no_data_count +=1
            pt = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available
        
        if pt is not None:
            print np.linalg.norm(self.lasttwo[0,:]-pt)
            if np.linalg.norm(self.lasttwo[0,:]-pt)<10 or self.isstart==1:
                self.lasttwo = np.roll(self.lasttwo,1,axis=0)
                self.lasttwo[0,:]=pt
                self.plant.set_endpoint_pos(np.mean(self.lasttwo,axis=0))
                self.isstart=0

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
        if self.no_data_count < 6:
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

        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial

    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index % 2]
        self.target_location = self.targs[self.target_index] + self.targ_shift
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
        # super(ManualControlMulti, self)._start_reward()
        self.targets[self.target_index % 2].show()

    #### Generator functions ####
    @staticmethod
    def rect_plane_3D(length=1000, boundaries=(-18,18,-10,10,-15,15),distance=10,width=30,height=20,hand=0):
        # Choose a random sequence of targets from a set of 6 discrete locations on a
        # vertical plane located *distance* away from the starting location. The 6
        # target locations form a rectangle of width *width* and height *height* 
        # in the target plane. If *hand* is 0, produce targets for both hands. If 1,
        # use only the left hand, and if 2, use only the right hand.

        z_options = np.array([-height/2,height/2])

        if hand==0:
            x_options = np.array([-width/2,0,width/2])
        elif hand==1:
            x_options = np.array([-width/2,-0.00001])   # make the center targets very slightly negative so
                                                        # that the task knows they belong to the left hand,
                                                        # but the subject still perceives the central
                                                        # targets as dead-center
        else:
            x_options = np.array([0.00001,width/2])

        x = np.random.choice(x_options,length)
        y = distance*np.ones(length)
        z = np.random.choice(z_options,length)

        pairs = np.zeros([length,2,3])
        pairs[:,1,0] = x
        pairs[:,1,1] = y
        pairs[:,1,2] = z

        return pairs

    @staticmethod
    def inout_plane_3D(length=1000, boundaries=(-18,18,-10,10,-15,15),distance=10,width=15,height=7.5,hemispheres=0):
        # Choose a random sequence of targets from a set of 8 discrete locations on a
        # vertical plane located *distance* away from the starting location. The 8
        # target locations form an oval of width-radius *width* and height-radius 
        # *height* in the target plane. If *hemispheres* is 0, use all targets. If 1,
        # use only the left targets, and if 2, use only the right targets.
        if hemispheres==0:
            theta = (2*np.random.randint(6,size=length)+1)*np.pi/8 - np.pi/4
        elif hemispheres==1:
            theta = (2*np.random.randint(3,size=length)+1)*np.pi/8 + 2*np.pi/4
        else:
            theta = (2*np.random.randint(3,size=length)+1)*np.pi/8 - np.pi/4

        x = width*np.cos(theta)
        y = height*np.sin(theta)
        z = distance*np.ones(length)

        pairs = np.zeros([length,2,3])
        pairs[:,1,0] = x
        pairs[:,1,1] = z
        pairs[:,1,2] = y

        return pairs

    @staticmethod
    def point_to_point_3D(length=2000, boundaries=(-18,18,-10,10,-15,15), distance=10, chain_length=2):1

    @staticmethod
    def centerout_3D(length=1000, boundaries=(-18,18,-10,10,-15,15),distance=8):
        # Choose a random sequence of points on the surface of a hemisphere of radius
        # "distance"
        theta = np.random.uniform(-3*np.pi/8, 3*np.pi/8, length)
        phi = np.random.uniform(np.pi/8, 7*np.pi/8, length)
        x = distance*np.cos(phi)*np.sin(theta)
        y = distance*np.sin(phi)*np.sin(theta)
        z = distance*np.cos(theta)

        pairs = np.zeros([length,2,3])
        pairs[:,1,0] = x
        pairs[:,1,1] = z
        pairs[:,1,2] = y

        return pairs

    @staticmethod
    def centerout_3D_2t_cong(length=1000, boundaries=(-18,18,-10,10,-15,15),distance=8):
        # Choose a random sequence of points on the surface of a hemisphere of radius
        # "distance" and generate 2 sets of identical points for either hand
        theta = np.random.uniform(-3*np.pi/8, 3*np.pi/8, length)
        phi = np.random.uniform(np.pi/8, 7*np.pi/8, length)
        x = distance*np.cos(phi)*np.sin(theta)
        y = distance*np.sin(phi)*np.sin(theta)
        z = distance*np.cos(theta)

        pairs = np.zeros([length,2,3])
        pairs[:,1,0] = x
        pairs[:,1,1] = z
        pairs[:,1,2] = y

        pairs2t = np.concatenate((pairs,pairs),axis=2)

        return pairs2t

    @staticmethod
    def centerout_3D_2t_incong(length=1000, boundaries=(-18,18,-10,10,-15,15),distance=8, rotation=np.pi):
        # Choose a random sequence of points on the surface of a hemisphere of radius
        # "distance", generating 2 sets of targets 
        theta = np.random.rand(length)*np.pi-np.pi/2
        phi = np.random.rand(length)*np.pi
        x1 = distance*np.cos(phi)*np.sin(theta)
        x2 = distance*np.cos(phi)*np.sin(theta) 
        y = distance*np.sin(phi)*np.sin(theta)
        z = distance*np.cos(theta)

        pairs = np.zeros([length,2,3])
        pairs[:,1,0] = x
        pairs[:,1,1] = z
        pairs[:,1,2] = y

        pairs2t = np.concatenate((pairs,pairs),axis=2)


        targs = np.zeros([])



        return pairs2t

    @staticmethod
    def centerout_3D_cube(length=1000, edge_length=8, boundaries=(-18,18,-12,12)):
        '''
        Choose a random sequence of points at the vertices of a cube with width
        "edge_length"
        '''
        coord = [-float(edge_length)/2, float(edge_length)/2]
        from itertools import product
        target_locs = [(x, y, z) for x, y, z in product(coord, coord, coord)]
        
        n_corners_in_cube = 8
        pairs = np.zeros([length, 2, 3])

        for k in range(length):
            pairs[k, 0, :] = np.zeros(3)
            pairs[k, 1, :] = target_locs[np.random.randint(0, n_corners_in_cube)]

        pairs[:,:,1] += -10 

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
    def centerout_2D_discrete_eyetracker_calibration(nblocks=100, ntargets=4, boundaries=(-18,18,-12,12),
        distance=10):
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
            temp2 = np.array([temp[2],temp[0],temp[1],temp[3]]) # Left Right Up Down
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
    def centerout_2D_discrete_far(nblocks=100, ntargets=8, xmax=25, xmin=-25, zmin=-14, zmax=14, distance=10):
        target_angles = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
        target_pos = np.vstack([np.cos(target_angles), np.zeros_like(target_angles), np.sin(target_angles)]).T*distance
        target_pos = np.vstack(filter(lambda targ: targ[0] < xmax and targ[0] > xmin and targ[2] > zmin and targ[2] < zmax, target_pos))

        from riglib.experiment.generate import block_random
        periph_targets_per_trial = block_random(target_pos, nblocks=nblocks)
        target_seqs = []
        for targ in periph_targets_per_trial:
            target_seqs.append(np.vstack([np.zeros(3), targ]))
        return target_seqs

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
    def centeroutback_2D_v2(length, boundaries=(-18,18,-12,12), distance=8):
        '''
        This fn exists purely for compatibility reasons?
        '''
        return centeroutback_2D(length, boundaries=boundaries, distance=distance)

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

    @staticmethod
    def centerout_2D_bimanual_congruent(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        '''

        Generates a sequence of 2D (x and z) target pairs for each hand with the first target
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
        pairs : [nblocks*ntargets x 4 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x1 = distance*np.cos(theta)-5
        x2 = distance*np.cos(theta)+5
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 4, 3])
        pairs[:,0,0] = -5*np.ones([len(theta)])
        pairs[:,1,:] = np.vstack([x1, y, z]).T
        pairs[:,2,0] = 5*np.ones([len(theta)])
        pairs[:,3,:] = np.vstack([x2, y, z]).T
        
        return pairs

    @staticmethod
    def centerout_2D_bimanual_incongruent(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        '''

        Generates a sequence of 2D (x and z) target pairs for each hand with the first target
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
        pairs : [nblocks*ntargets x 4 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x1 = distance*np.cos(theta)-5
        x2 = distance*np.cos(theta)+5
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 4, 3])
        pairs[:,0,0] = -5*np.ones([len(theta)])
        pairs[:,1,:] = np.vstack([x1, y, z]).T
        pairs[:,2,0] = 5*np.ones([len(theta)])
        pairs[:,3,:] = np.vstack([x2, y, z]).T
        
        return pairs


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
        #self.move_joystick()
        super(JoystickMulti, self)._cycle()

    def move_arm_motiontracker(self):
        ''' Returns the 3D coordinates of the cursor. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        #get data from phidget
        pt = self.joystick.get()
        #print pt

        if len(pt) > 0:

            pt = pt[-1][0]
            x = 1-pt[1]
            y = pt[0]


            pt[1]=1-pt[1]; #Switch U / D axes ***actually L/R now since U/D and L/R swapped, see line 1527
            #pt[0]=1-pt[0]; #Switch L / R axes
            calib = [0.5,0.5] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
            # calib = [ 0.487,  0.   ]

            #if self.joystick_method==0:                
            if self.joystick_method==0:
                pos = np.array(   [ (calib[1]-pt[1]), 0, (pt[0]-calib[0])]   )
                pos[0] = pos[0]*36
                pos[2] = pos[2]*24
                self.current_pt = pos

            elif self.joystick_method==1:
                #vel=np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                vel = np.array([x-calib[0], 0., y-calib[1]])
                epsilon = 2*(10**-2) #Define epsilon to stabilize cursor movement
                if sum((vel)**2) > epsilon:
                    self.current_pt=self.last_pt+20*vel*(1/60) #60 Hz update rate, dt = 1/60
                else:
                    self.current_pt = self.last_pt

                #self.current_pt = self.current_pt + (np.array([np.random.rand()-0.5, 0., np.random.rand()-0.5])*self.joystick_speed)

                if self.current_pt[0] < -20: self.current_pt[0] = -20
                if self.current_pt[0] > 20: self.current_pt[0] = 20
                if self.current_pt[-1] < -12: self.current_pt[-1] = -12
                if self.current_pt[-1] > 12: self.current_pt[-1] = 12

            self.plant.set_endpoint_pos(self.current_pt)
            self.last_pt = self.current_pt.copy()


class JoystickSpeedFreeChoice(JoystickMulti):
    '''
    Task where the virtual plant starts in configuration sampled from a discrete set and resets every trial
    '''
    #from tasks import choice_fa_tasks
    from tasks.manualcontrolfreechoice import target_colors

    sequence_generators = ['centerout_2D_discrete_w_free_choice_manual', 'centerout_2D_discrete_w_free_choice_manual_w_instructed'] 
    #sequence_generators = ['centerout_2D_discrete'] 
    
    joystick_speed_0 = traits.Float(1., desc='speed gain of cursor for choice 1')
    color_0 = traits.OptionsList(*target_colors.keys(), bmi3d_input_options=target_colors.keys())

    joystick_speed_1 = traits.Float(1., desc='speed gain of cursor for choice 2')
    color_1 = traits.OptionsList(*target_colors.keys(), bmi3d_input_options=target_colors.keys())

    choice_target_rad = traits.Float(2.)

    status = dict(
        wait = dict(start_trial="targ_transition", stop=None),
        pre_choice_orig = dict(enter_orig='choice_target', timeout='timeout_penalty'),
        choice_target = dict(enter_choice_target='targ_transition', timeout='timeout_penalty', stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", make_choice='pre_choice_orig'),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
        )

    def __init__(self, *args, **kwargs):
        super(JoystickSpeedFreeChoice, self).__init__(*args, **kwargs)
        from tasks.manualcontrolfreechoice import target_colors

        seq_params = eval(kwargs.pop('seq_params', '{}'))
        print 'SEQ PARAMS: ', seq_params, type(seq_params)
        self.choice_per_n_blocks = seq_params.pop('blocks_per_free_choice', 1)
        self.n_free_choices = seq_params.pop('n_free_choices', 2)
        self.n_targets = seq_params.pop('ntargets', 8)

        self.input_type_dict = dict()
        self.input_type_dict[0]=self.joystick_speed_0
        self.input_type_dict[0, 'color']=target_colors[self.color_0]

        self.input_type_dict[1]=self.joystick_speed_1
        self.input_type_dict[1, 'color']=target_colors[self.color_1]

        # Instantiate the choice targets
        self.choices_targ_list = []
        for c in range(self.n_free_choices):
            self.choices_targ_list.append(VirtualCircularTarget(target_radius=self.choice_target_rad, 
                target_color=self.input_type_dict[c, 'color']))

        for c in self.choices_targ_list:
            for model in c.graphics_models:
                self.add_model(model)

        self.subblock_cnt = 0
        self.subblock_end = self.choice_per_n_blocks*self.n_targets
        self.choice_made = 0
        self.choice_ts = 0
        self.chosen_input_ix = -1
        self.choice_locs = np.zeros((self.n_free_choices, 3))

    def init(self):
        self.add_dtype('trial_type', np.str_, 16)
        self.add_dtype('choice_ix', 'f8', (1, ))
        self.add_dtype('choice_targ_loc', 'f8', (self.n_free_choices, 3))

        super(JoystickSpeedFreeChoice, self).init()

    def _start_pre_choice_orig(self):
        target = self.targets[0]        
        target.move_to_position(np.array([0., 0., 0.]))
        target.cue_trial_start()
        self.chosen_input_ix = -1

    def _test_enter_orig(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos)
        return d <= self.target_radius

    def _parse_next_trial(self):
        pairs = self.next_trial[0]
        self.targs = pairs[:, :, 1]
        self.choice_locs = pairs[:, :, 0]
        self.choice_asst_ix = self.next_trial[1][0]
        self.choice_instructed = self.next_trial[2][0]

        if self.subblock_cnt >= self.subblock_end:
            self.choice_made = 0
            self.subblock_cnt = 0

    def _test_make_choice(self, ts):
        return not self.choice_made

    def move_joystick(self):
        super(JoystickSpeedFreeChoice, self).move_joystick()
        if self.choice_ts <= 60*1:
            self.plant.set_endpoint_pos(np.array([0., 0., 0.]))
            self.current_pt = np.zeros((3, ))

    def _start_choice_target(self):
        self.choice_ts = 0
        if self.choice_instructed == 'Free':
            for ic, c in enumerate(self.choices_targ_list):
                #move a target to current location (target1 and target2 alternate moving) and set location attribute
                c.move_to_position(self.choice_locs[ic, :])
                c.sphere.color = self.input_type_dict[ic, 'color']
                c.show()
        elif self.choice_instructed == 'Instructed':
            ic = self.choice_asst_ix
            c = self.choices_targ_list[ic]
            c.move_to_position(self.choice_locs[ic,:])
            c.sphere.color = self.input_type_dict[ic, 'color']
            c.show()

        target = self.targets[0]
        target.hide()
        self.choice_ts = 0

    def _while_choice_target(self):
        self.choice_ts += 1

    def _start_target(self):
        super(JoystickSpeedFreeChoice, self)._start_target()

        for ic, c in enumerate(self.choices_targ_list):
            c.hide()

    def _test_enter_choice_target(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        enter_targ = 0
        for ic, c in enumerate(self.choice_locs):
            d = np.linalg.norm(cursor_pos - c)
            if d <= self.choice_target_rad: #NOTE, gets in if CENTER of cursor is in target (not entire cursor)
                enter_targ+=1

                #Set chosen as new input: 
                self.chosen_input_ix = ic
                self.joystick_speed = self.input_type_dict[ic]
                print 'trial: ', self.choice_instructed, self.joystick_speed

                #Declare that choice has been made:
                self.choice_made = 1

                #Change color of cursor: 
                sph = self.plant.graphics_models[0]
                sph.color = self.input_type_dict[ic, 'color']

        return enter_targ > 0
    
    def _cycle(self):
        self.task_data['trial_type'] = self.choice_instructed
        self.task_data['choice_ix'] = self.chosen_input_ix
        self.task_data['choice_targ_loc'] = self.choice_locs

        super(JoystickSpeedFreeChoice, self)._cycle()

    def _test_trial_incomplete(self, ts):
        if self.choice_made == 0:
            return False
        else:
            return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts)
    
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the target radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius)
    
    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius
        return d > rad

    def _start_reward(self):
        self.subblock_cnt+=1
        super(JoystickSpeedFreeChoice, self)._start_reward()

    @staticmethod
    def centerout_2D_discrete_w_free_choice_manual_w_instructed(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10, n_free_choices=2, blocks_per_free_choice = 1, percent_instructed=50.):
        from tasks import choice_fa_tasks
        return choice_fa_tasks.FreeChoiceFA.centerout_2D_discrete_w_free_choice_v2(nblocks=nblocks, ntargets=ntargets, boundaries=boundaries,
            distance=distance, n_free_choices=n_free_choices, blocks_per_free_choice=blocks_per_free_choice,percent_instructed=percent_instructed)
    @staticmethod
    def centerout_2D_discrete_w_free_choice_manual(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10, n_free_choices=2, blocks_per_free_choice = 1):
        from tasks import choice_fa_tasks
        return choice_fa_tasks.FreeChoiceFA.centerout_2D_discrete_w_free_choice(nblocks=nblocks, ntargets=ntargets, boundaries=boundaries,
            distance=distance, n_free_choices=n_free_choices, blocks_per_free_choice=blocks_per_free_choice)


class JoystickMultiObstacles(JoystickMulti):

    sequence_generators = ['freeform_2D_discrete_w_2_obstacle_v2', 'centerout_2D_discrete_w_obstacle_v2']

    obstacle_size = traits.Float(2., desc='size of obtacles -- must match squeence generator')
    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", enter_obstacle="obstacle_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition", stop=None),
        targ_transition = dict(trial_complete="reward", trial_abort='wait', trial_incomplete="target", stop=None),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", stop=None),
        hold_penalty = dict(hold_penalty_end="targ_transition", stop=None),
        obstacle_penalty = dict(obstacle_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
        )

    def __init__(self, *args, **kwargs):
        super(JoystickMultiObstacles, self).__init__(*args, **kwargs)
        self.add_obstacles()
        self.i = 0

    def init(self, *args, **kwargs):
        self.add_dtype('obstacle_size', 'f8', (1,))
        self.add_dtype('obstacle_location', 'f8', (5,3))
        super(JoystickMultiObstacles, self).init(*args, **kwargs)
    
    def add_obstacles(self):
        import target_graphics
        #Add obstacle
        self.obstacle_list=[]
        for i in range(5):
            obstacle = target_graphics.VirtualRectangularTarget(target_width=self.obstacle_size, target_height=self.obstacle_size, 
                target_color=(0, 0, 1, .5), starting_pos=np.zeros(3))
            self.obstacle_list.append(obstacle)
            for model in obstacle.graphics_models:
                self.add_model(model)

    def _parse_next_trial(self):
        self.targs = self.next_trial[0]
        #Width and height of obstacle

        self.trial_obstacle_list = []
        self.trial_obstacle_loc = []

        obs = self.next_trial[1]
        if len(obs.shape) == 1:
            obs = np.array([obs])

        for io, o in enumerate(obs):
            self.trial_obstacle_list.append(self.obstacle_list[io])
            self.trial_obstacle_loc.append(o)
            #print 'obs number and loc: ', io, o
        for j in np.arange(io+1, 5):
            o = self.obstacle_list[j]
            o.move_to_position(np.array([-100., 0., -100.]))
            #print 'hiding: obs number and loc: ', j

    def _start_target(self):
        super(JoystickMultiObstacles, self)._start_target()
        for io, o in enumerate(self.trial_obstacle_list):
            o.move_to_position(self.trial_obstacle_loc[io])
            #o.cube.color = (0., 0., 1., .5)
            o.cube.color = (27/255., 121/255., 255/255., .5)
            o.show()

    def _test_enter_obstacle(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        in_obs = False
        for io, o in enumerate(self.trial_obstacle_loc):
            centered_cursor_pos = np.abs(cursor_pos - o)

            if np.all(centered_cursor_pos < self.obstacle_size/2.):
                print 'in'
                in_obs = True
                self.obs_entered_ix = io
        return in_obs

    def _start_obstacle_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _test_trial_complete(self, ts):
        return self.target_index==self.chain_length-1

    def _test_trial_incomplete(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts)

    def _test_trial_abort(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries==self.max_attempts)

    def add_obstacle_data(self):
        self.task_data['obstacle_size'] = self.obstacle_size
        tsk = np.zeros((5, 3))
        for io, o in enumerate(self.trial_obstacle_loc):
            tsk[io,:] = o
        self.task_data['obstacle_location'] = tsk
    def _cycle(self):
        super(JoystickMultiObstacles, self)._cycle()

    def _test_obstacle_penalty_end(self, ts):
        o = self.trial_obstacle_list[self.obs_entered_ix]
        o.cube.color = (1., 1., 0., .5)
        return ts >= self.timeout_penalty_time
    
    @staticmethod
    def freeform_2D_discrete_w_2_obstacle_v2(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):

       from bmimultitasks import BMIMultiObstacles
       return BMIMultiObstacles.freeform_2D_discrete_w_2_obstacle(nblocks=nblocks, ntargets=ntargets,
        boundaries=boundaries, distance=distance)

    @staticmethod
    def centerout_2D_discrete_w_obstacle(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        from bmimultitasks import BMIResettingObstacles
        return BMIResettingObstacles.centerout_2D_discrete_w_obstacle(nblocks=nblocks, ntargets=ntargets,
         boundaries=boundaries, distance=distance, obstacle_sizes=(2,3)) 

    @staticmethod
    def centerout_2D_discrete_w_obstacle_v2(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        from bmimultitasks import BMIResettingObstacles
        gen = BMIResettingObstacles.centerout_2D_discrete_w_obstacle(nblocks=nblocks, ntargets=ntargets,
         boundaries=boundaries, distance=distance, obstacle_sizes=(2,3)) 

        # Remove obstacle size: 
        gen_trunc = gen[:]
        for i, (t, os, ol) in enumerate(gen):
            gen_trunc[i] = (t, ol)
        return gen_trunc


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
        self.target_reward_time = self.reward_time


    def _start_move_reward(self):
        #if self.reward is not None:
        #    self.reward.reward(self.move_reward_time*1000.)
        self.reward_time = self.move_reward_time
        self._start_reward()
        self.reward_time = self.target_reward_time

    def _end_move_reward(self):
        pass#self._end_reward()
        
    def _test_move_reward_end(self,ts):        
        if self.move_reward_time>0:
           self.last_rew_pt =  self.current_pt.copy()
           if ts > self.move_reward_time: #to compensate for adding +1 to target index for moving. 
                self.target_index += -1
           return ts > self.move_reward_time
        else:
           return False

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



class BimanualMulti_Hold(ManualControlMulti):

    '''
    This is a task written for exp 1.3 of the ipsilateral control project. The task uses LED motion
    tracking and requires the subject to maintain one limb in a specified position while reaching
    with the other.
    Written by Tanner Dixon July 13, 2016
    '''

    # NOTE come back and fix it so that the hold before target appearance can't go to penalty (maybe...)

    plant_type_a = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())
    plant_type_b = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())

    status = dict(
        wait = dict(start_trial="target_a", stop=None),
        target_a = dict(enter_target_a="target_b", stop=None),
        # target_a = dict(enter_target_a="hold_a", stop=None),
        # hold_a = dict(enter_target_b="target_b", hold_a_complete="target_a",  stop=None),
        target_b = dict(first_enter_target_b="first_hold", enter_target_b="hold", timeout="timeout_penalty", leave_target_a="hold_penalty", stop=None),
        first_hold = dict(first_hold_complete="hold", timeout="timeout_penalty", leave_target_a="hold_penalty", stop=None), 
        hold = dict(leave_early_b="hold_penalty", hold_complete="targ_transition",  stop=None),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target_b", failed_attempt="target_a", stop=None),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", stop=None),
        hold_penalty = dict(hold_penalty_end="targ_transition", stop=None),
        reward = dict(reward_end="wait")
    )

    cursor_a_color = (1,0,0,0.5)
    cursor_b_color = (1,0,0,0.5)
    cursor_a_radius = traits.Float(.5, desc="Radius of the left hand cursor")
    cursor_b_radius = traits.Float(.5, desc="Radius of the right hand cursor")

    target_a_color = (1,0,0,0.5)
    target_b_color = (1,0,0,0.5)
    target_a_radius = traits.Float(4, desc="Radius of holding targets in cm")
    target_b_radius = traits.Float(2, desc="Radius of reaching targets in cm")
    hold_target_radius = traits.Float(4.5, desc="Radius of reaching targets in cm")

    left_start = traits.Float(0, desc="Distance from center to left hand starting position")
    right_start = traits.Float(0, desc="Distance from center to right hand starting position")

    plant_a_visible = traits.Bool(True, desc='Specifies whether entire plant a is displayed or just endpoint')
    plant_b_visible = traits.Bool(True, desc='Specifies whether entire plant b is displayed or just endpoint')

    

    marker_a_num = traits.Int(16, desc="The index of the motiontracker marker to use for the holding cursor position")
    marker_b_num = traits.Int(13, desc="The index of the motiontracker marker to use for the reaching cursor position")


    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_a_visible = True
        self.cursor_b_visible = True

        if self.righthand:
            self.targ_shift = np.array([5,3,0])
            self.target_a_color = YELLOW
            self.target_b_color = PURPLE
        else:
            self.targ_shift = np.array([-5,3,0])
            self.target_a_color = PURPLE
            self.target_b_color = YELLOW

        RH = 2*self.righthand-1
        if self.left_start == 0:
            self.target_a_location = np.array([-5*RH, 3, 0]) 
        else:
            self.target_a_location = np.array([-15*RH, 3, 0])


        self.alreward = 0

        # Initialize the plants
        # if not hasattr(self, 'plant'):
        self.plant_a = plantlist[self.plant_type_a]
        self.plant_b = plantlist[self.plant_type_b]
        self.plant_a_vis_prev = True
        self.plant_b_vis_prev = True

        # Add graphics models for the plants and targets to the window
        for model in self.plant_a.graphics_models:
            self.add_model(model)
            print self.plant_a.graphics_models
        for model in self.plant_b.graphics_models:
            self.add_model(model)
            print self.plant_b.graphics_models
        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            self.target_a = VirtualCircularTarget(target_radius=self.hold_target_radius, target_color=self.target_a_color)
            self.target_b1 = VirtualCircularTarget(target_radius=self.hold_target_radius, target_color=self.target_b_color)
            self.target_b2 = VirtualCircularTarget(target_radius=self.target_b_radius, target_color=self.target_b_color)

            self.b_targets = [self.target_b1, self.target_b2]
            for target in self.b_targets:
                for model in target.graphics_models:
                    self.add_model(model)

            # self.a_targets = [self.target_a1, self.target_a2]
            for model in self.target_a.graphics_models:
                self.add_model(model)
        
        # Initialize reaching target location variable
        self.target_b_location = np.array([0, 0, 0])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        # self.add_dtype('cursor_a', 'f8', (3,))
        # self.add_dtype('cursor_b', 'f8', (3,))
        for attr in self.plant_a.hdf_attrs:
            self.add_dtype(*attr)
        self.no_data_count_a = 0
        self.no_data_count_b = 0
        self.last_rew_pt_a = np.zeros([2,3])
        self.last_rew_pt_b = np.zeros([2,3])
        self.current_pt_a = np.zeros([3])
        self.current_pt_b = np.zeros([3])

        self.recent_a=np.zeros([30,3])
        self.recent_b=np.zeros([30,3])
        self.lastreal_a=0
        self.isstart_a=1
        self.lastreal_b=0
        self.isstart_b=1

        self.rew = np.copy(self.reward_time)


    def init(self):
        self.add_dtype('target_a', 'f8', (3,))
        self.add_dtype('target_b', 'f8', (3,))
        self.add_dtype('target_index', 'i', (1,))
        super(ManualControlMulti, self).init()


    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['target_a'] = self.target_a_location.copy()
        self.task_data['target_b'] = self.target_b_location.copy()
        self.task_data['target_index'] = self.target_index

        ## Run graphics commands to show/hide the plant if the visibility has changed
        if self.plant_type != 'CursorPlant':
            if self.plant_a_visible != self.plant_a_vis_prev:
                self.plant_a_vis_prev = self.plant_a_visible
                self.plant_a.set_visibility(self.plant_a_visible)
                # self.show_object(self.plant, show=self.plant_visible)
        if self.plant_type != 'CursorPlant':
            if self.plant_b_visible != self.plant_b_vis_prev:
                self.plant_b_vis_prev = self.plant_b_visible
                self.plant_b.set_visibility(self.plant_b_visible)
                # self.show_object(self.plant, show=self.plant_visible)

        self.move_arm_motiontracker()


        ## Save plant status to HDF file
        plant_a_data = self.plant_a.get_data_to_save()
        plant_b_data = self.plant_b.get_data_to_save()
        for key in plant_b_data:
            self.task_data[key] = plant_b_data[key]
        for key in plant_a_data:
            self.task_data[key] = plant_b_data[key]

        super(ManualControlMulti, self)._cycle()

        # if np.linalg.norm(self.current_pt_a - self.last_rew_pt_a) > 10:
        #     self.reward_time = self.reward_time/2
        #     self._start_reward()
        #     self.reward_time = self.reward_time*2
        #     self.last_rew_pt_a = self.current_pt_a
        # if np.linalg.norm(self.current_pt_b - self.last_rew_pt_b) > 10:
        #     self.reward_time = self.reward_time/2
        #     self._start_reward()
        #     self.reward_time = self.reward_time*2
        #     self.last_rew_pt_b=self.current_pt_b

    
        # print self.current_pt_a
        # print self.current_pt_b


        
    
    def move_arm_motiontracker(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        # Get all data from motion tracker since last update (~4 samples = 240Hz motiontracker / 60Hz task)
        pt = self.motiondata.get()

        if len(pt) > 0:
            # Select given marker number and identify indices of reliable samples
            pt_a = pt[:, self.marker_a_num, :]
            pt_b = pt[:, self.marker_b_num, :]
            conds_a = pt_a[:, 3]
            conds_b = pt_b[:, 3]
            inds_a = np.nonzero(conds_a>=0)[0]
            inds_b = np.nonzero(conds_b>=0)[0]
            if len(inds_a) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt_a = pt_a[inds_a,:3]             
                pt_a = pt_a.mean(0) #* self.scale_factor     
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt_a = pt_a * mm_per_cm 
                reorder_a = np.copy(pt_a)
                pt_a[0]= -5-reorder_a[0]
                pt_a[1]= 18+reorder_a[2]
                pt_a[2]= -2+reorder_a[1]
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt_a[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count_a = 0
            else: #if no usable data
                self.no_data_count_a+=1
                pt_a = None

            if len(inds_b) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt_b = pt_b[inds_b,:3]             
                pt_b = pt_b.mean(0) #* self.scale_factor     
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt_b = pt_b * mm_per_cm 
                reorder_b = np.copy(pt_b)
                pt_b[0]= -5-reorder_b[0]
                pt_b[1]= 18+reorder_b[2]
                pt_b[2]= -2+reorder_b[1]
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt_b[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count_b = 0
            else: #if no usable data
                self.no_data_count_b+=1
                pt_b = None

        else: #if no new data
            self.no_data_count_a+=1
            self.no_data_count_b+=1
            pt_a = None
            pt_b = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available

        # Keep 500ms (30 task samples) running log of values returned by motiontracker so that empty spots can be interpolated
        self.recent_a = np.roll(self.recent_a,1,axis=0)
        self.recent_b = np.roll(self.recent_b,1,axis=0)
        self.recent_a[0,:]=pt_a
        self.recent_b[0,:]=pt_b

        # If any block of motiontracker data in the recent history is bookended by valid samples, linearly interpolate between
        # This is largely defunct when not having a delay to gather more data
        if pt_a is not None:
            if self.lastreal_a > 0 and self.lastreal_a < 30:
                self.recent_a[0:self.lastreal_a+1,0]=np.linspace(self.recent_a[0,0],self.recent_a[self.lastreal_a,0],self.lastreal_a+1)
                self.recent_a[0:self.lastreal_a+1,1]=np.linspace(self.recent_a[0,1],self.recent_a[self.lastreal_a,1],self.lastreal_a+1)
                self.recent_a[0:self.lastreal_a+1,2]=np.linspace(self.recent_a[0,2],self.recent_a[self.lastreal_a,2],self.lastreal_a+1)
            self.lastreal_a = 0
        else:
            if self.lastreal_a < 6:
                self.recent_a[0,:]=self.recent_a[1,:]+0.8*(self.recent_a[1,:]-self.recent_a[2,:])
            self.lastreal_a += 1
        if pt_b is not None:
            if self.lastreal_b > 0 and self.lastreal_b < 30:
                self.recent_b[0:self.lastreal_b+1,0]=np.linspace(self.recent_b[0,0],self.recent_b[self.lastreal_b,0],self.lastreal_b+1)
                self.recent_b[0:self.lastreal_b+1,1]=np.linspace(self.recent_b[0,1],self.recent_b[self.lastreal_b,1],self.lastreal_b+1)
                self.recent_b[0:self.lastreal_b+1,2]=np.linspace(self.recent_b[0,2],self.recent_b[self.lastreal_b,2],self.lastreal_b+1)
            self.lastreal_b = 0
        else:
            if self.lastreal_b < 6:
                self.recent_b[0,:]=self.recent_b[1,:]+0.8*(self.recent_b[1,:]-self.recent_b[2,:])
            self.lastreal_b += 1


        # Delay feedback 100ms (6 samples) to allow for interpolation to affect feedback - this will prevent the cursor from
        # disappearing as long as there is at least one valid sample in both the ranges 0:100ms and 100:500ms
        self.plant_a.set_endpoint_pos(self.recent_a[0,:])
        self.plant_b.set_endpoint_pos(self.recent_b[0,:])
        self.current_pt_a = np.copy(self.recent_a[0,:])
        self.current_pt_b = np.copy(self.recent_b[0,:])


    def run(self):
        '''
        See experiment.Experiment.run for documentation. 
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.plant_a.start()
        self.plant_b.start()
        try:
            super(ManualControlMulti, self).run()
        finally:
            self.plant_a.stop()
            self.plant_b.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_cursor_visibility(self):
        ''' Update cursor visible flag to hide cursor if there has been no good data for more than 3 frames in a row'''
        prev_a = self.cursor_a_visible
        if self.no_data_count_a < 3:
            self.cursor_a_visible = True
            if prev != self.cursor_a_visible:
                self.show_object(self.cursor_a, show=True)
        else:
            self.cursor_a_visible = False
            if prev != self.cursor_a_visible:
                self.show_object(self.cursor_a, show=False)

        prev_b = self.cursor_b_visible
        if self.no_data_count_b < 3:
            self.cursor_b_visible = True
            if prev != self.cursor_b_visible:
                self.show_object(self.cursor_b, show=True)
        else:
            self.cursor_b_visible = False
            if prev != self.cursor_b_visible:
                self.show_object(self.cursor_b, show=False)


    #### TEST FUNCTIONS ####
    def _test_enter_target_a(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        if self.righthand:
            cursor_a_pos = self.plant_a.get_endpoint_pos()
        else:
            cursor_a_pos = self.plant_b.get_endpoint_pos()
        d = np.linalg.norm(cursor_a_pos - self.target_a_location)
        t1= d <= (self.hold_target_radius - self.cursor_a_radius) - 0.5

        if self.righthand:
            cursor_b_pos = self.plant_b.get_endpoint_pos()
        else:
            cursor_b_pos = self.plant_a.get_endpoint_pos()
        d = np.linalg.norm(cursor_b_pos - self.target_b_location)
        t2= d <= (self.hold_target_radius - self.cursor_b_radius) - 0.5

        return t1 & t2
        #return t1

    def _test_leave_target_a(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        if self.righthand:
            cursor_a_pos = self.plant_a.get_endpoint_pos()
        else:
            cursor_a_pos = self.plant_b.get_endpoint_pos()
        d = np.linalg.norm(cursor_a_pos - self.target_a_location)
        return d > (self.hold_target_radius - self.cursor_a_radius) + 0.5

    def _test_first_enter_target_b(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        if self.righthand:
            cursor_b_pos = self.plant_b.get_endpoint_pos()
        else:
            cursor_b_pos = self.plant_a.get_endpoint_pos()
        d = np.linalg.norm(cursor_b_pos - self.target_b_location)
        if self.target_index==0:
            return d <= (self.hold_target_radius - self.cursor_b_radius) - 0.5
        else:
            return False

    def _test_enter_target_b(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        if self.righthand:
            cursor_b_pos = self.plant_b.get_endpoint_pos()
        else:
            cursor_b_pos = self.plant_a.get_endpoint_pos()
        d = np.linalg.norm(cursor_b_pos - self.target_b_location)
        if self.target_index==1:
            return d <= (self.target_b_radius - self.cursor_b_radius) - 0.5
        else:
            return False
        
    def _test_leave_early_b(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        if self.target_index == 0:
            targ_rad = self.hold_target_radius
        else:
            targ_rad = self.target_b_radius

        if self.righthand:
            cursor_a_pos = self.plant_a.get_endpoint_pos()
        else:
            cursor_a_pos = self.plant_b.get_endpoint_pos()
        d = np.linalg.norm(cursor_a_pos - self.target_a_location)
        t1= d > (targ_rad - self.cursor_a_radius) + 0.5

        if self.righthand:
            cursor_b_pos = self.plant_b.get_endpoint_pos()
        else:
            cursor_b_pos = self.plant_a.get_endpoint_pos()
        d = np.linalg.norm(cursor_b_pos - self.target_b_location)
        t2= d > (targ_rad - self.cursor_b_radius) + 0.5

        return t1 or t2

    def _test_trial_incomplete(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts) and (not self._test_failed_attempt(ts)) 

    def _test_trial_complete(self, ts):
        return self.target_index==self.chain_length-1

    def _test_hold_a_complete(self, ts):
        return ts>=2*self.hold_time

    def _test_hold_complete(self, ts):
        if self.target_index ==1:
            return ts>= (np.random.randn(1)/10 + self.hold_time)     # instructed delay length sampled from gaussian with sd=200ms
        else:
            return ts>=self.instr_delay

    def _test_first_hold_complete(self, ts):
        return ts>=0.3

    def _test_failed_attempt(self, ts):
        return self.target_index==-1


    #### STATE FUNCTIONS ####


    def _start_target_a(self):
        # self.target_b1.hide()
        # self.target_b2.hide()
        # for target in self.targets:
        #     target.hide()
        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        # target = self.target_a #self.target_index % 2]
        self.target_a.move_to_position(self.target_a_location)
        self.target_a.show()
        target = self.b_targets[0]
        self.target_b_location = self.targs[0]+self.targ_shift
        target.move_to_position(self.target_b_location)
        target.cue_trial_start()
        self.target_a.change_color(self.target_a_color)
        target.change_color(self.target_b_color)



    def _start_target_b(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.b_targets[self.target_index % 2]
        if self.target_index+1 < self.chain_length:
            self.target_b_location = self.targs[self.target_index]+self.targ_shift
        else:
            self.target_b_location = self.targs[self.target_index]+np.array([0,14,-1])
        target.move_to_position(self.target_b_location)
        target.cue_trial_start()
        target.change_color(self.target_b_color)
        # if self.target_index==0:
        #     self.reward_time = 1
        #     self._start_reward()
        #     self.reward_time = float(self.rew)



    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.b_targets[idx % 2]
            new_loc = self.targs[idx] + np.array([0,14,-1])
            target.move_to_position(new_loc)
            for target in self.b_targets:
                target.show()
                target.change_color(self.target_b_color)
        self.alreward = 0


    
    def _end_hold(self):
        # change current target color to green
        self.b_targets[self.target_index % 2].cue_trial_end_success()
    
    def _start_timeout_penalty(self):
        #hide targets
        for target in self.b_targets:
            target.hide()
        self.target_a.hide()

        self.tries += 1
        self.target_index = -1

    def _start_targ_transition(self):
        #hide targets
        # self.target_a.hide()
        if self.target_index==0:
            self.reward_time = 0.0 # just to cause the beep for go cue
            self._start_reward()
            self.reward_time = float(self.rew) # return to proper reward time
        for target in self.b_targets:
            target.hide()
        # self.b_targets[0].change_color(self.target_b_color)
        # self.b_targets[0].show()
        # print self.current_pt_a
        # print self.current_pt_b
        # print self.targs[1]


    def _start_reward(self):
        # super(ManualControlMulti, self)._start_reward()
        self.b_targets[self.target_index % 2].show()

    def _start_hold_penalty(self):
        #hide targets
        for target in self.b_targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.b_targets:
            target.hide()

        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial

        # Setup starting positions
        if self.targs[1,0]==0:
            self.righthand = np.random.randn()>=0
        else:
            self.righthand = int(self.targs[1,0]>0)

        if self.righthand:
            self.targ_shift = np.array([self.right_start,15,-5.5])
            self.target_a_color = YELLOW
            self.target_b_color = PURPLE
            self.target_a_location = np.array([-self.left_start, 15, -5.5])

        else:
            self.targ_shift = np.array([-self.left_start,15,-5.5])
            self.target_a_color = PURPLE
            self.target_b_color = YELLOW
            self.target_a_location = np.array([self.right_start, 15, -5.5])

        # print self.current_pt_a
        # print self.current_pt_b





class BimanualMulti_Standard(BimanualMulti_Hold):

    '''
    This is a task written for exp 2.2 of the ipsilateral control project. The task uses LED motion
    tracking and requires the subject to simultaneously reach for one target with one hand, and another
    target with the other hand.
    Written by Tanner Dixon August 23, 2016
    '''

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(first_enter="between_enters", timeout="timeout_penalty", stop=None),
        between_enters = dict(both_enter="hold", not_together="timeout_penalty", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition", stop=None),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", stop=None),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", stop=None),
        hold_penalty = dict(hold_penalty_end="targ_transition", stop=None),
        reward = dict(reward_end="wait")
    )


    allowed_time_dif = traits.Float(.5, desc="Maximum amount of time allowed between one cursor entering its target and the other")


    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_a_visible = True
        self.cursor_b_visible = True


        self.alreward = 0

        # Initialize the plants, LEFT HAND=a=YELLOW, RIGHT HAND=b=PURPLE
        # if not hasattr(self, 'plant'):
        self.plant_a = plantlist[self.plant_type_a]
        self.plant_b = plantlist[self.plant_type_b]
        self.plant_a_vis_prev = True
        self.plant_b_vis_prev = True

        # Add graphics models for the plants and targets to the window
        for model in self.plant_a.graphics_models:
            self.add_model(model)
            print self.plant_a.graphics_models
        for model in self.plant_b.graphics_models:
            self.add_model(model)
            print self.plant_b.graphics_models
        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            self.target_a1 = VirtualCircularTarget(target_radius=self.target_a_radius, target_color=YELLOW)
            self.target_a2 = VirtualCircularTarget(target_radius=self.target_a_radius, target_color=YELLOW)
            self.target_b1 = VirtualCircularTarget(target_radius=self.target_b_radius, target_color=PURPLE)
            self.target_b2 = VirtualCircularTarget(target_radius=self.target_b_radius, target_color=PURPLE)

            self.targets_a = [self.target_a1, self.target_a2]
            self.targets_b = [self.target_b1, self.target_b2]
            for target in self.targets_a:
                for model in target.graphics_models:
                    self.add_model(model)
            for target in self.targets_b:
                for model in target.graphics_models:
                    self.add_model(model)

        
        # Initialize target location variables
        self.targ_a_shift = np.array([-10, 3, -1])
        self.targ_b_shift = np.array([10, 3, -1])
        self.target_a_location = np.array([-10, 3, -1])
        self.target_b_location = np.array([10, 3, -1])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        # self.add_dtype('cursor_a', 'f8', (3,))
        # self.add_dtype('cursor_b', 'f8', (3,))
        for attr in self.plant_a.hdf_attrs:
            self.add_dtype(*attr)
        self.no_data_count_a = 0
        self.no_data_count_b = 0
        self.last_rew_pt = np.zeros([2,3])
        self.current_pt_a = np.zeros([3])
        self.current_pt_b = np.zeros([3])

        self.recent_a=np.zeros([30,3])
        self.recent_b=np.zeros([30,3])
        self.lastreal_a=0
        self.isstart_a=1
        self.lastreal_b=0
        self.isstart_b=1
        


    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        ###########################################################MARK FOR CHANGE!!!!!
        self.task_data['target'] = self.target_b_location.copy() 
        self.task_data['target_index'] = self.target_index

        ## Run graphics commands to show/hide the plant if the visibility has changed
        if self.plant_type != 'CursorPlant':
            if self.plant_a_visible != self.plant_a_vis_prev:
                self.plant_a_vis_prev = self.plant_a_visible
                self.plant_a.set_visibility(self.plant_a_visible)
                # self.show_object(self.plant, show=self.plant_visible)
        if self.plant_type != 'CursorPlant':
            if self.plant_b_visible != self.plant_b_vis_prev:
                self.plant_b_vis_prev = self.plant_b_visible
                self.plant_b.set_visibility(self.plant_b_visible)
                # self.show_object(self.plant, show=self.plant_visible)

        self.move_arm_motiontracker()


        ## Save plant status to HDF file
        plant_a_data = self.plant_a.get_data_to_save()
        plant_b_data = self.plant_b.get_data_to_save()
        for key in plant_b_data:
            self.task_data[key] = plant_b_data[key]

        super(ManualControlMulti, self)._cycle()



    ##### HELPER AND UPDATE FUNCTIONS ####



    #### TEST FUNCTIONS ####
    def _test_first_enter(self, ts):
        '''
        return true if the distance between center of either cursor and its target is smaller than the cursor radius.
        '''
        cursor_a_pos = self.plant_a.get_endpoint_pos()
        cursor_b_pos = self.plant_b.get_endpoint_pos()
        d_a = np.linalg.norm(cursor_a_pos - self.target_a_location)
        d_b = np.linalg.norm(cursor_b_pos - self.target_b_location)
        return d_a <= (self.target_a_radius - self.cursor_a_radius) or d_b <= (self.target_b_radius - self.cursor_b_radius)

    def _test_both_enter(self, ts):
        '''
        return true if the distance between center of both cursors and their targets is smaller than the cursor radius
        '''
        cursor_a_pos = self.plant_a.get_endpoint_pos()
        cursor_b_pos = self.plant_b.get_endpoint_pos()
        d_a = np.linalg.norm(cursor_a_pos - self.target_a_location)
        d_b = np.linalg.norm(cursor_b_pos - self.target_b_location)
        return d_a <= 1*(self.target_a_radius - self.cursor_a_radius) and d_b <= 1*(self.target_b_radius - self.cursor_b_radius)
        
    def _test_leave_early(self, ts):
        '''
        return true if the distance between center of either cursors and its target is smaller than the cursor radius
        '''
        cursor_a_pos = self.plant_a.get_endpoint_pos()
        cursor_b_pos = self.plant_b.get_endpoint_pos()
        d_a = np.linalg.norm(cursor_a_pos - self.target_a_location)
        d_b = np.linalg.norm(cursor_b_pos - self.target_b_location)
        return d_a > 1*(self.target_a_radius - self.cursor_a_radius) or d_b > 1*(self.target_b_radius - self.cursor_b_radius)

    def _test_not_together(self, ts):
        '''
        return true if one cursor enters its target prior to the other cursor by more than the allotted time.
        '''
        if self.target_index==1:
            return ts>self.allowed_time_dif     
        else:
            return 0  

    def _test_hold_complete(self, ts):
        if self.target_index ==1:
            return ts>=self.hold_time
        else:
            return ts>=self.instr_delay

    def _test_trial_complete(self, ts):
        return self.target_index==self.chain_length-1


    #### STATE FUNCTIONS ####

    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets_a:
            target.hide()
        for target in self.targets_b:
            target.hide()

        self.chain_length = self.targs.shape[0]/2 #Number of sequential targets in a single trial

    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target_a = self.targets_a[self.target_index % 2]
        target_b = self.targets_b[self.target_index % 2]
        self.target_a_location = self.targs[self.target_index] #+ self.targ_a_shift
        self.target_b_location = self.targs[self.target_index] #+ self.targ_b_shift
        self.target_a_location = self.target_a_location[0:3] + self.targ_a_shift
        self.target_b_location = self.target_b_location[3:6] + self.targ_b_shift

        target_a.move_to_position(self.target_a_location)
        target_b.move_to_position(self.target_b_location)
        target_a.cue_trial_start()
        target_b.cue_trial_start()
        # print self.targs

    def _start_between_enters(self):
        pass

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target_a = self.targets_a[idx % 2]
            target_b = self.targets_b[idx % 2]
            new_loc = self.targs[idx]
            target_a.move_to_position(new_loc[0:3] + self.targ_a_shift)
            target_b.move_to_position(new_loc[3:6] + self.targ_b_shift)
            target_a.cue_trial_start()
            target_b.cue_trial_start()
            # self.reward_time = self.reward_time/5
            # self._start_reward()
            # self.reward_time = self.reward_time*5
    
    def _end_hold(self):
        # change current target color to green
        self.targets_a[self.target_index % 2].cue_trial_end_success()
        self.targets_b[self.target_index % 2].cue_trial_end_success()

    def _start_hold_penalty(self):
        #hide targets
        for target in self.targets_a:
            target.hide()
        for target in self.targets_b:
            target.hide()

        self.tries += 1
        self.target_index = -1
    
    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets_a:
            target.hide()
        for target in self.targets_b:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_targ_transition(self):
        #hide targets
        for target in self.targets_a:
            target.hide()
        for target in self.targets_b:
            target.hide()

    def _start_reward(self):
        # super(ManualControlMulti, self)._start_reward()
        self.targets_a[self.target_index % 2].show()
        self.targets_b[self.target_index % 2].show()

    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets_b:
            target.hide()
        for target in self.targets_a:
            target.hide()

        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial


class ManualControlMulti_plusMove(ManualControlMulti):

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
    move_dist_z = traits.Float(5,desc="Minimum Distance to move on screen for Reward")

    def __init__(self, *args, **kwargs):
        super(ManualControlMulti_plusMove, self).__init__(*args, **kwargs)
        self.last_rew_pt = np.zeros([2,3])
        self.current_pt = np.zeros([3])
        self.target_reward_time = self.reward_time

        self.recent=np.zeros([30,3])
        self.lastreal=0
        self.isstart=1
        self.no_data_count = 0
        self.d_x=0
        self.d_y=0

        self.consec_move_rews = 0


    def move_arm_motiontracker(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        # Get all data from motion tracker since last update (~4 samples = 240Hz motiontracker / 60Hz task)
        pt = self.motiondata.get()

        if len(pt) > 0:
            # Select given marker number and identify indices of reliable samples
            pt = pt[:, self.marker_num, :]
            conds = pt[:, 3]
            inds = np.nonzero(conds>=0)[0]
            if len(inds) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt = pt[inds,:3]             
                pt = pt.mean(0) #* self.scale_factor     
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt = pt * mm_per_cm 
                reorder = np.copy(pt)
                pt[0]=reorder[0]
                # if self.righthand==0:
                #     pt[0]=pt[0]   # for left hand use
                pt[1]= -15-reorder[2]
                if self.righthand==0:
                    pt[1]=pt[1]   # for left hand use
                pt[2]=-28+reorder[1] 




                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count = 0
            else: #if no usable data
                self.no_data_count+=1
                pt = None
        else: #if no new data
            self.no_data_count+=1
            pt = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available

        # Keep 500ms (30 task samples) running log of values returned by motiontracker so that empty spots can be interpolated
        self.recent = np.roll(self.recent,1,axis=0)
        self.recent[0,:]=pt


        # If any block of motiontracker data in the recent history is bookended by valid samples, linearly interpolate between
        if pt is not None:

            if self.lastreal > 0 and self.lastreal < 30:
                self.recent[0:self.lastreal+1,0]=np.linspace(self.recent[0,0],self.recent[self.lastreal,0],self.lastreal+1)
                self.recent[0:self.lastreal+1,1]=np.linspace(self.recent[0,1],self.recent[self.lastreal,1],self.lastreal+1)
                self.recent[0:self.lastreal+1,2]=np.linspace(self.recent[0,2],self.recent[self.lastreal,2],self.lastreal+1)
            self.lastreal = 0

        else:
            if self.lastreal < 3:
                self.recent[0,:]=self.recent[1,:]+0.2*(self.recent[1,:]-self.recent[2,:])
            self.lastreal += 1




        # Delay feedback 50ms (3 samples) to allow for interpolation to affect feedback - this will prevent the cursor from
        # disappearing as long as there is at least one valid sample in both the ranges 0:100ms and 100:500ms
        self.plant.set_endpoint_pos(self.recent[2,:])
        self.current_pt = np.copy(self.recent[2,:])

        # self.plant.set_endpoint_pos(self.recent[5,:])
        # self.current_pt = np.copy(self.recent[5,:])


    def _start_move_reward(self):
        if self.move_reward_time > 0:
            if self.d_x > self.move_dist_x:
                self.reward_time = self.move_reward_time
                self._start_reward()
                self.reward_time = self.target_reward_time
            else:
                self.reward_time = self.move_reward_time
                self._start_reward()
                self.reward_time = self.target_reward_time

    def _end_move_reward(self):
        pass#self._end_reward()
        
        
    def _test_move_reward_end(self,ts):        
        if self.move_reward_time>0:
            if math.isnan(self.current_pt[0]) == False:
                self.last_rew_pt = np.roll(self.last_rew_pt,1,0)
                self.last_rew_pt[0,:] =  self.current_pt.copy()
            else:
                pass
            if ts > self.move_reward_time: #to compensate for adding +1 to target index for moving. 
                self.target_index += -1
            return ts > self.move_reward_time
        else:
            return False

    def _test_move(self,ts):
        if self.move_reward_time>0: 
            try:
                self.d_x =  ( (self.current_pt[0] - self.last_rew_pt[0,0]) **2)  **.5
                self.d_y =  ( (self.current_pt[2] - self.last_rew_pt[0,2]) **2)  **.5
                self.d_z =  ( (self.current_pt[1] - self.last_rew_pt[0,1]) **2)  **.5
                distance_from_last = np.linalg.norm(self.current_pt-self.last_rew_pt[1,:])

                if (self.d_x > self.move_dist_x or self.d_y > self.move_dist_y or self.d_z > self.move_dist_z) and self.consec_move_rews<2:
                    r=1
                    self.consec_move_rews+=1
                    #print 'Move Reward'
                else:
                    r=0
                return r
            except:
                print self.current_pt[0]
                print self.last_rew_pt[0]

        else: 
            return False

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx % 2]
            target.move_to_position(self.targs[idx])
        # self.reward_time = self.reward_time/2
        # self._start_reward()
        # self.reward_time = self.reward_time*2
        self.consec_move_rews = 0

        

    def _end_hold(self):
        # change current target color to green
        self.targets[self.target_index % 2].cue_trial_end_success()

    def _test_hold_complete(self, ts):
        return ts>=self.hold_time

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super(ManualControlMulti, self).update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120), decimals=2)




class ManualControlMulti_alt(ManualControlMulti):

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
    plant_type_a = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())
    plant_type_b = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())
    move_reward_time = traits.Float(1,desc="Reward Time for moving Joystick")
    #move_dist = traits.Float(5,desc="Minimum Distance to move on screen for Reward")
    move_dist_x = traits.Float(5,desc="Minimum Distance to move on screen for Reward")
    move_dist_y = traits.Float(5,desc="Minimum Distance to move on screen for Reward")
    move_dist_z = traits.Float(5,desc="Minimum Distance to move on screen for Reward")

    cursor_a_color = YELLOW
    cursor_b_color = PURPLE
    cursor_a_radius = traits.Float(.5, desc="Radius of the left hand cursor")
    cursor_b_radius = traits.Float(.5, desc="Radius of the right hand cursor")

    target_a_color = YELLOW
    target_b_color = PURPLE
    target_color = PURPLE
    # target_a_width = traits.Float(4, desc="Width of holding targets in cm")
    # target_b_radius = traits.Float(2, desc="Radius of reaching targets in cm")


    plant_a_visible = traits.Bool(True, desc='Specifies whether entire plant a is displayed or just endpoint')
    plant_b_visible = traits.Bool(True, desc='Specifies whether entire plant b is displayed or just endpoint')

    

    marker_a_num = traits.Int(13, desc="The index of the motiontracker marker to use for the holding cursor position")
    marker_b_num = traits.Int(1, desc="The index of the motiontracker marker to use for the reaching cursor position")


    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_a_visible = True
        self.cursor_b_visible = True

        # Initialize the plants
        # if not hasattr(self, 'plant'):
        self.plant_a = plantlist[self.plant_type_a]
        self.plant_b = plantlist[self.plant_type_b]
        self.plant_a_vis_prev = True
        self.plant_b_vis_prev = True

        # Add graphics models for the plants and targets to the window
        for model in self.plant_a.graphics_models:
            self.add_model(model)
            print self.plant_a.graphics_models
        for model in self.plant_b.graphics_models:
            self.add_model(model)
            print self.plant_b.graphics_models

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_b_color)
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_b_color)

            self.targets = [target1, target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)

        
        # Initialize target location variable
        self.targ_shift = np.array([5,5,0])
        self.target_location = self.targ_shift

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        # self.add_dtype('cursor_a', 'f8', (3,))
        # self.add_dtype('cursor_b', 'f8', (3,))
        for attr in self.plant_a.hdf_attrs:
            self.add_dtype(*attr)
        self.no_data_count_a = 0
        self.no_data_count_b = 0
        self.last_rew_pt = np.zeros([2,3])
        self.current_pt_a = np.zeros([3])
        self.current_pt_b = np.zeros([3])

        self.recent_a=np.zeros([30,3])
        self.recent_b=np.zeros([30,3])
        self.lastreal_a=0
        self.isstart_a=1
        self.lastreal_b=0
        self.isstart_b=1

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['target'] = self.target_location.copy()
        self.task_data['target_index'] = self.target_index

        ## Run graphics commands to show/hide the plant if the visibility has changed
        if self.plant_type != 'CursorPlant':
            if self.plant_a_visible != self.plant_a_vis_prev:
                self.plant_a_vis_prev = self.plant_a_visible
                self.plant_a.set_visibility(self.plant_a_visible)
                # self.show_object(self.plant, show=self.plant_visible)
        if self.plant_type != 'CursorPlant':
            if self.plant_b_visible != self.plant_b_vis_prev:
                self.plant_b_vis_prev = self.plant_b_visible
                self.plant_b.set_visibility(self.plant_b_visible)
                # self.show_object(self.plant, show=self.plant_visible)

        self.move_arm_motiontracker()


        ## Save plant status to HDF file
        plant_a_data = self.plant_a.get_data_to_save()
        plant_b_data = self.plant_b.get_data_to_save()
        for key in plant_b_data:
            self.task_data[key] = plant_b_data[key]

        super(ManualControlMulti, self)._cycle()


    def move_arm_motiontracker(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        # Get all data from motion tracker since last update (~4 samples = 240Hz motiontracker / 60Hz task)
        pt = self.motiondata.get()

        if len(pt) > 0:
            # Select given marker number and identify indices of reliable samples
            pt_a = pt[:, self.marker_a_num, :]
            pt_b = pt[:, self.marker_b_num, :]
            conds_a = pt_a[:, 3]
            conds_b = pt_b[:, 3]
            inds_a = np.nonzero(conds_a>=0)[0]
            inds_b = np.nonzero(conds_b>=0)[0]
            if len(inds_a) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt_a = pt_a[inds_a,:3]             
                pt_a = pt_a.mean(0) #* self.scale_factor     
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt_a = pt_a * mm_per_cm 
                reorder_a = np.copy(pt_a)
                pt_a[0]= reorder_a[0]
                pt_a[1]= -2-reorder_a[2]
                pt_a[2]= -28+reorder_a[1]
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt_a[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count_a = 0
            else: #if no usable data
                self.no_data_count_a+=1
                pt_a = None

            if len(inds_b) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt_b = pt_b[inds_b,:3]             
                pt_b = pt_b.mean(0) #* self.scale_factor     
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt_b = pt_b * mm_per_cm 
                reorder_b = np.copy(pt_b)
                pt_b[0]= reorder_b[0]
                pt_b[1]= -2-reorder_b[2]
                pt_b[2]= -28+reorder_b[1]
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt_b[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count_b = 0
            else: #if no usable data
                self.no_data_count_b+=1
                pt_b = None

        else: #if no new data
            self.no_data_count_a+=1
            self.no_data_count_b+=1
            pt_a = None
            pt_b = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available

        # Keep 500ms (30 task samples) running log of values returned by motiontracker so that empty spots can be interpolated
        self.recent_a = np.roll(self.recent_a,1,axis=0)
        self.recent_b = np.roll(self.recent_b,1,axis=0)
        self.recent_a[0,:]=pt_a
        self.recent_b[0,:]=pt_b

        # If any block of motiontracker data in the recent history is bookended by valid samples, linearly interpolate between
        if pt_a is not None:
            if self.lastreal_a > 0 and self.lastreal_a < 30:
                self.recent_a[0:self.lastreal_a+1,0]=np.linspace(self.recent_a[0,0],self.recent_a[self.lastreal_a,0],self.lastreal_a+1)
                self.recent_a[0:self.lastreal_a+1,1]=np.linspace(self.recent_a[0,1],self.recent_a[self.lastreal_a,1],self.lastreal_a+1)
                self.recent_a[0:self.lastreal_a+1,2]=np.linspace(self.recent_a[0,2],self.recent_a[self.lastreal_a,2],self.lastreal_a+1)
            self.lastreal_a = 0
        else:
            if self.lastreal_a < 3:
                self.recent_a[0,:]=self.recent_a[1,:]+0.2*(self.recent_a[1,:]-self.recent_a[2,:])
            self.lastreal_a += 1
        if pt_b is not None:
            if self.lastreal_b > 0 and self.lastreal_b < 30:
                self.recent_b[0:self.lastreal_b+1,0]=np.linspace(self.recent_b[0,0],self.recent_b[self.lastreal_b,0],self.lastreal_b+1)
                self.recent_b[0:self.lastreal_b+1,1]=np.linspace(self.recent_b[0,1],self.recent_b[self.lastreal_b,1],self.lastreal_b+1)
                self.recent_b[0:self.lastreal_b+1,2]=np.linspace(self.recent_b[0,2],self.recent_b[self.lastreal_b,2],self.lastreal_b+1)
            self.lastreal_b = 0
        else:
            if self.lastreal_b < 3:
                self.recent_b[0,:]=self.recent_b[1,:]+0.2*(self.recent_b[1,:]-self.recent_b[2,:])
            self.lastreal_b += 1


        # Delay feedback 100ms (6 samples) to allow for interpolation to affect feedback - this will prevent the cursor from
        # disappearing as long as there is at least one valid sample in both the ranges 0:100ms and 100:500ms
        self.plant_a.set_endpoint_pos(self.recent_a[2,:])
        self.plant_b.set_endpoint_pos(self.recent_b[2,:])
        self.current_pt_a = np.copy(self.recent_a[2,:])
        self.current_pt_b = np.copy(self.recent_b[2,:])


    def run(self):
        '''
        See experiment.Experiment.run for documentation. 
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.plant_a.start()
        self.plant_b.start()
        try:
            super(ManualControlMulti, self).run()
        finally:
            self.plant_a.stop()
            self.plant_b.stop()


    #### TEST FUNCTIONS ####
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        if self.righthand==1:
            cursor_pos = self.plant_b.get_endpoint_pos()
        else:
            cursor_pos = self.plant_a.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius - self.cursor_radius)
        
    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        if self.righthand==1:
            cursor_pos = self.plant_b.get_endpoint_pos()
        else:
            cursor_pos = self.plant_a.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius - self.cursor_radius
        return d > rad


    def _start_move_reward(self):
        if self.move_reward_time > 0:
            if self.d_x > self.move_dist_x:
                self.reward_time = self.move_reward_time
                self._start_reward()
                self.reward_time = self.target_reward_time
            else:
                self.reward_time = self.move_reward_time
                self._start_reward()
                self.reward_time = self.target_reward_time

    #### STATE FUNCTIONS ####
    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index % 2]
        self.target_location = self.targs[self.target_index] + self.targ_shift
        target.move_to_position(self.target_location)
        target.cue_trial_start()
        target.change_color(self.target_color)

    def _start_reward(self):
        # super(ManualControlMulti, self)._start_reward()
        self.targets[self.target_index % 2].show()
        self.righthand = (np.random.random()>0.5)
        if self.righthand==1:
            self.target_color = self.target_b_color
            self.targ_shift = np.array([5,5,0])
        else:
            self.target_color = self.target_a_color
            self.targ_shift = np.array([-5,5,0])


    def _end_move_reward(self):
        pass#self._end_reward()
        

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx % 2]
            target.move_to_position(self.targs[idx])
        # self.reward_time = self.reward_time/1.5
        # self._start_reward()
        # self.reward_time = self.reward_time*1.5
        self.consec_move_rews = 0

        

class BimanualMulti_Coord(ManualControlMulti):

    '''
    This is a task written for exp 2.3 of the ipsilateral control project. The task uses joystick and LED motion tracking 
    to coordinate joystick pulls at distinct phases of a reaching action.
    Written by Tanner Dixon March 9, 2017
    '''

    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", stop=None),
        hold = dict(leave_early="hold_penalty", first_hold_complete="first_action", final_hold_complete="targ_transition", stop=None),

        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", stop=None),

        first_action = dict(first_initiated="delay", second_initiated="hold_penalty", stop=None),
        delay = dict(second_initiated="hold_penalty", delay_complete="second_action", stop=None),
        second_action = dict(too_late="hold_penalty", second_initiated="targ_transition", timeout="timeout_penalty", stop=None),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", stop=None),
        hold_penalty = dict(hold_penalty_end="targ_transition", stop=None),
        reward = dict(reward_end="wait")
    )

    cursor_color = (1,0,0,0.5)
    cursor_radius = traits.Float(.5, desc="Radius of the reaching cursor")

    target_color = (1,0,0,0.5)
    target_radius = traits.Float(4, desc="Radius of targets in cm")

    plant_visible = traits.Bool(True, desc='Specifies whether entire plant a is displayed or just endpoint')

    marker_num = traits.Int(19, desc="The index of the motiontracker marker to use for the holding cursor position")

    is_pull_first = traits.Bool(True, desc='Specifies whether the pull or the reach is the first action')
    sequence_latency = traits.Float(0, desc="Time between initiation of first action and beginning of second action time window (s)")
    second_action_window = traits.Float(0.5, desc="The length of the window for second action in sequence (s)")



    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        if self.righthand:
            self.targ_shift = np.array([10,1,-1])
            self.target_color = PURPLE
        else:
            self.targ_shift = np.array([-10,1,-1])
            self.target_color = YELLOW

        self.alreward = 0

        # Initialize the plants
        # if not hasattr(self, 'plant'):
        self.plant = plantlist[self.plant_type]
        self.plant_vis_prev = True

        # Add graphics models for the plants and targets to the window
        for model in self.plant.graphics_models:
            self.add_model(model)
            print self.plant.graphics_models

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            self.target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_color)
            self.target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_color)

            self.targets = [self.target1, self.target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
        
        # Initialize target location variables
        self.target_location = np.array([0, 0, 0])
        if self.righthand:
            self.start_location = np.array([10,1,-1])
        else:
            self.start_location = np.array([-10,1,-1])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        # self.add_dtype('cursor_a', 'f8', (3,))
        # self.add_dtype('cursor_b', 'f8', (3,))
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)
        self.no_data_count = 0
        self.last_rew_pt = np.zeros([2,3])
        self.current_pt = np.zeros([3])

        self.recent=np.zeros([30,3])
        self.lastreal=0
        self.isstart=1

        self.rew = np.copy(self.reward_time)

        self.first_action = False
        self.second_action = False


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

        self.move_arm_motiontracker()


        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(ManualControlMulti, self)._cycle()

        # if self.joy_y < -1:
        #     self.reward_time = 0 # just to cause the beep
        #     self._start_reward()
        #     self.reward_time = float(self.rew) # return to proper reward time

        
    
    def move_arm_motiontracker(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        # Get all data from motion tracker since last update (~4 samples = 240Hz motiontracker / 60Hz task)
        pt = self.motiondata.get()

        if len(pt) > 0:
            # Select given marker number and identify indices of reliable samples
            pt = pt[:, self.marker_num, :]
            conds = pt[:, 3]
            inds = np.nonzero(conds>=0)[0]
            if len(inds) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt = pt[inds,:3]             
                pt = pt.mean(0) #* self.scale_factor     
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt = pt * mm_per_cm 
                reorder = np.copy(pt)
                pt[0]= -5-reorder[0]
                pt[1]= 4+reorder[2]
                pt[2]= -2+reorder[1]
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count = 0
            else: #if no usable data
                self.no_data_count+=1
                pt = None

        else: #if no new data
            self.no_data_count+=1
            pt = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available

        # Keep 500ms (30 task samples) running log of values returned by motiontracker so that empty spots can be interpolated
        self.recent = np.roll(self.recent,1,axis=0)
        self.recent[0,:]=pt

        # If any block of motiontracker data in the recent history is bookended by valid samples, linearly interpolate between
        # This is largely defunct when not having a delay to gather more data
        if pt is not None:
            if self.lastreal > 0 and self.lastreal < 30:
                self.recent[0:self.lastreal+1,0]=np.linspace(self.recent[0,0],self.recent[self.lastreal,0],self.lastreal+1)
                self.recent[0:self.lastreal+1,1]=np.linspace(self.recent[0,1],self.recent[self.lastreal,1],self.lastreal+1)
                self.recent[0:self.lastreal+1,2]=np.linspace(self.recent[0,2],self.recent[self.lastreal,2],self.lastreal+1)
            self.lastreal = 0
        else:
            if self.lastreal < 3:
                self.recent[0,:]=self.recent[1,:]+0.2*(self.recent[1,:]-self.recent[2,:])
            self.lastreal += 1

        # Delay feedback to allow for interpolation to affect feedback - this will prevent the cursor from
        # disappearing as long as there is at least one valid sample in both before and after the current position polling
        self.plant.set_endpoint_pos(self.recent[0,:])
        self.current_pt = np.copy(self.recent[0,:])

        '''
        Joystick tracking for 'pull'
        '''
        #get data from phidget
        pt_joy = self.joystick.get()
        #print pt_joy

        if len(pt_joy) > 0:
            pt_joy = pt_joy[-1][0]

            #pt_joy[1]=1-pt_joy[1]; #Switch U / D axes
            #pt_joy[0]=1-pt_joy[0]; #Switch L / R axes
            calib = [0.5,0.5] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
            # calib = [ 0.487,  0.   ]

            pos = np.array([(pt_joy[0]-calib[0]) , 0, (calib[1]-pt_joy[1])])
            pos[0] = pos[0]*36
            pos[2] = pos[2]*24
            self.joy_y = pos[0]



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


    #### TEST FUNCTIONS ####
    def _test_first_initiated(self,ts):
        if self.is_pull_first:
            return (self.joy_y < -5)
        else:
            cursor_pos = self.plant.get_endpoint_pos()
            return (np.linalg.norm(cursor_pos - self.start_location) > 3)

    def _test_second_initiated(self,ts):
        if self.is_pull_first:
            cursor_pos = self.plant.get_endpoint_pos()
            return (np.linalg.norm(cursor_pos - self.start_location) > 3)
        else:
            return (self.joy_y < -5)

    def _test_too_late(self, ts):
        return ts > (self.sequence_latency + self.second_action_window)

    def _test_on_time(self, ts):
        return (self.second_action) & (ts>self.reach_latency) & (ts < (self.pull_latency + self.reach_window_length) )

    def _test_trial_complete(self, ts):
        return self.target_index==self.chain_length-1

    def _test_hold_a_complete(self, ts):
        return ts>=2*self.hold_time

    def _test_first_hold_complete(self, ts):
        if self.target_index ==0:
            return ts>=self.instr_delay
        else:
            return False

    def _test_final_hold_complete(self, ts):
        if self.target_index ==1:
            return ts>=self.hold_time
        else:
            return False

    def _test_delay_complete(self, ts):
        return ts>=self.sequence_latency


    #### STATE FUNCTIONS ####
    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index % 2]
        if self.target_index+1 < self.chain_length:
            self.target_location = self.targs[self.target_index]+self.targ_shift
        else:
            self.target_location = self.targs[self.target_index]+np.array([0,1,-1])
        target.move_to_position(self.target_location)
        target.cue_trial_start()
        target.change_color(self.target_color)
        # if self.target_index==0:
        #     self.reward_time = self.reward_time/10
        #     self._start_reward()
        #     self.reward_time = self.reward_time*10

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx % 2]
            new_loc = self.targs[idx] + np.array([0,1,-1])
            target.move_to_position(new_loc)
            for target in self.targets:
                target.show()
                target.change_color(self.target_color)
        self.alreward = 0

    def _end_hold(self):
        # change current target color to green
        self.targets[self.target_index % 2].cue_trial_end_success()

    def _start_first_action(self):
        self.start_location = self.plant.get_endpoint_pos()
        if self.is_pull_first:
            target = self.targets[self.target_index % 2]
            target.change_color(RED)
        else:
            target = self.targets[0]
            target.hide()
            self.reward_time = 0 # just to cause the beep
            self._start_reward()
            self.reward_time = float(self.rew) # return to proper reward time


    def _end_first_action(self):
        target = self.targets[self.target_index % 2]
        target.change_color(self.target_color)
        self.reward_time = 0 # just to cause the beep
        self._start_reward()
        self.reward_time = float(self.rew) # return to proper reward time

    def _start_second_action(self):
        if self.is_pull_first:
            # target = self.targets[0]
            # target.hide()           
            self.reward_time = 0 # just to cause the beep
            self._start_reward()
            self.reward_time = float(self.rew) # return to proper reward time
            for target in self.targets:
                target.hide()
            target = self.targets[1]
            target.show()
        else:
            target = self.targets[self.target_index % 2]
            target.change_color(RED)

    # def _end_delay(self):
    #     for target in self.targets:
    #         target.hide()


    def _end_second_action(self):
        target = self.targets[self.target_index % 2]
        target.change_color(self.target_color)
    
    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()
        self.target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_targ_transition(self):
        #instructed delay (off)
        # if self.target_index==0:
        #     self.reward_time = 0 # just to cause the beep
        #     self._start_reward()
        #     self.reward_time = float(self.rew) # return to proper reward time
        #hide targets
        for target in self.targets:
            target.hide()

    def _start_reward(self):
        # super(ManualControlMulti, self)._start_reward()
        self.targets[self.target_index % 2].show()
        # self.targets[self.target_index % 2].cue_trial_end_success()

    def _start_hold_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets:
            target.hide()
        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial


class BimanualMulti_Indep_train_A(ManualControlMulti):

    '''
    This is a task written for exp 2.3 of the ipsilateral control project. The task uses joystick and LED motion tracking 
    to coordinate joystick pulls at the completion of a reach.
    Written by Tanner Dixon March 9, 2017
    '''

    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())


    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", joy_pull="hold_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition", joy_pull="hold_penalty", stop=None),
        targ_transition = dict(trial_complete="delay",trial_abort="wait", trial_incomplete="target", stop=None),

        delay = dict(delay_complete="joy_pull_window", joy_pull="hold_penalty", stop=None),
        joy_pull_window = dict(joy_pull="reward", too_late="hold_penalty", stop=None),

        timeout_penalty = dict(timeout_penalty_end="targ_transition", stop=None),
        hold_penalty = dict(hold_penalty_end="targ_transition", stop=None),
        reward = dict(reward_end="wait")
    )




    cursor_color = (1,0,0,0.5)
    cursor_radius = traits.Float(.5, desc="Radius of the reaching cursor")

    target_color = (1,0,0,0.5)
    target_radius = traits.Float(4, desc="Radius of targets in cm")

    plant_visible = traits.Bool(True, desc='Specifies whether entire plant a is displayed or just endpoint')

    marker_num = traits.Int(19, desc="The index of the motiontracker marker to use for the holding cursor position")

    pull_latency = traits.Float(0, desc="Time between entering target and start of pull window (s)")
    pull_window_length = traits.Float(0.5, desc="The length of the window for pulling the joystick (s)")



    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        if self.righthand:
            self.targ_shift = np.array([4,15,-5.5])
            self.target_color = PURPLE
        else:
            self.targ_shift = np.array([-4,15,-5.5])
            self.target_color = YELLOW

        self.alreward = 0

        # Initialize the plants
        # if not hasattr(self, 'plant'):
        self.plant = plantlist[self.plant_type]
        self.plant_vis_prev = True

        # Add graphics models for the plants and targets to the window
        for model in self.plant.graphics_models:
            self.add_model(model)
            print self.plant.graphics_models

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            self.target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_color)
            self.target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_color)
            self.pull_cue = VirtualCircularTarget(target_radius=(self.target_radius+1), target_color=RED)

            self.targets = [self.target1, self.target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
            for model in self.pull_cue.graphics_models:
                self.add_model(model)
        
        # Initialize target location variables
        self.target_location = np.array([0, 0, 0])
        if self.righthand:
            self.start_location = np.array([15,15,-5.5])
        else:
            self.start_location = np.array([-15,15,-5.5])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        # self.add_dtype('cursor_a', 'f8', (3,))
        # self.add_dtype('cursor_b', 'f8', (3,))
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)
        self.no_data_count = 0
        self.last_rew_pt = np.zeros([2,3])
        self.current_pt = np.zeros([3])

        self.recent=np.zeros([30,3])
        self.lastreal=0
        self.isstart=1

        self.rew = np.copy(self.reward_time)

        self.reach = False
        self.second_action = False

        self.joy_buffer = 3



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

        self.move_arm_motiontracker()


        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(ManualControlMulti, self)._cycle()

        # if self.joy_y < -1:
        #     self.reward_time = 0 # just to cause the beep
        #     self._start_reward()
        #     self.reward_time = float(self.rew) # return to proper reward time

        
    
    def move_arm_motiontracker(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        # Get all data from motion tracker since last update (~4 samples = 240Hz motiontracker / 60Hz task)
        pt = self.motiondata.get()

        if len(pt) > 0:
            # Select given marker number and identify indices of reliable samples
            pt = pt[:, self.marker_num, :]
            conds = pt[:, 3]
            inds = np.nonzero(conds>=0)[0]
            if len(inds) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt = pt[inds,:3]             
                pt = pt.mean(0) #* self.scale_factor     2
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt = pt * mm_per_cm 
                reorder = np.copy(pt)
                pt[0]= -5-reorder[0]
                pt[1]= 18+reorder[2]
                pt[2]= -2+reorder[1]
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count = 0
            else: #if no usable data
                self.no_data_count+=1
                pt = None

        else: #if no new data
            self.no_data_count+=1
            pt = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available

        # Keep 500ms (30 task samples) running log of values returned by motiontracker so that empty spots can be interpolated
        self.recent = np.roll(self.recent,1,axis=0)
        self.recent[0,:]=pt

        # If any block of motiontracker data in the recent history is bookended by valid samples, linearly interpolate between
        # This is largely defunct when not having a delay to gather more data
        if pt is not None:
            if self.lastreal > 0 and self.lastreal < 30:
                self.recent[0:self.lastreal+1,0]=np.linspace(self.recent[0,0],self.recent[self.lastreal,0],self.lastreal+1)
                self.recent[0:self.lastreal+1,1]=np.linspace(self.recent[0,1],self.recent[self.lastreal,1],self.lastreal+1)
                self.recent[0:self.lastreal+1,2]=np.linspace(self.recent[0,2],self.recent[self.lastreal,2],self.lastreal+1)
            self.lastreal = 0
        else:
            if self.lastreal < 3:
                self.recent[0,:]=self.recent[1,:]+0.8*(self.recent[1,:]-self.recent[2,:])
            self.lastreal += 1

        # Delay feedback to allow for interpolation to affect feedback - this will prevent the cursor from
        # disappearing as long as there is at least one valid sample in both before and after the current position polling
        self.plant.set_endpoint_pos(self.recent[0,:])
        self.current_pt = np.copy(self.recent[0,:])

        '''
        Joystick tracking for 'pull'
        '''
        #get data from phidget
        pt_joy = self.joystick.get()
        #print pt_joy

        if len(pt_joy) > 0:
            pt_joy = pt_joy[-1][0]

            #pt_joy[1]=1-pt_joy[1]; #Switch U / D axes
            #pt_joy[0]=1-pt_joy[0]; #Switch L / R axes
            calib = [0.5,0.5] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
            # calib = [ 0.487,  0.   ]

            pos = np.array([(pt_joy[0]-calib[0]) , 0, (calib[1]-pt_joy[1])])
            pos[0] = pos[0]*36
            pos[2] = pos[2]*24
            self.joy_y = pos[0]



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


    #### TEST FUNCTIONS ####
    def _test_joy_pull(self,ts):
        return (self.joy_y < (-11 + self.joy_buffer) )

    def _test_too_late(self, ts):
        return ts > self.pull_window_length

    def _test_trial_complete(self, ts):
        return self.target_index==self.chain_length-1

    def _test_delay_complete(self, ts):
        return ts>= np.abs(np.random.randn(1)/10 + self.pull_latency)

    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius - self.cursor_radius)
        # return True
        
    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius - self.cursor_radius
        return d > rad
        # return False

    def _test_hold_complete(self, ts):
        return ts>=np.abs(np.random.randn(1)/10 + self.hold_time)


    #### STATE FUNCTIONS ####
    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index % 2]
        if self.target_index==1:
            self.target_location = np.array([-15,28,5])
        else:
            self.target_location = self.start_location

        # self.target_location = self.start_location

        target.move_to_position(self.target_location)
        target.cue_trial_start()
        target.change_color(self.target_color)
        # if self.target_index==1:
        #     self.reward_time = self.reward_time/5
        #     self._start_reward()
        #     self.reward_time = self.reward_time*5
        # print self.joy_y

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx]
            target.move_to_position(self.start_location)
            for target in self.targets:
                target.show()
                target.change_color(self.target_color)
        self.alreward = 0

    def _end_hold(self):
        # change current target color to green
        self.targets[self.target_index % 2].cue_trial_end_success()

    def _start_joy_pull_window(self):
        self.pull_cue.move_to_position(self.target_location)
        self.pull_cue.show()
        self.joy_buffer=0

    def _end_joy_pull_window(self):
        self.pull_cue.hide()

    # def _end_delay(self):
    #     for target in self.targets:
    #         target.hide()

    
    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_targ_transition(self):
        #instructed delay (off)
        # if self.target_index==0:
        #     self.reward_time = 0 # just to cause the beep
        #     self._start_reward()
        #     self.reward_time = float(self.rew) # return to proper reward time
        #hide targets
        for target in self.targets:
            target.hide()

    def _start_reward(self):
        # super(ManualControlMulti, self)._start_reward()
        self.targets[self.target_index % 2].show()
        # self.targets[self.target_index % 2].cue_trial_end_success()
        self.joy_buffer = 3
        # print self.joy_y

    def _start_hold_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets:
            target.hide()
        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial
        self.pull_cue.hide()

        

class BimanualMulti_ReachPull(ManualControlMulti):

    '''
    This is a task written for exp 2.3 of the ipsilateral control project. The task uses joystick and LED motion tracking 
    to coordinate joystick pulls at the completion of a reach.
    Written by Tanner Dixon April 18, 2017


    THIS COULD BE REWRITTEN TO WHERE ALL PHASES ARE A SINGLE STATE OF THE FSM, AND A COUNTER KEEPS TRACK OF WHERE IT IS
    IN THE PROGRESSION. FOR NOW, I THOUGHT THIS MADE IT MORE READABLE (THOUGH MUCH MORE VERBOSE).
    '''

    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())


    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(hold_complete="targ_transition", leave_early="hold_penalty", stop=None),
        targ_transition = dict(trial_incomplete="phase_one", trial_complete="phase_three", trial_abort="wait", stop=None),

        phase_one = dict(advance="phase_two", wrong_time="hold_penalty", stop=None),
        phase_two = dict(advance="target", wrong_time="hold_penalty", stop=None),

        phase_three = dict(advance="phase_four", wrong_time="hold_penalty", stop=None),
        phase_four = dict(advance="reward", wrong_time="hold_penalty", stop=None),

        timeout_penalty = dict(timeout_penalty_end="wait", stop=None),
        hold_penalty = dict(hold_penalty_end="wait", stop=None),
        reward = dict(reward_end="wait")
    )




    cursor_color = (1,0,0,0.5)
    cursor_radius = traits.Float(.5, desc="Radius of the reaching cursor")

    target_color = (1,0,0,0.5)
    target_radius = traits.Float(4, desc="Radius of targets in cm")

    plant_visible = traits.Bool(True, desc='Specifies whether entire plant a is displayed or just endpoint')

    marker_num = traits.Int(18, desc="The index of the motiontracker marker to use for the holding cursor position")

    pull_response_time = traits.Float(0.5, desc="The amount of time for pull response window (s)")
    coord = traits.Int(0, desc="Specifies whether the reach and pull are consistently coordinated or randomly cued")



    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        if self.righthand:
            self.targ_shift = np.array([4,15,-5.5])
            self.target_color = PURPLE
        else:
            self.targ_shift = np.array([-4,15,-5.5])
            self.target_color = YELLOW

        self.alreward = 0

        # Initialize the plants
        # if not hasattr(self, 'plant'):
        self.plant = plantlist[self.plant_type]
        self.plant_vis_prev = True

        # Add graphics models for the plants and targets to the window
        for model in self.plant.graphics_models:
            self.add_model(model)
            print self.plant.graphics_models

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            self.target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_color)
            self.target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_color)
            self.pull_cue = VirtualCircularTarget(target_radius=(self.target_radius+1), target_color=RED)

            self.targets = [self.target1, self.target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
            for model in self.pull_cue.graphics_models:
                self.add_model(model)
        
        # Initialize target location variables
        self.target_location = np.array([15, 15, -5.5])
        if self.righthand:
            self.start_location = np.array([15,15,-5.5])
        else:
            self.start_location = np.array([-15,15,-5.5])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        # self.add_dtype('cursor_a', 'f8', (3,))
        # self.add_dtype('cursor_b', 'f8', (3,))
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)
        self.no_data_count = 0
        self.last_rew_pt = np.zeros([2,3])
        self.current_pt = np.zeros([3])

        self.recent=np.zeros([30,3])
        self.lastreal=0
        self.isstart=1

        self.rew = np.copy(self.reward_time)

        self.prev_joy_pull = False
        self.joy_pull = False            #follows whether the joystick is currently being pulled, binary instantaneous measure
        self.pull_completed = False      #keeps track of whether the pull required for trial completion has been accomplished yet
        self.current_phase = 1
        self.pull_phase = 3



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

        self.move_arm_motiontracker()


        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(ManualControlMulti, self)._cycle()

        
    
    def move_arm_motiontracker(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        # Get all data from motion tracker since last update (~4 samples = 240Hz motiontracker / 60Hz task)
        pt = self.motiondata.get()

        if len(pt) > 0:
            # Select given marker number and identify indices of reliable samples
            pt = pt[:, self.marker_num, :]
            conds = pt[:, 3]
            inds = np.nonzero(conds>=0)[0]
            if len(inds) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt = pt[inds,:3]             
                pt = pt.mean(0) #* self.scale_factor     
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt = pt * mm_per_cm 
                reorder = np.copy(pt)
                pt[0]= -5-reorder[0]
                pt[1]= 18+reorder[2]
                pt[2]= -2+reorder[1]
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count = 0
            else: #if no usable data
                self.no_data_count+=1
                pt = None

        else: #if no new data
            self.no_data_count+=1
            pt = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available

        # Keep 500ms (30 task samples) running log of values returned by motiontracker so that empty spots can be interpolated
        # NOT CURRENTLY BEING USED
        self.recent = np.roll(self.recent,1,axis=0)
        self.recent[0,:]=pt

        # If any block of motiontracker data in the recent history is bookended by valid samples, linearly interpolate between
        # This is largely defunct when not having a delay to gather more data
        if pt is not None:
            if self.lastreal > 0 and self.lastreal < 30:
                self.recent[0:self.lastreal+1,0]=np.linspace(self.recent[0,0],self.recent[self.lastreal,0],self.lastreal+1)
                self.recent[0:self.lastreal+1,1]=np.linspace(self.recent[0,1],self.recent[self.lastreal,1],self.lastreal+1)
                self.recent[0:self.lastreal+1,2]=np.linspace(self.recent[0,2],self.recent[self.lastreal,2],self.lastreal+1)
            self.lastreal = 0
        # If the last valid sample was not too long ago, extrapolate missing data points
        else:
            if self.lastreal < 3:
                self.recent[0,:]=self.recent[1,:]+0.8*(self.recent[1,:]-self.recent[2,:])
            self.lastreal += 1

        # Delay feedback to allow for interpolation to affect feedback - this will prevent the cursor from
        # disappearing as long as there is at least one valid sample in both before and after the current position polling
        # NOT CURRENTLY BEING USED
        self.plant.set_endpoint_pos(self.recent[0,:])
        self.current_pt = np.copy(self.recent[0,:])


        '''
        Joystick tracking for 'pull'
        '''
        #get data from phidget
        pt_joy = self.joystick.get()
        #print pt_joy

        if len(pt_joy) > 0:
            pt_joy = pt_joy[-1][0]

            #pt_joy[1]=1-pt_joy[1]; #Switch U / D axes
            #pt_joy[0]=1-pt_joy[0]; #Switch L / R axes
            calib = [0.5,0.5] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
            # calib = [ 0.487,  0.   ]
            pos = np.array([(pt_joy[0]-calib[0]) , 0, (calib[1]-pt_joy[1])])
            pos[0] = pos[0]*36
            pos[2] = pos[2]*24
            self.joy_y = pos[0]

        #if the joystick has not been pulled since the beginning of the phase when self.joy_pull was reset, test to see if the joystick
        #has been pulled. If it has been pulled, hide the pull cue to indicate completion of the response
        self.prev_joy_pull = self.joy_pull
        self.joy_pull = (self.joy_y < -5)
        if self.joy_pull and self.current_phase>2 and not self.prev_joy_pull:
            self.pull_cue.hide()
            self.pull_completed = True



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


    #### TEST FUNCTIONS ####
    def _test_wrong_time(self,ts):
        if (self.current_phase==self.pull_phase) and (ts >= self.pull_response_time):    
            return (not self.pull_completed)                        #if this is the pull phase, test whether time ran out without a pull
        elif (self.current_phase!=self.pull_phase) and (ts >= 0.5):                                                      
            return False#self.joy_pull                                    #if this was not the pull phase, test whether there was an inappropriate pull
        else:                                                       #allowing time for release of the joystick from a previous phase
            return False

    def _test_advance(self,ts):
        if (self.current_phase % 2):                                #assign lengths of each phase of the trial
            phase_length = 0.6
        else:
            phase_length = 0.6

        if (self.current_phase >= 3) and (self.current_phase == self.pull_phase):                               #do not wait until the end of post-reach phases to assess whether the correct
            return self.pull_completed
        elif (self.current_phase==4):
            return self.pull_completed                              #action/inaction was performed. Aint nobody got time to wait for reward 
        elif (ts >= phase_length):                                   
            if (self.current_phase==self.pull_phase):               #if the end of a pre-reach phase arrives without early failure, advance if
                return self.pull_completed                          #the current action/inaction was performed
            else:
                return True  
        else:
            return False

    def _test_hold_complete(self, ts):
        if self.target_index == 1:
            return ts>=0.3
        else:
            return ts>=0.3


    #### STATE FUNCTIONS ####
    def _start_target(self):
        self.current_phase=0
        self.target_index += 1
        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index % 2]
        # self.target_location = self.targs[self.target_index] + self.targ_shift

        if self.target_index==1:
            self.target_location = np.array([-15,28,5])
        else:
            self.target_location = self.start_location

        target.move_to_position(self.target_location)
        target.cue_trial_start()
        #deliver audio cue and hide home target for reach
        if self.target_index==1:
            self.reward_time = 0.3 # just to cause the beep for go cue
            self._start_reward()
            self.reward_time = float(self.rew) # return to proper reward time
            self.targets[0].hide()

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx % 2]
            # target.move_to_position(self.targs[idx]+self.targ_shift)
            target.move_to_position(np.array([-15,28,5]))
            target.show()
            target.change_color(self.target_color)

    def _start_targ_transition(self):
        pass

    def _start_phase_one(self):
        #reset self.joy_pull event tracker, show pull cue if necessary during this phase
        for target in self.targets:
            target.show()
            target.change_color(self.target_color)
        self.current_phase=1
        if self.pull_phase==1:
            self.pull_cue.move_to_position(self.target_location)
            self.pull_cue.show()
        # print 'phase 1'

    def _start_phase_two(self):
        #reset self.joy_pull event tracker, show pull cue if necessary during this phase
        self.current_phase=2
        if self.pull_phase==2:
            self.pull_cue.move_to_position(self.target_location)
            self.pull_cue.show()
        # print 'phase 2'

    def _start_phase_three(self):
        #reset self.joy_pull event tracker, show pull cue if necessary during this phase
        self.targets[1].show()
        self.current_phase=3
        if self.pull_phase==3:
            self.pull_cue.move_to_position(self.target_location)
            self.pull_cue.show()
        # print 'phase 3'

    def _start_phase_four(self):
        #reset self.joy_pull event tracker, show pull cue if necessary during this phase
        self.current_phase=4
        if self.pull_phase==4:
            self.pull_cue.move_to_position(self.target_location)
            self.pull_cue.show()
        # print 'phase 4'

    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()
        self.pull_cue.hide()

        self.tries += 1
        self.target_index = -1
        self.pull_completed = False
        self.current_phase = 0

    def _start_hold_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()
        self.pull_cue.hide()

        self.tries += 1
        self.target_index = -1

        # print 'hold penalty'
        # print self.joy_pull
        self.pull_completed = False
        self.current_phase = 0

    def _start_reward(self):
        # super(ManualControlMulti, self)._start_reward()
        self.targets[self.target_index % 2].show()
        self.current_phase = 0

    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets:
            target.hide()
        self.pull_cue.hide()
        self.chain_length = self.targs.shape[0]         #Number of sequential targets in a single trial
        if not self.coord:                              #Assign random pull phase if indep condition
            self.pull_phase = np.random.randint(4)+1
        self.pull_completed = False
        self.current_phase = 0
        print self.pull_phase

class BimanualMulti_ReachPull_B(ManualControlMulti):

    '''
    This is a task written for exp 2.3 of the ipsilateral control project. The task uses joystick and LED motion tracking 
    to coordinate joystick pulls with reaching movements.
    Written by Tanner Dixon April 21, 2017
    '''

    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())


    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(hold_complete="targ_transition", leave_early="hold_penalty", stop=None),
        targ_transition = dict(to_pull="pull", to_reach="reach", trial_complete="reward", trial_abort="wait", stop=None),

        pull = dict(first_action_completed="delay", second_action_completed="target", improper_response="response_penalty", stop=None),
        reach = dict(first_action_completed="delay", second_action_completed="target", improper_response="response_penalty", stop=None),
        delay = dict(delay_complete="action_transition", second_action_completed="response_penalty", stop=None),
        action_transition = dict(to_pull="pull", to_reach="reach", stop=None),

        timeout_penalty = dict(timeout_penalty_end="wait", stop=None),
        hold_penalty = dict(hold_penalty_end="wait", stop=None),
        response_penalty = dict(response_penalty_end="wait", stop=None),
        reward = dict(reward_end="wait")
    )




    cursor_color = (1,0,0,0.5)
    cursor_radius = traits.Float(.5, desc="Radius of the reaching cursor")

    target_color = (1,0,0,0.5)
    target_radius = traits.Float(4, desc="Radius of targets in cm")

    plant_visible = traits.Bool(True, desc='Specifies whether entire plant a is displayed or just endpoint')

    marker_num = traits.Int(18, desc="The index of the motiontracker marker to use for the holding cursor position")

    response_time = traits.Float(0.5, desc="The duration of the response window following reach or pull cues (s)")
    coord = traits.Int(1, desc="Specifies whether the reach and pull are consistently coordinated or randomly cued")

    



    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        if self.righthand:
            self.targ_shift = np.array([4,15,-5.5])
            self.target_color = PURPLE
        else:
            self.targ_shift = np.array([-4,15,-5.5])
            self.target_color = YELLOW

        self.alreward = 0

        # Initialize the plants
        # if not hasattr(self, 'plant'):
        self.plant = plantlist[self.plant_type]
        self.plant_vis_prev = True

        # Add graphics models for the plants and targets to the window
        for model in self.plant.graphics_models:
            self.add_model(model)
            print self.plant.graphics_models

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            self.target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_color)
            self.target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self.target_color)
            self.pull_cue = VirtualCircularTarget(target_radius=(self.target_radius+1), target_color=RED)

            self.targets = [self.target1, self.target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
            for model in self.pull_cue.graphics_models:
                self.add_model(model)
        
        # Initialize target location variables
        self.target_location = np.array([0, 0, 0])
        if self.righthand:
            self.start_location = np.array([4,15,-5.5])
        else:
            self.start_location = np.array([-4,15,-5.5])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        # self.add_dtype('cursor_a', 'f8', (3,))
        # self.add_dtype('cursor_b', 'f8', (3,))
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)
        self.no_data_count = 0
        self.last_rew_pt = np.zeros([2,3])
        self.current_pt = np.zeros([3])

        self.recent=np.zeros([30,3])
        self.lastreal=0
        self.isstart=1

        self.rew = np.copy(self.reward_time)
        self.joy_buffer = 7

        self.joy_pull = False            #follows whether the joystick is currently being pulled, binary instantaneous measure
        self.pull_first = 0
        self.delay_time = 0.3
        self.first_action_completed = 0



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

        self.move_arm_motiontracker()


        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(ManualControlMulti, self)._cycle()

        
    
    def move_arm_motiontracker(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        # Get all data from motion tracker since last update (~4 samples = 240Hz motiontracker / 60Hz task)
        pt = self.motiondata.get()

        if len(pt) > 0:
            # Select given marker number and identify indices of reliable samples
            pt = pt[:, self.marker_num, :]
            conds = pt[:, 3]
            inds = np.nonzero(conds>=0)[0]
            if len(inds) > 0: #some usable data
                # Take  mean of all reliable samples since last cycle and scale actual movement to desired amount of screen movement
                pt = pt[inds,:3]             
                pt = pt.mean(0) #* self.scale_factor     
                # Convert to cm and adjust for desired screen position, reorder axes if needed
                pt = pt * mm_per_cm 
                reorder = np.copy(pt)
                pt[0]= -5-reorder[0]
                pt[1]= 18+reorder[2]
                pt[2]= -2+reorder[1]
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: 
                    pt[1] = 0
                # Keep track of how many consecutive task cycles have returned no valid data
                self.no_data_count = 0
            else: #if no usable data
                self.no_data_count+=1
                pt = None

        else: #if no new data
            self.no_data_count+=1
            pt = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available

        # Keep 500ms (30 task samples) running log of values returned by motiontracker so that empty spots can be interpolated
        # NOT CURRENTLY BEING USED
        self.recent = np.roll(self.recent,1,axis=0)
        self.recent[0,:]=pt

        # If any block of motiontracker data in the recent history is bookended by valid samples, linearly interpolate between
        # This is largely defunct when not having a delay to gather more data
        if pt is not None:
            if self.lastreal > 0 and self.lastreal < 30:
                self.recent[0:self.lastreal+1,0]=np.linspace(self.recent[0,0],self.recent[self.lastreal,0],self.lastreal+1)
                self.recent[0:self.lastreal+1,1]=np.linspace(self.recent[0,1],self.recent[self.lastreal,1],self.lastreal+1)
                self.recent[0:self.lastreal+1,2]=np.linspace(self.recent[0,2],self.recent[self.lastreal,2],self.lastreal+1)
            self.lastreal = 0
        # If the last valid sample was not too long ago, extrapolate missing data points
        else:
            if self.lastreal < 3:
                self.recent[0,:]=self.recent[1,:]+0.8*(self.recent[1,:]-self.recent[2,:])
            self.lastreal += 1

        # Delay feedback to allow for interpolation to affect feedback - this will prevent the cursor from
        # disappearing as long as there is at least one valid sample in both before and after the current position polling
        # NOT CURRENTLY BEING USED
        self.plant.set_endpoint_pos(self.recent[0,:])
        self.current_pt = np.copy(self.recent[0,:])


        '''
        Joystick tracking for 'pull'
        '''
        #get data from phidget
        pt_joy = self.joystick.get()
        #print pt_joy

        if len(pt_joy) > 0:
            pt_joy = pt_joy[-1][0]

            #pt_joy[1]=1-pt_joy[1]; #Switch U / D axes
            #pt_joy[0]=1-pt_joy[0]; #Switch L / R axes
            calib = [0.5,0.5] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
            # calib = [ 0.487,  0.   ]
            pos = np.array([(pt_joy[0]-calib[0]) , 0, (calib[1]-pt_joy[1])])
            pos[0] = pos[0]*36
            pos[2] = pos[2]*24
            self.joy_y = pos[0]

        #if the joystick has not been pulled since the beginning of the phase when self.joy_pull was reset, test to see if the joystick
        #has been pulled. If it has been pulled, hide the pull cue to indicate completion of the response
        self.joy_pull = (self.joy_y < -5)
        if self.joy_pull and self.current_phase>0:
            self.pull_cue.hide()
            self.pull_completed = True



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


    #### TEST FUNCTIONS ####
    def _test_joy_pull(self,ts):
        return (self.joy_y < -12 + self.joy_buffer)

    def _test_reach_initiated(self,ts):
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius - self.cursor_radius
        return d > rad

    def _test_first_action_completed(self,ts):
        if not self.first_action_completed:
            if self.pull_first:
                return self._test_joy_pull(ts)
            else:
                return self._test_reach_initiated(ts)
        else:
            return False

    def _test_second_action_completed(self,ts):
        if self.first_action_completed:
            if self.pull_first:
                return self._test_reach_initiated(ts)
            else:
                return self._test_joy_pull(ts)
        else:
            return False

    def _test_improper_response(self,ts):
        if (ts > self.response_time):
            return True
        elif self.first_action_completed:
            return False
        elif self.pull_first:
            return self._test_reach_initiated(ts)
        elif not self.pull_first:
            return self._test_joy_pull(ts)

    def _test_hold_complete(self, ts):
        if (self.target_index == 1) and (not self.pull_first):
            return ts>=0.1
        else:
            return ts>=0.3

    def _test_to_pull(self, ts):
        if self.target_index < self.chain_length-1:
            if (not self.first_action_completed):
                return self.pull_first
            else:
                return (not self.pull_first)
        else:
            return False

    def _test_to_reach(self, ts):
        if self.target_index < self.chain_length-1:
            if self.first_action_completed:
                return self.pull_first
            else:
                return (not self.pull_first)
        else:
            return False

    def _test_delay_complete(self, ts):
        return (ts>self.delay_time)

    def _test_response_penalty_end(self, ts):
        return ts>self.hold_penalty_time


    #### STATE FUNCTIONS ####
    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets:
            target.hide()
        self.pull_cue.hide()
        self.chain_length = self.targs.shape[0]         #Number of sequential targets in a single trial
        if not self.coord:                              #Assign random pull phase if indep condition
            self.pull_first = np.random.randint(2)
            self.delay_time = 0.5 #0.3*(np.random.randint(2)+1)
        self.first_action_completed = 0

    def _start_target(self):
        self.current_phase=0
        self.target_index += 1
        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index % 2]
        self.target_location = self.targs[self.target_index] + self.targ_shift
        target.move_to_position(self.target_location)
        target.cue_trial_start()

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx % 2]
            target.move_to_position(self.targs[idx]+self.targ_shift)
            target.show()
            target.change_color(self.target_color)

    def _start_targ_transition(self):
        #hide targets
        # for target in self.targets:
        #     target.hide()
        pass

    def _start_pull(self):
        #show pull cue at reaching target location
        reach_loc = self.targs[self.target_index] + self.targ_shift
        self.pull_cue.move_to_position(reach_loc)
        self.pull_cue.show()

    def _end_pull(self):
        self.pull_cue.hide()

    def _start_reach(self):
        #deliver audio cue and hide home target for reach
        self.reward_time = 0.0 # just to cause the beep for go cue
        self._start_reward()
        self.reward_time = float(self.rew) # return to proper reward time
        self.targets[0].hide()

    def _start_delay(self):
        pass

    def _start_action_transition(self):
        self.first_action_completed = 1

    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()
        self.tries += 1
        self.target_index = -1

    def _start_hold_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()
        self.tries += 1
        self.target_index = -1

    def _start_response_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()
        self.tries += 1
        self.target_index = -1

    def _start_reward(self):
        # super(ManualControlMulti, self)._start_reward()
        self.targets[self.target_index % 2].show()
