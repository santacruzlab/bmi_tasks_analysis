'''
Base tasks for generic point-to-point reaching
'''

from __future__ import division
import numpy as np
import manualcontrolmulti_COtasks
from manualcontrolmultitasks import VirtualCircularTarget
from onedim_lfp_tasks import VirtualSquareTarget
from riglib.experiment import traits, Sequence
import tasks
import math
import traceback

####### CONSTANTS

sec_per_min = 60.0
RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)
mm_per_cm = 1./10

class ManualControlMulti_memory(manualcontrolmulti_COtasks.ManualControlMulti_plusvar):

    target_flash_time = traits.Float(0.2, desc="Time for target to flash")
    flash_target_radius = traits.Float(4., desc="Radius of flahs target cue")
    hold2_time = traits.Float(1.0, desc="hold #2, memory hold")
    hold2_var = traits.Float(.1, desc="hold #2, memory hold")
    ntargets= traits.Int(2)
    status = dict(
        wait = dict(start_trial="origin", stop=None),
        origin = dict(enter_origin="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_flash"),
        targ_flash = dict(targ_flash_done="hold2", leave_orig='hold_penalty'),
        hold2 = dict(leave_early = "hold_penalty", hold_complete='choice_targets'),
        choice_targets = dict(enter_target="periph_hold",enter_wrong_target="wrong_target_penalty",timeout="timeout_penalty"),
        periph_hold = dict(hold_complete="reward",leave_periph_early="hold_penalty"),
        timeout_penalty = dict(timeout_penalty_end="wait"),
        hold_penalty = dict(hold_penalty_end="wait"),
        wrong_target_penalty = dict(timeout_penalty_end="wait"),
        reward = dict(reward_end="wait")
    )

    sequence_generators = ['twoD_choice_CO']

    def __init__(self, *args, **kwargs):
        super(ManualControlMulti_memory, self).__init__(*args, **kwargs)
        self.hold_time_pls_var = self.hold_time #+ np.random.uniform(low=-1,high=1)*self.orig_hold_variance
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
        target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=RED)
        target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=RED)
        choice_targets = []
        for i in range(self.ntargets):
            choice_targets.extend([VirtualCircularTarget(target_radius=self.target_radius, target_color=RED)])

        self.target_dict = dict()
        self.target_dict['mc_orig'] = target1
        self.target_dict['mc_targ'] = target2
        self.target_dict['choice_targets'] = choice_targets
        self.targets = list(choice_targets+[target1, target2])
        for target in self.targets:
            for model in target.graphics_models:
                self.add_model(model)
        
        # Initialize target location variable
        self.target_location = np.array([0, 0, 0])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        # print 'PKK: ', self.dtype
        # for attr in self.plant.hdf_attrs:
        #     print 'PKKKKK: ', attr
        #     self.add_dtype(*attr) 
        self.add_dtype('choice_on', 'f8', (1,))
        self.choice_on = 0
        self.twoD_cursor_pos = np.zeros((3,))

    def init(self):
        super(ManualControlMulti_memory, self).init()
        

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['target'] = self.target_location.copy()
        #self.task_data['target_index'] = self.target_index
        self.task_data['choice_on'] = self.choice_on

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

        pos = self.plant.get_endpoint_pos()
        if self.planar_hand:
            self.twoD_cursor_pos = pos.copy()
        else:
            self.twoD_cursor_pos = pos.copy()

        super(ManualControlMulti_memory, self)._cycle()

    def _parse_next_trial(self):
        t = self.next_trial
        self.mc_targ_loc = t['mc_targ']
        self.mc_label = t['mc_label'][0,0]
        #print 'MC LABEL: ', self.mc_label
       #print 'MC TARG LOC', self.mc_targ_loc
        self.choice_targ_loc = t['choice_targ']
        self.mc_orig_loc = t['mc_orig']

        self.targs = np.vstack((self.mc_orig_loc, self.mc_targ_loc))
 
    def _start_origin(self):
        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        orig = self.target_dict['mc_orig']
        self.target_location = self.mc_orig_loc[0,:]
        orig.move_to_position(self.target_location)
        orig.cue_trial_start()

    def _start_targ_flash(self):
        #Turn on flash
        periph = self.target_dict['mc_targ']
        self.target_location = self.mc_targ_loc[0,:]
        periph.move_to_position(self.target_location)
        periph.cue_trial_start() 

    def _start_hold2(self):
        #Turn off flash: 
        periph = self.target_dict['mc_targ']
        periph.hide()

        self.hold_time_pls_var = self.hold2_time + np.random.uniform(low=-1,high=1)*self.hold2_var
        self.target_location = self.mc_orig_loc[0,:]

    def _start_choice_targets(self):
        for t in self.targets:
            t.hide()

        #Turn on Go cue: 
        orig = self.target_dict['mc_orig']
        orig.cue_trial_end_success() #turn green
        # orig.sphere.color = GREEN
        orig.show()
        #orig.hide()
        
        #Turn on choice targets
        choice_targs = self.target_dict['choice_targets']
        #print 'choice targ loc', self.choice_targ_loc
        #print 'choice shpae: ', choice_targs
        for i, c in enumerate(choice_targs):
            #print i, c
            c.move_to_position(self.choice_targ_loc[i,:])
            c.cue_trial_start()


    def _start_periph_hold(self):
        self.target_dict['mc_orig'].hide()
        #self.target_dict['mc_targ'].hide()
        #for i, c in enumerate( self.target_dict['choice_targets']):
            #c.hide()
        self.hold_time_pls_var = self.hold_time 

    def _start_wrong_target_penalty(self):
        #Turn on choice targets
        choice_targs = self.target_dict['choice_targets']
        #print 'choice targ loc', self.choice_targ_loc
        #print 'choice shpae: ', choice_targs
        for i, c in enumerate(choice_targs):
            #print i, c
            c.move_to_position(self.choice_targ_loc[i,:])
            c.hide()

        orig = self.target_dict['mc_orig']
        orig.hide()


    def _start_reward(self):

        super(ManualControlMulti_memory, self)._start_reward()

        for i, c in enumerate( self.target_dict['choice_targets']):
            c.hide()
        orig = self.target_dict['mc_orig']
        orig.hide()

        for t in self.targets:
            t.hide()
        
        self.target_dict['mc_targ'].sphere.color = GREEN
        self.target_dict['mc_targ'].show()


        # self.target_dict['choice_targets'][int(self.mc_label)].show()

    ########################
    ### TEST FUNCTIONS #####
    ########################
    def _test_leave_orig(self, ts):
        cursor_pos = self.twoD_cursor_pos
        d = np.linalg.norm(cursor_pos - self.mc_orig_loc[0,:])
        rad = self.target_radius - self.cursor_radius
        return d > rad

    def _test_enter_origin(self, ts):
        cursor_pos = self.twoD_cursor_pos
        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius - self.cursor_radius)
        
    def _test_targ_flash_done(self, ts):
        return ts > self.target_flash_time

    def _test_enter_target(self, ts):
        correct_target_loc = self.choice_targ_loc[self.mc_label]
        cursor_pos = self.twoD_cursor_pos
        d = np.linalg.norm(cursor_pos - correct_target_loc)
        return d <= (self.target_radius - self.cursor_radius)

    def _test_enter_wrong_target(self, ts):
        enter_wrong_targ = False

        cursor_pos = self.twoD_cursor_pos
        for i in range(len(self.choice_targ_loc)):
            if abs(i-self.mc_label)>0:
                target_loc = self.choice_targ_loc[i]
                d = np.linalg.norm(cursor_pos - target_loc)
                if d<= (self.target_radius - self.cursor_radius):
                    enter_wrong_targ = True
                    #print "WRONG TARGET: ", self.choice_targ_loc[int(self.mc_label)], target_loc, self.mc_label, cursor_pos, d, self.target_radius-self.cursor_radius
        return enter_wrong_targ



    ##############################
    ######## Generators ########
    ##############################


    @staticmethod
    def twoD_choice_CO(nblocks=100, boundaries=(-18,18,-12,12), target_distance=6, ntargets=4, mc_target_angle_offset=0):

        theta = []
        label = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            temp = temp + (mc_target_angle_offset*np.pi/180)
            temp2 = np.vstack((temp, np.arange(ntargets) ))
            np.random.shuffle(temp2.T)

            theta = theta + list(temp2[0,:])
            label = label + list(temp2[1,:])
        theta = np.hstack(theta)
        label = np.hstack(label)

        x = target_distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = target_distance*np.sin(theta)

        mc_targ_flash = np.vstack([x, y, z]).T
        mc_targ_flash = mc_targ_flash[:,np.newaxis,:]

        mc_label = np.array(label)
        mc_label = mc_label[:,np.newaxis, np.newaxis]

        temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets) + (mc_target_angle_offset*np.pi/180)
        x = target_distance*np.cos(temp)
        y = np.zeros(len(temp))
        z = target_distance*np.sin(temp)

        choice_targ = np.vstack([x, y, z]).T
        choice_targ = np.tile(choice_targ, (nblocks*ntargets, 1, 1))

        mc_orig = np.zeros((mc_targ_flash.shape))

        it = iter([dict(mc_targ=mc_targ_flash[i,:,:], mc_label=mc_label[i,:,:], choice_targ=choice_targ[i,:,:], mc_orig = mc_orig[i,:,:]) for i in range(len(label))])
        return it


