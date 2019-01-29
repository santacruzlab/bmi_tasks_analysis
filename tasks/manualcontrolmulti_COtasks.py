'''
Base tasks for generic point-to-point reaching
'''

from __future__ import division
import numpy as np
import manualcontrolmultitasks
from riglib.experiment import traits, Sequence
import tasks
import math
import traceback

####### CONSTANTS


sec_per_min = 60.0
RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)
mm_per_cm = 1./10

class ManualControlMulti_plusvar(manualcontrolmultitasks.ManualControlMulti):

    planar_hand = traits.Float(0, desc="0: For Standard Manual Control, 1: Kinarm style Manual Control")
    hold_variance = traits.Float(100, desc = "Variance of hold period for origin hold")

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )

    def __init__(self, *args, **kwargs):
        super(ManualControlMulti_plusvar, self).__init__(*args, **kwargs)
        self.hold_time_pls_var = self.hold_time + np.random.uniform(low=-1,high=1)*self.hold_variance
 
    def _start_hold(self):
        self.hold_time_pls_var = self.hold_time + np.random.uniform(low=-1,high=1)*self.hold_variance
        super(ManualControlMulti_plusvar, self)._start_hold()

    def _test_hold_complete(self, ts):
        return ts >= self.hold_time_pls_var

    def move_effector(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None. Changes configuration
        depending on self.planar_hand (whether to ignore y or z variable)'''
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
                if self.planar_hand==0: #Set y coordinate to 0 for 2D tasks
                    pt[1] = 0
                    pt[1] = pt[1]*2 #From ManualControlMulti
                elif self.planar_hand==1:
                    pt[2] = pt[1].copy()
                    pt[1] = 0
                
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
            #print pt
            self.plant.set_endpoint_pos(pt)





