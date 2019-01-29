'''
Tasks which control a plant under pure machine control. Used typically for initializing BMI decoder parameters.
'''
from __future__ import division

import manualcontrolmultitasks

import numpy as np
import time

from riglib.experiment import traits, experiment

import os
from riglib.bmi import clda, assist, extractor, train, goal_calculators, ppfdecoder
import riglib.bmi
import pdb
import multiprocessing as mp
import pickle
import tables
import re

from riglib.stereo_opengl import ik
import tempfile, cPickle, traceback, datetime

from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter
from bmimultitasks import BMIControlMulti
from riglib.bmi.extractor import DummyExtractor
from riglib.stereo_opengl.window import WindowDispl2D, FakeWindow





class VisualFeedbackMulti(manualcontrolmultitasks.ManualControlMulti):
    '''
    Displays task to subject but cursor moves automatically to targets with some
    noise added. Subject still gets reward when cursor hits targets.
    '''
    background = (.5,.5,.5,1) # Set the screen background color to grey
    noise_level = traits.Float(0.5,desc="Percent noise to add to straight line movements.")
    smoothparam = 10 # number of frames to move in one direction before switching
    smoothcounter = 9
    velnoise = np.array([0,0,0])
    ordered_traits = ['session_length']
    exclude_parent_traits = ['marker_count', 'marker_num']

    def __init__(self, *args, **kwargs):
        self.cursor_visible = True
        super(VisualFeedbackMulti, self).__init__(*args, **kwargs)
        self.prev_cursor = self.plant.get_endpoint_pos()

    def move_effector(self):
        ''' 
        Returns the 3D coordinates of the cursor.
        '''
        # calculate straight line x and z velocities
        targetvec = self.target_location - self.prev_cursor
        vecmag = np.sqrt(targetvec[0]**2 + targetvec[1]**2 + targetvec[2]**2)
        if vecmag < .05:
            velideal = np.array([0,0,0])
        else:
            direction = targetvec/vecmag
            velideal = direction*.1 # constant velocity for now, maybe change to bell shaped curve later??
        if self.smoothcounter == (self.smoothparam-1):
            # create random noise x and z velocities
            self.velnoise = np.array([np.random.randn(1)[0], 0, np.random.randn(1)[0]])*.1
            self.smoothcounter = 0
        else:
            self.smoothcounter += 1
        # combine ideal velocity with noise
        vel = velideal*(1-self.noise_level) + self.velnoise*self.noise_level
        
        # calculate new cursor position
        #self.set_arm_endpoint(self.prev_cursor + vel, time_limit=0.012)
        self.plant.set_endpoint_pos(self.prev_cursor + vel)

    def update_cursor(self):
        ''' Update the cursor's location and visibility status.'''
        pt = self.get_cursor_location()
        self.move_cursor(pt)
        self.prev_cursor = pt.copy()

class VisualFeedbackMulti2DWindow(VisualFeedbackMulti, WindowDispl2D):
    is_bmi_seed = True
    channel_list_name = 'emg14_bip'
    def __init__(self, *args, **kwargs):
        super(VisualFeedbackMulti2DWindow, self).__init__(*args, **kwargs)      


bmi_ssm_options = ['Endpt2D', 'Tentacle', 'Joint2L']
class EndPostureFeedbackController(BMILoop, traits.HasTraits):
    ssm_type_options = bmi_ssm_options
    ssm_type = traits.OptionsList(*bmi_ssm_options, bmi3d_input_options=bmi_ssm_options)

    def load_decoder(self):
        from db.namelist import bmi_state_space_models
        # from config import config
        # with open(os.path.join(config.log_dir, 'EndPostureFeedbackController'), 'w') as fh:
        #     fh.write('%s' % self.ssm_type)
        self.ssm = bmi_state_space_models[self.ssm_type]
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()


class TargetCaptureVisualFeedback(EndPostureFeedbackController, BMIControlMulti):
    assist_level = (1, 1)
    is_bmi_seed = True

    def move_effector(self):
        pass

class TargetCaptureVFB2DWindow(TargetCaptureVisualFeedback, WindowDispl2D):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(TargetCaptureVFB2DWindow, self).__init__(*args, **kwargs)

    def _start_wait(self):
        self.wait_time = 0.
        super(TargetCaptureVFB2DWindow, self)._start_wait()
        
    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause