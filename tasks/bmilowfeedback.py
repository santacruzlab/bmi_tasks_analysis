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

try:
    import pygame
except:
    import warnings
    warnings.warn('tasks/manualcontrolmultitasks.py: not importing name pygame')

import bmimultitasks
from riglib.bmi import clda
import os
import riglib.bmi
import pdb
import multiprocessing as mp
import pickle

class BMIControlManipulatedFB(bmimultitasks.BMIControlMulti):

    feedback_rate = traits.Float(60, desc="Rate in hz that cursor position is updated on screen (best if factor of 60)")
    task_update_rate = traits.Float(60, desc="Rate in hz that decoded cursor position is updated within task (best if factor of 60)")
    ordered_traits = ['session_length', 'assist_level', 'assist_time', 'feedback_rate', 'task_update_rate']

    def __init__(self, *args, **kwargs):
        super(BMIControlManipulatedFB, self).__init__(*args, **kwargs)
        self.visible_cursor = Sphere(radius=self.cursor_radius, color=(1,1,1,1))
        self.add_model(self.visible_cursor)
        self.cursor_visible = True

    def init(self):
        self.dtype.append(('visible_cursor','f8',3))
        super(BMIControlManipulatedFB, self).init()
        
        self.feedback_num = int(60.0/self.feedback_rate)
        self.task_update_num = int(60.0/self.task_update_rate)
        self.loopcount = 0

    def update_cursor(self):
        ''' Update the cursor's location and visibility status.'''
        pt = self.get_cursor_location()
        prev = self.cursor_visible
        self.cursor_visible = False
        if prev != self.cursor_visible:
            self.show_object(self.cursor, show=False) #self.cursor.detach()
            self.requeue()
        #update the "real" cursor location only according to specified task update rate
        if self.loopcount%self.task_update_num==0:
            if pt is not None:
                self.move_cursor(pt)
        #update the visible cursor location only according to specified feedback rate
        if self.loopcount%self.feedback_num==0:
            loc = self.cursor.xfm.move
            self.visible_cursor.translate(*loc,reset=True)

    def _cycle(self):
        ''' Overwriting parent methods since this one works differently'''
        self.update_assist_level()
        self.task_data['assist_level'] = self.current_assist_level
        self.update_cursor()
        self.task_data['cursor'] = self.cursor.xfm.move.copy()
        self.task_data['target'] = self.target_location.copy()
        self.task_data['target_index'] = self.target_index
        self.task_data['visible_cursor'] = self.visible_cursor.xfm.move.copy()
        self.loopcount += 1
        #write to screen
        self.draw_world()



class CLDAManipulatedFB(BMIControlManipulatedFB):
    '''
    BMI task that periodically refits the decoder parameters based on intended
    movements toward the targets. Inherits directly from BMIControl. Can be made
    to automatically linearly decrease assist level over set time period, or
    to provide constant assistance by setting assist_level and assist_min equal.
    '''

    batch_time = traits.Float(80.0, desc='The length of the batch in seconds')
    half_life = traits.Tuple((120., 120.0), desc='Half life of the adaptation in seconds')
    decoder_sequence = traits.String('test', desc='signifier to group together sequences of decoders')
    #assist_min = traits.Float(0, desc="Assist level to end task at")
    #half_life_final = traits.Float(120.0, desc='Half life of the adaptation in seconds')
    half_life_decay_time = traits.Float(900.0, desc="Time to go from initial half life to final")


    def __init__(self, *args, **kwargs):
        super(CLDAManipulatedFB, self).__init__(*args, **kwargs)
        #self.assist_start = self.assist_level
        self.learn_flag = True

    def init(self):
        '''
        Secondary init function. Decoder has already been created by inclusion
        of the 'bmi' feature in the task. Create the 'learner' and 'updater'
        components of the CLDA algorithm
        '''
        # Add CLDA-specific data to save to HDF file 
        self.dtype.append(('half_life', 'f8', (1,)))

        super(CLDAManipulatedFB, self).init()

        self.batch_size = int(self.batch_time/self.decoder.binlen)
        self.create_learner()

        # Create the updater second b/c the update algorithm might need to force
        # a particular batch size for the learner
        self.create_updater()

        # Create the BMI system which combines the decoder, learner, and updater
        self.bmi_system = riglib.bmi.BMISystem(self.decoder, self.learner,
            self.updater)

        

    def create_learner(self):
        self.learner = clda.CursorGoalLearner(self.batch_size)

        # Start "learn flag" at True
        self.learn_flag = True
        homedir = os.getenv('HOME')
        f = open(os.path.join(homedir, 'learn_flag_file'), 'w')
        f.write('1')
        f.close()

    def create_updater(self):
        clda_input_queue = mp.Queue()
        clda_output_queue = mp.Queue()
        half_life_start, half_life_end = self.half_life
        self.updater = clda.KFSmoothbatch(clda_input_queue, clda_output_queue,self.batch_time, half_life_start)

    def update_learn_flag(self):
        # Tell the adaptive BMI when to learn (skip parts of the task where we
        # assume the subject is not trying to move toward the target)
        prev_learn_flag = self.learn_flag

        # Open file to read learn flag
        try:
            homedir = os.getenv('HOME')
            f = open(os.path.join(homedir, 'learn_flag_file'))
            new_learn_flag = bool(int(f.readline().rstrip('\n')))
        except:
            new_learn_flag = True

        if new_learn_flag and not prev_learn_flag:
            print "CLDA enabled"
        elif prev_learn_flag and not new_learn_flag:
            try:
                print "CLDA disabled after %d successful trials" % self.calc_n_rewards()
            except:
                print "CLDA disabled"
        self.learn_flag = new_learn_flag

    def call_decoder(self, spike_counts):
        half_life_start, half_life_end = self.half_life
        current_half_life = self._linear_change(half_life_start, half_life_end, self.half_life_decay_time)
        self.task_data['half_life'] = current_half_life

        # Get the decoder output
        decoder_output, uf =  self.bmi_system(spike_counts, self.target_location,
            self.state, task_data=self.task_data, assist_level=self.current_assist_level,
            target_radius=self.target_radius, speed=self.assist_speed*self.decoder.binlen, 
            learn_flag=self.learn_flag, half_life=current_half_life)
        if uf:
            #send msg to hdf file to indicate decoder update
            self.hdf.sendMsg("update_bmi")
        return decoder_output #self.decoder['hand_px', 'hand_py', 'hand_pz']

    def _cycle(self):
        self.update_learn_flag()
        super(CLDAManipulatedFB, self)._cycle()

    def cleanup(self, database, saveid, **kwargs):
        super(CLDAManipulatedFB, self).cleanup(database, saveid, **kwargs)
        import tempfile, cPickle, traceback, datetime

        # Open a log file in case of error b/c errors not visible to console
        # at this point
        f = open(os.path.join(os.getenv('HOME'), 'Desktop/log'), 'a')
        f.write('Opening log file\n')
        
        # save out the parameter history and new decoder unless task was stopped
        # before 1st update
        try:
            f.write('# of paramter updates: %d\n' % len(self.bmi_system.param_hist))
            if len(self.bmi_system.param_hist) > 0:
                f.write('Starting to save parameter hist\n')
                tf = tempfile.NamedTemporaryFile()
                # Get the update history for C and Q matrices and save them
                #C, Q, m, sd, intended_kin, spike_counts = zip(*self.bmi_system.param_hist)
                #np.savez(tf, C=C, Q=Q, mean=m, std=sd, intended_kin=intended_kin, spike_counts=spike_counts)
                pickle.dump(self.bmi_system.param_hist, tf)
                tf.flush()
                # Add the parameter history file to the database entry for this
                # session
                database.save_data(tf.name, "bmi_params", saveid)
                f.write('Finished saving parameter hist\n')

                # Save the final state of the decoder as a new decoder
                tf2 = tempfile.NamedTemporaryFile(delete=False) 
                cPickle.dump(self.decoder, tf2)
                tf2.flush()
                # create suffix for new decoder that has the sequence and the current day
                # and time. This suffix will be appended to the name of the
                # decoder that we started with and saved as a new decoder.
                now = datetime.datetime.now()
                decoder_name = self.decoder_sequence + '%02d%02d%02d%02d' % (now.month, now.day, now.hour, now.minute)
                database.save_bmi(decoder_name, saveid, tf2.name)
        except:
            traceback.print_exc(file=f)
        f.close()

class CLDAControlPPFContAdaptMFB(CLDAManipulatedFB):
    def create_learner(self):
        self.learner = clda.OFCLearner3DEndptPPF(1, dt=self.decoder.filt.dt)

        # Start "learn flag" at True
        self.learn_flag = True
        homedir = os.getenv('HOME')
        f = open(os.path.join(homedir, 'learn_flag_file'), 'w')
        f.write('1')
        f.close()
        
    def create_updater(self):
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder)
