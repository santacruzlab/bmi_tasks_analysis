'''
Online prosthesis simulator (OPS)
'''
from __future__ import division

import manualcontrolmultitasks

import numpy as np
import time

from riglib.experiment import traits, experiment

import os
from riglib.bmi import clda, assist, extractor, train, goal_calculators, ppfdecoder, kfdecoder
import riglib.bmi
import pdb
import multiprocessing as mp
import pickle
import tables
import re

from riglib.stereo_opengl import ik
import tempfile, cPickle, traceback, datetime

from bmimultitasks import BMIControlMulti, BMILoop

from riglib.bmi import sim_neurons

cm_to_m = 0.01

class JoystickNeuralSim(object):
    feature_type = 'spike_counts'
    def __init__(self, encoder, ctrl_source, gain, n_subbins=1):
        # self.feature_dtype = ('spike_counts', 'u4', (encoder.n_neurons, n_subbins))
        self.feature_dtype = [('spike_counts', 'u4', (encoder.n_neurons, n_subbins)), ('joystick_vel', 'f8', 3)]
        self.ctrl_source = ctrl_source
        self.gain = gain
        self.encoder = encoder

    def __call__(self, start_time):
        pt = self.ctrl_source.get()

        calib = [0.497,0.517] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
        try:
            # Get the last sensor value
            pt = pt[-1][0]

            # Left/right direction switch because the display is mirrored for the stereo rig
            pt[0] = 1-pt[0] 

            joystick_vel = np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
        except:
            joystick_vel = np.zeros(3)

        epsilon = 2*(10**-2) #Define epsilon to stabilize cursor movement
        if sum((joystick_vel)**2) < epsilon:
            joystick_vel = np.zeros(3)

        vel = self.gain * joystick_vel
        spike_counts = self.encoder(vel[[0,2]]).reshape(-1,1)
        bin_edges = None

        # return spike_counts, bin_edges
        return dict(spike_counts=spike_counts, joystick_vel=vel)


class JoystickDrivenCursorOPS(BMIControlMulti):
    joystick_gain = traits.Float(20, desc="Gain factor on joystick velocity")
    def __init__(self, *args, **kwargs):
        super(JoystickDrivenCursorOPS, self).__init__(*args, **kwargs)

    def load_decoder(self):
        '''
        Instantiate the neural encoder and "train" the decoder
        '''
        try:
            self.encoder = pickle.load(open(os.path.expandvars('/storage/task_data/JoystickDrivenCursorOPS/encoder1.pkl')))
            self.decoder = pickle.load(open(os.path.expandvars('/storage/task_data/JoystickDrivenCursorOPS/decoder1.pkl')))
        except:
            print "recreating encoder/decoder"
            from riglib.bmi import sim_neurons
            from riglib.bmi import train
            n_units = 5
            encoder = sim_neurons.CosEnc(n_neurons=n_units, baselines=2, mod_depth=3)

            n_samples = 10000
            vel = np.random.normal(size=(n_samples, 2))*4
            spike_counts = np.zeros([n_units, n_samples])
            for k in range(n_samples):
                spike_counts[:,k] = encoder(vel[k])

            ssm = train.endpt_2D_state_space
            kin = np.zeros([6, n_samples])
            kin[ssm.train_inds, :] = vel.T

            units = np.vstack([np.arange(n_units), np.ones(n_units)]).T
            self.decoder = train.train_KFDecoder(ssm, kin, spike_counts, units)
            self.encoder = encoder

            pickle.dump(self.encoder, open(os.path.expandvars('/storage/task_data/JoystickDrivenCursorOPS/encoder1.pkl'), 'w'))
            pickle.dump(self.decoder, open(os.path.expandvars('/storage/task_data/JoystickDrivenCursorOPS/decoder1.pkl'), 'w'))

    def create_feature_extractor(self):
        self.extractor = JoystickNeuralSim(self.encoder, self.joystick, self.joystick_gain)
        if isinstance(self.extractor.feature_dtype, tuple):
            self.add_dtype(*self.extractor.feature_dtype)
        else:
            for x in self.extractor.feature_dtype:
                self.add_dtype(*x)

class JoystickDrivenCursorOPSBiased(JoystickDrivenCursorOPS):
    bias_angle = traits.Float(0, desc="Angle to bias cursor velocity, in degrees")
    bias_gain = traits.Float(0, desc="Gain of directional velocity bias in cm/sec")

    def load_decoder(self):
        super(JoystickDrivenCursorOPSBiased, self).load_decoder()
        assert isinstance(self.decoder, kfdecoder.KFDecoder)
        self.decoder.filt.A[3,-1] += self.bias_gain * np.cos(self.bias_angle * np.pi/180)
        self.decoder.filt.A[5,-1] += self.bias_gain * np.sin(self.bias_angle * np.pi/180)
        print self.decoder.filt.A