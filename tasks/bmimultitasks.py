'''
BMI tasks in the new structure, i.e. inheriting from manualcontrolmultitasks
'''
from __future__ import division

import manualcontrolmultitasks

import numpy as np
import time, random

from riglib.experiment import traits, experiment
from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife

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

from riglib.bmi.bmi import GaussianStateHMM, Decoder, GaussianState, BMISystem, BMILoop
from riglib.bmi.assist import Assister, SSMLFCAssister, FeedbackControllerAssist
from riglib.bmi import feedback_controllers
from riglib.stereo_opengl.window import WindowDispl2D
from riglib.stereo_opengl.primitives import Line
from manualcontrolfreechoice import target_colors

np.set_printoptions(suppress=False)

###################
####### Assisters
##################
class OFCEndpointAssister(FeedbackControllerAssist):
    '''
    Assister for cursor PPF control which uses linear feedback (infinite horizon LQR) to drive the cursor toward the target state
    '''
    def __init__(self, decoding_rate=180):
        '''
        Constructor for OFCEndpointAssister

        Parameters
        ----------
        decoding_rate : int
            Rate that the decoder should operate, in Hz. Should be a multiple or divisor of 60 Hz

        Returns
        -------
        OFCEndpointAssister instance
        '''
        F_dict = pickle.load(open('/storage/assist_params/assist_20levels_ppf.pkl'))
        B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))
        fb_ctrl = feedback_controllers.MultiModalLFC(A=B, B=B, F_dict=F_dict)
        super(OFCEndpointAssister, self).__init__(fb_ctrl, style='additive_cov')
        self.n_assist_levels = len(F_dict)

    def get_F(self, assist_level):
        '''
        Look up the feedback gain matrix based on the assist_level

        Parameters
        ----------
        assist_level : float
            Float between 0 and 1 to indicate the level of the assist (1 being the highest)

        Returns
        -------
        np.mat
        '''
        assist_level_idx = min(int(assist_level * self.n_assist_levels), self.n_assist_levels-1)
        F = np.mat(self.fb_ctrl.F_dict[assist_level_idx])    
        return F

class TentacleAssist(FeedbackControllerAssist):
    '''
    Assister which can be used for a kinematic chain of any length. The cost function is calibrated for the experiments with the 4-link arm
    '''
    def __init__(self, ssm, kin_chain, update_rate=0.1):
        '''
        Constructor for TentacleAssist

        Parameters
        ----------
        ssm: riglib.bmi.state_space_models.StateSpace instance
            The state-space model's A and B matrices represent the system to be controlled
        args: positional arguments
            These are ignored (none are necessary)
        kwargs: keyword arguments
            The constructor must be supplied with the 'kin_chain' kwarg, which must have the attribute 'link_lengths'
            This is specific to 'KinematicChain' plants.

        Returns
        -------
        TentacleAssist instance

        '''
        A, B, W = ssm.get_ssm_matrices(update_rate=update_rate)
        Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros_like(kin_chain.link_lengths), 0])))
        R = 10000*np.mat(np.eye(B.shape[1]))

        fb_ctrl = LQRController(A, B, Q, R)

        super(TentacleAssist, self).__init__(fb_ctrl, style='additive')

class ObstacleAssist(FeedbackControllerAssist):
    def __init__(self, ssm):
        A, B, W = ssm.get_ssm_matrices(update_rate=0.1)
        Q = np.mat(np.diag([10, 10, 10, 5, 5, 5, 0]))
        R = 10**6*np.mat(np.eye(B.shape[1]))
        fb_ctrl = LQRController(A, B, Q, R)

        super(ObstacleAssist, self).__init__(fb_ctrl, style='additive')

class SimpleEndpointAssister(Assister):
    '''
    Constant velocity toward the target if the cursor is outside the target. If the
    cursor is inside the target, the speed becomes the distance to the center of the
    target divided by 2.
    '''
    def __init__(self, *args, **kwargs):
        '''    Docstring    '''
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''    Docstring    '''
        Bu = None
        assist_weight = 0.

        if assist_level > 0:
            cursor_pos = np.array(current_state[0:3,0]).ravel()
            target_pos = np.array(target_state[0:3,0]).ravel()
            decoder_binlen = self.decoder_binlen
            speed = self.assist_speed * decoder_binlen
            target_radius = self.target_radius
            Bu = self.endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, assist_level)
            assist_weight = assist_level 

        # return Bu, assist_weight
        return dict(x_assist=Bu, assist_level=assist_weight)

    @staticmethod 
    def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_level=0.):
        '''
        Estimate the next state using a constant velocity estimate moving toward the specified target

        Parameters
        ----------
        cursor_pos: np.ndarray of shape (3,)
            Current position of the cursor
        target_pos: np.ndarray of shape (3,)
            Specified target position
        decoder_binlen: float
            Time between iterations of the decoder
        speed: float
            Speed of the machine-assisted cursor
        target_radius: float
            Radius of the target. When the cursor is inside the target, the machine assisted cursor speed decreases.
        assist_level: float
            Scalar between (0, 1) where 1 indicates full machine control and 0 indicates full neural control.

        Returns
        -------
        x_assist : np.ndarray of shape (7, 1)
            Control vector to add onto the state vector to assist control.
        '''
        diff_vec = target_pos - cursor_pos 
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
        
        if dist_to_target > target_radius:
            assist_cursor_pos = cursor_pos + speed*dir_to_target
        else:
            assist_cursor_pos = cursor_pos + speed*diff_vec/2

        assist_cursor_vel = (assist_cursor_pos-cursor_pos)/decoder_binlen
        x_assist = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])
        x_assist = np.mat(x_assist.reshape(-1,1))
        return x_assist

class SimpleEndpointAssisterLFC(feedback_controllers.MultiModalLFC):
    '''
    Docstring
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        dt = 0.1
        A = np.mat([[1., 0, 0, dt, 0, 0, 0], 
                    [0., 1, 0, 0,  dt, 0, 0],
                    [0., 0, 1, 0, 0, dt, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 1]])

        I = np.mat(np.eye(3))
        B = np.vstack([0*I, I, np.zeros([1,3])])
        F_target = np.hstack([I, 0*I, np.zeros([3,1])])
        F_hold = np.hstack([0*I, 0*I, np.zeros([3,1])])
        F_dict = dict(hold=F_hold, target=F_target)
        super(SimpleEndpointAssisterLFC, self).__init__(B=B, F_dict=F_dict)


#################
##### Tasks #####
#################
class BMIControlMulti(BMILoop, LinearlyDecreasingAssist, manualcontrolmultitasks.ManualControlMulti):
    '''
    Target capture task with cursor position controlled by BMI output.
    Cursor movement can be assisted toward target by setting assist_level > 0.
    '''

    background = (.5,.5,.5,1) # Set the screen background color to grey
    reset = traits.Int(0, desc='reset the decoder state to the starting configuration')

    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'reward_time','timeout_time','timeout_penalty_time']
    exclude_parent_traits = ['marker_count', 'marker_num', 'goal_cache_block']

    static_states = [] # states in which the decoder is not run
    hidden_traits = ['arm_hide_rate', 'arm_visible', 'hold_penalty_time', 'rand_start', 'reset', 'target_radius', 'window_size']

    is_bmi_seed = False

    cursor_color_adjust = traits.OptionsList(*target_colors.keys(), bmi3d_input_options=target_colors.keys())

    def __init__(self, *args, **kwargs):     
        super(BMIControlMulti, self).__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        sph = self.plant.graphics_models[0]
        sph.color = target_colors[self.cursor_color_adjust]
        sph.radius = self.cursor_radius
        self.plant.cursor_radius = self.cursor_radius   
        self.plant.cursor.radius = self.cursor_radius
        super(BMIControlMulti, self).init(*args, **kwargs)


    def move_effector(self, *args, **kwargs):
        pass

    def create_assister(self):
        # Create the appropriate type of assister object
        start_level, end_level = self.assist_level
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed

        from db import namelist

        if self.decoder.ssm == namelist.endpt_2D_state_space and isinstance(self.decoder, ppfdecoder.PPFDecoder):
            self.assister = OFCEndpointAssister()
        elif self.decoder.ssm == namelist.endpt_2D_state_space:
            self.assister = SimpleEndpointAssister(**kwargs)
        elif (self.decoder.ssm == namelist.tentacle_2D_state_space) or (self.decoder.ssm == namelist.joint_2D_state_space):
            # kin_chain = self.plant.kin_chain
            # A, B, W = self.decoder.ssm.get_ssm_matrices(update_rate=self.decoder.binlen)
            # Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros_like(kin_chain.link_lengths), 0])))
            # R = 10000*np.mat(np.eye(B.shape[1]))

            # fb_ctrl = LQRController(A, B, Q, R)
            # self.assister = FeedbackControllerAssist(fb_ctrl, style='additive')
            self.assister = TentacleAssist(ssm=self.decoder.ssm, kin_chain=self.plant.kin_chain, update_rate=self.decoder.binlen)
        else:
            raise NotImplementedError("Cannot assist for this type of statespace: %r" % self.decoder.ssm)        
        
        print self.assister

    def create_goal_calculator(self):
        from db import namelist
        if self.decoder.ssm == namelist.endpt_2D_state_space:
            self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)
        elif self.decoder.ssm == namelist.joint_2D_state_space:
            self.goal_calculator = goal_calculators.PlanarMultiLinkJointGoal(self.decoder.ssm, self.plant.base_loc, self.plant.kin_chain, multiproc=False, init_resp=None)
        elif self.decoder.ssm == namelist.tentacle_2D_state_space:
            shoulder_anchor = self.plant.base_loc
            chain = self.plant.kin_chain
            q_start = self.plant.get_intrinsic_coordinates()
            x_init = np.hstack([q_start, np.zeros_like(q_start), 1])
            x_init = np.mat(x_init).reshape(-1, 1)

            cached = True

            if cached:
                goal_calc_class = goal_calculators.PlanarMultiLinkJointGoalCached
                multiproc = False
            else:
                goal_calc_class = goal_calculators.PlanarMultiLinkJointGoal
                multiproc = True

            self.goal_calculator = goal_calc_class(namelist.tentacle_2D_state_space, shoulder_anchor, 
                                                   chain, multiproc=multiproc, init_resp=x_init)
        else:
            raise ValueError("Unrecognized decoder state space!")

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoalCached):
            task_eps = np.inf
        else:
            task_eps = 0.5
        ik_eps = task_eps/10
        data, solution_updated = self.goal_calculator(self.target_location, verbose=False, n_particles=500, eps=ik_eps, n_iter=10, q_start=self.plant.get_intrinsic_coordinates())
        target_state, error = data

        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoal) and error > task_eps and solution_updated:
            self.goal_calculator.reset()

        return np.array(target_state).reshape(-1,1)

    def _end_timeout_penalty(self):
        if self.reset:
            self.decoder.filt.state.mean = self.init_decoder_mean
            self.hdf.sendMsg("reset")

    def move_effector(self):
        pass

    # def _test_enter_target(self, ts):
    #     '''
    #     return true if the distance between center of cursor and target is smaller than the cursor radius
    #     '''
    #     cursor_pos = self.plant.get_endpoint_pos()
    #     d = np.linalg.norm(cursor_pos - self.target_location)
    #     return d <= self.target_radius

class BMIControlMulti2DWindow(BMIControlMulti, WindowDispl2D):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(BMIControlMulti2DWindow, self).__init__(*args, **kwargs)
        self.braimamp_channels = ['InterFirst', 'AbdPolLo', 'ExtCU',
            'ExtCarp',
            'ExtDig',
            'FlexDig',
            'FlexCarp',
            'PronTer',
            'Biceps',
            'Triceps',
            'FrontDelt',
            'MidDelt',
            'TeresMajor',
            'PectMajor',
        ]
    
    def create_assister(self):
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed    
        self.assister = SimpleEndpointAssister(**kwargs)
    
    def create_goal_calculator(self):
        self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)

    def _start_wait(self):
        self.wait_time = 0.
        super(BMIControlMulti2DWindow, self)._start_wait()
        
    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

class BMIControlEMGBiofeedback(BMIControlMulti2DWindow):
    
    from ismore.invasive import emg_decoder
    from ismore import brainamp_channel_lists

    fps = 20.
    emg_decoder = traits.InstanceFromDB(emg_decoder.EMGBioFeedback, bmi3d_db_model="Decoder", 
        bmi3d_query_kwargs=dict(name__startswith='emgbiofeedback'))
    channels = brainamp_channel_lists.emg14_bip_filt
    brainamp_channels = brainamp_channel_lists.emg14_bip
    sequence_generators = ['onedim_up_down1']

    def init(self):
        from utils.ringbuffer import RingBuffer
        from ismore.noninvasive.emg_feature_extraction import EMGMultiFeatureExtractor
        
        self.emg_decoder_extractor = EMGMultiFeatureExtractor(None, 
            emg_channels = self.channels, 
            feature_names = self.emg_decoder.extractor_kwargs['feature_names'], 
            win_len=self.emg_decoder.extractor_kwargs['win_len'], 
            fs=self.emg_decoder.extractor_kwargs['fs'])
        
        self.chanix = self.emg_decoder.extractor_kwargs['subset_muscles_ix']
        self.recent_features = np.zeros((len(self.channels), ))
        self.zscored_fts = np.zeros((len(self.channels), ))

        self.add_dtype('emg_fts', 'f8', (len(self.channels), ))
        self.add_dtype('zsc_emg_fts', 'f8', (len(self.channels), ))

        super(BMIControlEMGBiofeedback, self).init()  
        print ' n features: ', self.emg_decoder_extractor.n_features

        self.features_buffer = RingBuffer(
            item_len=self.emg_decoder_extractor.n_features,
            capacity=0.2*self.fps)

        self.features_buffer2 = RingBuffer(
            item_len=self.emg_decoder_extractor.n_features,
            capacity=60*self.fps)

        for i in range(int(0.2*self.fps)):
            self.features_buffer.add(np.zeros((14, )))
            self.features_buffer2.add(np.zeros((14, )))
        self.emg_decoder_extractor.source = self.brainamp_source

        # Draw lines: 
        self.lines = {}
        for i, j in enumerate(self.chanix):
            if 'Ext' in self.emg_decoder.extractor_kwargs['subset_muscles'][i]:
                initpos = [12, 0]
            elif 'Flex' in self.emg_decoder.extractor_kwargs['subset_muscles'][i]:
                initpos = [-13, 0]
            self.lines[i] = Line(initpos, 1.5, 1., 0., np.array([0., 0., 1.]), True)
        for i, (key, val) in enumerate(self.lines.items()):
            self.add_model(val)

    def move_plant(self):
        emg_decoder_features = self.emg_decoder_extractor() # emg_features is of type 'dict'
        
        self.features_buffer.add(emg_decoder_features[self.emg_decoder_extractor.feature_type])
        self.features_buffer2.add(emg_decoder_features[self.emg_decoder_extractor.feature_type])

        self.recent_features = self.features_buffer.get_all()
        long_features = self.features_buffer2.get_all()
        
        mean_fts = np.nanmean(long_features, axis=1)[:, np.newaxis]
        std_fts = np.nanstd(long_features, axis=1)[:, np.newaxis]
        std_fts[std_fts==0] = 1

        try:
            self.zscored_fts = np.sum((self.recent_features - mean_fts) / std_fts , axis=1)
        except:
            self.zscored_fts = np.sum((self.recent_features[:, np.newaxis] - mean_fts) / std_fts , axis=1)

        self.cursor = np.array([self.emg_decoder(self.zscored_fts[self.chanix]), 0, 0])
        self.plant.set_endpoint_pos(self.cursor)

        for i, j in enumerate(self.chanix):
            self.lines[i].width = self.zscored_fts[j] + 4 # make negative 

    def _cycle(self):
        #self.task_data['emg_fts'] = np.mean(self.recent_features, axis=1)[:, np.newaxis]
        self.task_data['zsc_emg_fts'] = self.zscored_fts
        super(BMIControlEMGBiofeedback, self)._cycle()

    @staticmethod
    def onedim_up_down1(length=100, rad=8):
        pairs = np.zeros([length, 1, 3])
        pairs[np.arange(0, length, 2), :, 0] = -1*rad
        pairs[np.arange(0, length-2, 2)+1, :, 0] = rad
        return pairs

class BMIControlTargetJump(BMIControlMulti):
    '''
    Version of the BMIControlMulti task where the targets can jump mid-trial
    '''
    sequence_generators = ['center_out_widescreen_jump']    
    is_bmi_seed = False
    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None, jump="target"),  # 'jump' event not present in original ManualControlMulti task
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    target_jump_time = traits.Float(0.5, desc="Time from go cue to target jump")

    def _test_jump(self, ts):
        jump = self.is_jump_trial and (ts > self.target_jump_time) and (self.target_index == 1)
        target = self.targets[self.target_index % 2]
        if jump:
            target.hide()
        return jump

    def _parse_next_trial(self):
        self.targs, self.is_jump_trial = self.next_trial

    @staticmethod
    def center_out_widescreen_jump(nblocks=100, inner_dist=8, jump_rate=0.1):
        pi = np.pi
        target_angles = [-7*pi/8, -6*pi/8, -pi/4, -pi/8, 0, pi/8, pi/4, 3*pi/4, 7*pi/8, pi]
        target_positions = np.vstack([np.cos(target_angles), np.zeros_like(target_angles), np.sin(target_angles)]).T * inner_dist
        center = np.zeros(3)

        targ_sequences = [np.vstack([center, targ_pos]) for targ_pos in target_positions]

        from riglib.experiment.generate import block_random
        block_random_targ_sequences = block_random(targ_sequences, nblocks=nblocks)
        n_jumps_per_block = jump_rate / (1./len(target_angles))

        nopts = len(target_angles)
        opt_inds = np.arange(nopts)
        opt_inds_shuf = opt_inds.copy()
        if int(n_jumps_per_block) == n_jumps_per_block: # jump rate is a multiple of the number of target options
            data = []
            for k in range(nblocks):
                block = block_random_targ_sequences[k*nopts:(k+1)*nopts]
                # randomly pick which trials in the block will be jump trials
                np.random.shuffle(opt_inds_shuf)
                jump_inds = opt_inds_shuf[:n_jumps_per_block]

                block_with_jump = []
                for m, targs in enumerate(block):
                    targs = targs[0]
                    if m in jump_inds:
                        targs = np.vstack([targs, 2*targs[-1]])
                    block_with_jump.append((targs, (m in jump_inds)))

                data += block_with_jump #[(targs[0], (m in jump_inds)) for m, targs in enumerate(block)]
            return data
        else:
            raise NotImplementedError

class BMIResetting(BMIControlMulti):
    '''
    Task where the virtual plant starts in configuration sampled from a discrete set and resets every trial
    '''
    status = dict(
        wait = dict(start_trial="premove", stop=None),
        premove=dict(premove_complete="target"),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", trial_restart="premove"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )

    plant_visible = 1
    plant_hide_rate = -1
    premove_time = traits.Float(.1, desc='Time before subject must start doing BMI control')
    # static_states = ['premove'] # states in which the decoder is not run
    add_noise = 0.35
    sequence_generators = BMIControlMulti.sequence_generators + ['outcenter_half_hidden', 'short_long_centerout']

    # def __init__(self, *args, **kwargs):
    #     super(BMIResetting, self).__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        #self.add_dtype('bmi_P', 'f8', (self.decoder.ssm.n_states, self.decoder.ssm.n_states))
        super(BMIResetting, self).init(*args, **kwargs)

    # def move_plant(self, *args, **kwargs):
    #     super(BMIResetting, self).move_plant(*args, **kwargs)
    #     c = self.plant.get_endpoint_pos()
    #     self.plant.set_endpoint_pos(c + self.add_noise*np.array([np.random.rand()-0.5, 0., np.random.rand()-0.5]))

    def _cycle(self, *args, **kwargs):
        #self.task_data['bmi_P'] = self.decoder.filt.state.cov 
        super(BMIResetting, self)._cycle(*args, **kwargs)

    def _while_premove(self):
        self.plant.set_endpoint_pos(self.targs[0])
        self.decoder['q'] = self.plant.get_intrinsic_coordinates()
        # self.decoder.filt.state.mean = self.calc_perturbed_ik(self.targs[0])

    def _start_premove(self):

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[(self.target_index+1) % 2]
        target.move_to_position(self.targs[self.target_index+1])
        target.cue_trial_start()
        
    def _end_timeout_penalty(self):
        pass

    def _test_premove_complete(self, ts):
        return ts>=self.premove_time

    def _parse_next_trial(self):
        try:
            self.targs, self.plant_visible = self.next_trial        
        except:
            self.targs = self.next_trial

    def _test_hold_complete(self,ts):
        ## Disable origin holds for this task
        if self.target_index == 0:
            return True
        else:
            return ts>=self.hold_time

    def _test_trial_incomplete(self, ts):
        return (self.target_index<self.chain_length-1) and (self.target_index != -1) and (self.tries<self.max_attempts)

    def _test_trial_restart(self, ts):
        return (self.target_index==-1) and (self.tries<self.max_attempts)

    @staticmethod
    def outcenter_half_hidden(nblocks=100, ntargets=4, distance=8, startangle=45):
        startangle = np.deg2rad(startangle)
        target_angles = np.arange(startangle, startangle+2*np.pi, 2*np.pi/ntargets)
        origins = distance * np.vstack([np.cos(target_angles), 
                                        np.zeros_like(target_angles),
                                        np.sin(target_angles)]).T
        terminus = np.zeros(3)
        trial_target_sequences = [np.vstack([origin, terminus]) for origin in origins]
        visibility = [True, False]
        from riglib.experiment.generate import block_random
        seq = block_random(trial_target_sequences, visibility, nblocks=nblocks)
        return seq
    
    @staticmethod
    def short_long_centerout(nblocks=100, ntargets=4, distance2=(8, 12)):
        theta = []
        dist = []
        for i in range(nblocks):
            for j in range(2):
                if j==0:
                    temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
                    tempdist = np.zeros((ntargets, )) + distance2[j]
                else:
                    temp = np.hstack((temp, np.arange(0, 2*np.pi, 2*np.pi/ntargets)))
                    tempdist = np.hstack((tempdist, np.zeros((ntargets, ))+distance2[j]))
            
            ix = np.random.permutation(ntargets*2)
            theta = theta + [temp[ix]]
            dist = dist + list(tempdist[ix])
        theta = np.hstack(theta)
        distance = np.hstack(dist)
        
        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs

class BMIResettingObstacles(BMIResetting):
    
    status = dict(
        wait = dict(start_trial="premove", stop=None),
        premove=dict(premove_complete="target"),
        target = dict(enter_target="hold", timeout="timeout_penalty", enter_obstacle="obstacle_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", trial_restart="premove"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        obstacle_penalty = dict(obstacle_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )

    sequence_generators = ['centerout_2D_discrete_w_obstacle', 'centerout_2D_discrete']
    obstacle_sizes = traits.Tuple((2, 3),desc='must match generator sizes!')
    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super(BMIResettingObstacles, self).__init__(*args, **kwargs)
        self.add_obstacles()

    def add_obstacles(self):
        import target_graphics
        #Add obstacle
        self.obstacle_list=[]
        self.obstacle_dict= {}
        for i in self.obstacle_sizes:
            obstacle = target_graphics.VirtualRectangularTarget(target_width=i, target_height=i, target_color=(0, 0, 1, .5), starting_pos=np.zeros(3))
            self.obstacle_list.append(obstacle)
            self.obstacle_dict[i] = len(self.obstacle_list) - 1
            for model in obstacle.graphics_models:
                self.add_model(model)

    def init(self, *args, **kwargs):
        self.add_dtype('obstacle_size', 'f8', (1,))
        self.add_dtype('obstacle_location', 'f8', (3,))
        super(BMIResettingObstacles, self).init(*args, **kwargs)

    def create_goal_calculator(self):
        self.goal_calculator = goal_calculators.Obs_Goal_Calc(self.decoder.ssm)

    def create_assister(self, *args, **kwargs):
        self.assister = OFCEndpointAssister()
        #self.assister = ObstacleAssist(self.decoder.ssm)

    def _start_wait(self):
        for obs in self.obstacle_list:
            obs.hide()
        super(BMIResetting, self)._start_wait()

    def _start_premove(self):
        super(BMIResettingObstacles, self)._start_premove()

    def _start_target(self):
        print 'start target BMIRes'
        self.goal_calculator.clear()
        super(BMIResettingObstacles, self)._start_target()

    # def _start_target(self):
    #     super(BMIResettingObstacles, self)._start_target()

    def _parse_next_trial(self):
        self.targs = self.next_trial[0]
        #Width and height of obstacle
        self.obstacle_size = self.next_trial[1]
        self.obstacle = self.obstacle_list[self.obstacle_dict[self.obstacle_size]]
        self.obstacle_location = self.next_trial[2]


    def _start_target(self):
        super(BMIResettingObstacles, self)._start_target()
        self.obstacle.move_to_position(self.obstacle_location)
        self.obstacle.cube.color = (0., 0., 1., .5)
        self.obstacle.show()
        # print 'targ loc: ', self.target_location.astype(int)
        # print 'obstacle_location: ', self.obstacle_location.astype(int)
        # print 'self.targs: ', self.targs.astype(int)

    def _test_enter_obstacle(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        centered_cursor_pos = np.abs(cursor_pos - self.obstacle_location)
        return np.all(centered_cursor_pos < self.obstacle_size/2.)

    def _test_obstacle_penalty_end(self, ts):
        self.obstacle.cube.color = (1., 1., 0., .5)
        return ts >= self.timeout_penalty_time

    def _start_obstacle_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _cycle(self):
        self.add_obstacle_data()
        super(BMIResettingObstacles, self)._cycle()

    def add_obstacle_data(self):
        self.task_data['obstacle_size'] = self.obstacle_size
        self.task_data['obstacle_location'] = self.obstacle_location



    @staticmethod
    def centerout_2D_discrete_w_obstacle(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10, obstacle_sizes=(2, 3)):
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
        obstacle_sizes: tuple of varying sizes

        Returns
        -------
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp_master = []
            for o in obstacle_sizes:
                angs = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
                obs_sz = [o]*ntargets
                temp = np.vstack((angs, obs_sz)).T
                np.random.shuffle(temp)
                temp_master.append(temp)
            x = np.vstack((temp_master))
            np.random.shuffle(x)
            theta = theta + [x]
        theta = np.vstack(theta)


        x = distance*np.cos(theta[:,0])
        y = np.zeros(len(theta[:,0]))
        z = distance*np.sin(theta[:,0])

        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T

        obstacle_location = (pairs[:, 1, :] - pairs[:, 0, :])*0.5

        return zip(pairs, theta[:,1], obstacle_location)

class BMIResettingObstacles2D(BMIResettingObstacles, WindowDispl2D):
    fps = 20.
    def __init__(self, *args, **kwargs):
        super(BMIResettingObstacles2D, self).__init__(*args, **kwargs)

    def _start_wait(self):
        self.wait_time = 0.
        super(BMIResettingObstacles2D, self)._start_wait()
        
    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

class BMIMultiObstacles(BMIResettingObstacles, manualcontrolmultitasks.ManualControlMulti):
    sequence_generators = ['freeform_2D_discrete_w_2_obstacle']
    obstacle_size = traits.Float(2.)
    
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
        self.i = 0

    def init(self, *args, **kwargs):
        self.add_dtype('obstacle_size', 'f8', (1,))
        self.add_dtype('obstacle_location', 'f8', (5,3))
        super(BMIResettingObstacles, self).init(*args, **kwargs)

    def _parse_next_trial(self):
        self.targs = self.next_trial[0]
        #Width and height of obstacle

        self.trial_obstacle_list = []
        self.trial_obstacle_loc = []

        for io, o in enumerate(self.next_trial[1]):
            self.trial_obstacle_list.append(self.obstacle_list[io])
            self.trial_obstacle_loc.append(o)
        for j in np.arange(io+1, 5):
            o = self.obstacle_list[j]
            o.move_to_position(np.array([-100., 0., -100.]))

    def _start_target(self):
        super(BMIResettingObstacles, self)._start_target()
        for io, o in enumerate(self.trial_obstacle_list):
            o.move_to_position(self.trial_obstacle_loc[io])
            o.cube.color = (0., 0., 1., .5)
            o.show()

    def _test_enter_obstacle(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        in_obs = False
        for io, o in enumerate(self.trial_obstacle_loc):
            centered_cursor_pos = np.abs(cursor_pos - o)

            if np.all(centered_cursor_pos < self.obstacle_size/2.):
                #print 'in'
                in_obs = True
                self.obs_entered_ix = io
        return in_obs

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
        self.i += 1
        if not np.mod(self.i, 60):
            print self.chain_length, self.target_index, self.tries, self.max_attempts
        super(BMIMultiObstacles, self)._cycle()

    def _test_obstacle_penalty_end(self, ts):
        o = self.trial_obstacle_list[self.obs_entered_ix]
        o.cube.color = (1., 1., 0., .5)
        return ts >= self.timeout_penalty_time


    @staticmethod
    def freeform_2D_discrete_w_2_obstacle(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        '''
        Generates a sequence of 2D (x and z) target pairs with the first target anywhere
        always at the origin.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between the targets in a pair.
        obstacle_sizes: tuple of varying sizes

        Returns
        -------
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''
        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        angs = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
        theta1 = []
        theta2 = []
        I2  = []
        I1 = []
        for i in range(nblocks):
            temp_master = []
            
            #target 1: 
            ix1 = np.arange(ntargets)
            np.random.shuffle(ix1)

            #Target 2: 
            ix2 = []
            for t in ix1:
                ix2.append(np.mod(random.choice(np.arange(t+2, t+6)), ntargets))

            theta1.append(angs[ix1])
            theta2.append(angs[ix2])
            I1.append(ix1)
            I2.append(ix2)
        I1 = np.vstack((I1))
        I1 = I1.reshape(-1)

        I2 = np.vstack((I2))
        I2 = I2.reshape(-1)

        theta1 = np.hstack((theta1))
        theta2 = np.hstack((theta2))

        pairs = np.zeros([len(theta1), 2, 3])

        for i, t in enumerate([theta1, theta2]):
            x = distance*np.cos(t)
            y = np.zeros(len(t))
            z = distance*np.sin(t)

            pairs[:,i,:] = np.vstack((x, y, z)).T

        obstacle_locations_list = []

        for i, (i1, i2) in enumerate(zip(I1, I2)):
            obstacle_locations = []

            if np.mod(i1 - i2, ntargets) <= 4: 
                d = np.mod(i1-i2, ntargets)
            elif np.mod(i2 - i1, ntargets) <= 4:
                d = np.mod(i2 - i1, ntargets)

            if d == 2:
                obstacle_location1 = 0.5*(pairs[i,1,:] - pairs[i,0,:]) + pairs[i,0,:]
                obstacle_locations.append(obstacle_location1)
                obstacle_locations.append(np.array([-1*obstacle_location1[0], 0., obstacle_location1[2]]))
                obstacle_locations.append(np.array([-1*obstacle_location1[0], 0., -1*obstacle_location1[2]]))
                obstacle_locations.append(np.array([obstacle_location1[0], 0., -1*obstacle_location1[2]]))
                obstacle_locations.append(np.array([0., 0., 0.]))

            elif d == 3:
                obstacle_locations.append(0.5*(pairs[i,1,:] - pairs[i,0,:]) + pairs[i,0,:])
                if np.mod(i1+3, ntargets) == i2:
                    # CCW
                    ix = np.nonzero(I2==np.mod(i1-3, ntargets))[0]

                elif np.mod(i1-3, ntargets) == i2:
                    # CW
                    ix = np.nonzero(I2==np.mod(i1+3, ntargets))[0]

                obstacle_locations.append(0.5*(pairs[ix[0], 1, :]-pairs[i, 0, :]) + pairs[i,0,:])
                obstacle_locations.append(0.5*(pairs[ix[0], 1, :]-pairs[i, 1, :]) + pairs[i,1,:])


            elif d == 4:
                obstacle_locations.append(np.array([0., 0., 0.]))
                vect = (pairs[i, 1, :] - pairs[i, 0, :]) / np.linalg.norm(pairs[i, 1, :] - pairs[i, 0, :])

                #Rotate 90 degrees: 
                vect_perp = np.array([-1*vect[2], 0., vect[0]])
                obstacle_locations.append(0.5*distance*vect_perp)
                obstacle_locations.append(-0.5*distance*vect_perp)

            else: 
                print 'error: wrong allocation of target pairs'

            obstacle_locations_list.append(np.vstack((obstacle_locations)))

        return zip(pairs, obstacle_locations_list)


class BaselineControl(BMIControlMulti):
    background = (0.0, 0.0, 0.0, 1) # Set background to black to make it appear to subject like the task is not running

    def show_object(self, obj, show=False):
        '''
        Show or hide an object
        '''
        obj.detach()

    def init(self, *args, **kwargs):
        super(BaselineControl, self).init(*args, **kwargs)

    def _cycle(self, *args, **kwargs):
        for model in self.plant.graphics_models:
            model.detach()
        super(BaselineControl, self)._cycle(*args, **kwargs)

    def _start_wait(self, *args, **kwargs):
        for model in self.plant.graphics_models:
            model.detach()
        super(BaselineControl, self)._start_wait(*args, **kwargs)


#########################
######## Simulation tasks
#########################
from features.simulation_features import SimKalmanEnc, SimKFDecoderSup, SimCosineTunedEnc
from riglib.bmi.feedback_controllers import LQRController
class SimBMIControlMulti(SimCosineTunedEnc, SimKFDecoderSup, BMIControlMulti):
    win_res = (250, 140)
    sequence_generators = ['sim_target_seq_generator_multi']
    def __init__(self, *args, **kwargs):
        from riglib.bmi.state_space_models import StateSpaceEndptVel2D
        ssm = StateSpaceEndptVel2D()

        A, B, W = ssm.get_ssm_matrices()
        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = 10000*np.mat(np.diag([1., 1., 1.]))
        self.fb_ctrl = LQRController(A, B, Q, R)

        self.ssm = ssm

        super(SimBMIControlMulti, self).__init__(*args, **kwargs)

    @staticmethod
    def sim_target_seq_generator_multi(n_targs, n_trials):
        '''
        Simulated generator for simulations of the BMIControlMulti and CLDAControlMulti tasks
        '''
        center = np.zeros(2)
        pi = np.pi
        targets = 8*np.vstack([[np.cos(pi/4*k), np.sin(pi/4*k)] for k in range(8)])

        target_inds = np.random.randint(0, n_targs, n_trials)
        target_inds[0:n_targs] = np.arange(min(n_targs, n_trials))
        for k in range(n_trials):
            targ = targets[target_inds[k], :]
            yield np.array([[center[0], 0, center[1]],
                            [targ[0], 0, targ[1]]])        
