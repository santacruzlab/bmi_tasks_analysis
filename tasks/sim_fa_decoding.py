from features.simulation_features import SimKFDecoderSup, SimCosineTunedEnc, SimTime, SimFAEnc
from features.hdf_features import SaveHDF
from features.generator_features import Autostart
from riglib.stereo_opengl.window import FakeWindow
from riglib.bmi.state_space_models import StateSpaceEndptVel2D, State, offset_state
from riglib.bmi.feedback_controllers import LQRController
from riglib.bmi.kfdecoder import KalmanFilter, FAKalmanFilter
from riglib import experiment
from riglib.bmi import extractor
from riglib.bmi.assist import FeedbackControllerAssist

from tasks.bmimultitasks import BMIControlMulti, BMIResetting, BMIResettingObstacles
from tasks import manualcontrolmultitasks

import plantlist

import numpy as np
import os, shutil, pickle
import pickle
import time, datetime

from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi.feedback_controllers import LQRController
from riglib.bmi import feedback_controllers

class SuperSimpleEndPtAssister(object):
    '''
    Constant velocity toward the target if the cursor is outside the target. If the
    cursor is inside the target, the speed becomes the distance to the center of the
    target divided by 2.
    '''
    def __init__(self, *args, **kwargs):
        '''    Docstring    '''
        self.decoder_binlen = 0.1
        self.assist_speed = 15.
        self.target_radius = 2.

    def calc_next_state(self, current_state, target_state, mode=None, **kwargs):
        '''    Docstring    '''
        
        cursor_pos = np.array(current_state[0:3,0]).ravel()
        target_pos = np.array(target_state[0:3,0]).ravel()
        decoder_binlen = self.decoder_binlen
        speed = self.assist_speed * decoder_binlen
        target_radius = self.target_radius

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

class EndPtAssisterOFC(LQRController):
    def __init__(self, cost_err_ratio = 10000., include_vel_in_error=True, **kwargs):
        ssm = StateSpaceEndptVel2D()
        A, B, W = ssm.get_ssm_matrices()
        if include_vel_in_error:
            Q = np.mat(np.diag(np.array([1, 1, 1, 1, 1, 1, 0])))
        else:
            Q = np.mat(np.diag(np.array([1, 1, 1, 0, 0, 0, 0])))
        R = cost_err_ratio*np.mat(np.eye(len(ssm.drives_obs_inds)))
        super(EndPtAssisterOFC, self).__init__(A, B, Q, R)

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
        F_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/assist_20levels_ppf.pkl'))
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

#SimCosineTunedEnc
class Sim_FA_BMI(Autostart, FakeWindow, SimTime, BMIResetting):
    sequence_generators = ['centerout_2D_discrete']

    def __init__(self, ssm, *args, **kwargs):
        self.assist_level = kwargs.pop('assist_level',(0., 0.))
        encoder_name = kwargs.pop('encoder_name', None)
        self.encoder = pickle.load(open(encoder_name))
        self.decoder = kwargs.pop('decoder', None)
        if self.decoder is None:
            print "NO DECODER FROM TASK kwargs ! \n \n \n"
        #self.decoder = self.encoder.corresp_dec

        if isinstance(self.decoder.filt, KalmanFilter):
            self.decoder.filt.__class__ = FAKalmanFilter

        fa_dict = kwargs.pop('fa_dict', None)
        self.input_type = kwargs.pop('input_type', 'all')
        self.n_neurons = self.decoder.n_units
        self.decoder.filt.FA_kwargs = fa_dict
        self.decoder.filt.FA_input = self.input_type
        self.fb_ctrl = SuperSimpleEndPtAssister()

        self.ssm = ssm
        self.plant = plantlist.plantlist[self.plant_type]

        super(Sim_FA_BMI, self).__init__(*args, **kwargs)

    def init(self):
        self.add_dtype(self.input_type, 'f8', (self.n_neurons, 1))
        super(Sim_FA_BMI, self).init()

    def _cycle(self):
        try:
            self.task_data[self.input_type] = self.decoder.filt.FA_input_dict[self.input_type+'_input']
        except:
            print 'no data'
        super(Sim_FA_BMI, self)._cycle()

    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.SimBinnedSpikeCountsExtractor(self.fb_ctrl, self.encoder, 
            n_subbins=self.decoder.n_subbins, units=self.decoder.units, task=self)
        self._add_feature_extractor_dtype()

class Sim_FA_Obs_BMI(Sim_FA_BMI, BMIResettingObstacles):
    #sequence_generators = ['centerout_2D_discrete_w_obstacle']
    def __init__(self, ssm, *args, **kwargs):
        super(Sim_FA_Obs_BMI, self).__init__(ssm, *args, **kwargs)
        self.fb_ctrl = SuperSimpleEndPtAssister()

    def create_assister(self):
        self.assister = OFCEndpointAssister() #OFCEndpointAssister

class SimVFB(Autostart, SimTime, FakeWindow, SimKFDecoderSup, SimFAEnc, BMIResetting):
    sequence_generators = ['centerout_2D_discrete']

    def __init__(self, ssm, init_C, *args, **kwargs):
        self.assist_level = kwargs.pop('assist_level',(1, 1))
        self.n_neurons = kwargs.pop('n_neurons', 20)
        #OFC FB CTRL   from BMIControlMulti
        # F = np.eye(7)
        # decoding_rate = 10.
        # B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))
        # B = np.hstack((np.zeros((7, 3)), B, np.zeros((7, 1)) ))
        # self.fb_ctrl = feedback_controllers.LinearFeedbackController(A=B, B=B, F=F)
        if kwargs['fb'] == 'OFC':
            self.fb_ctrl = EndPtAssisterOFC()
        else:
            self.fb_ctrl = SuperSimpleEndPtAssister()

        self.sim_C = init_C

        self.plant = plantlist.plantlist[self.plant_type]
        self.ssm=ssm

        super(SimVFB, self).__init__(*args, **kwargs)


    def _cycle(self, *args, **kwargs):
        super(SimVFB, self)._cycle(*args, **kwargs)

class SimVFB_obs(Autostart, SimTime, FakeWindow, SimKFDecoderSup, SimFAEnc, BMIResettingObstacles):
    
    def __init__(self, ssm, init_C, *args, **kwargs):
        self.assist_level = kwargs.pop('assist_level',(1, 1))
        self.n_neurons = kwargs.pop('n_neurons', 20)
        #OFC FB CTRL   from BMIControlMulti
        # F = np.eye(7)
        # decoding_rate = 10.
        # B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))
        # B = np.hstack((np.zeros((7, 3)), B, np.zeros((7, 1)) ))
        # self.fb_ctrl = feedback_controllers.LinearFeedbackController(A=B, B=B, F=F)
        if kwargs['fb'] == 'OFC':
            self.fb_ctrl = EndPtAssisterOFC()
        else:
            self.fb_ctrl = SuperSimpleEndPtAssister()

        self.sim_C = init_C

        self.plant = plantlist.plantlist[self.plant_type]
        self.ssm=ssm

        super(SimVFB_obs, self).__init__(*args, **kwargs)


    def _cycle(self, *args, **kwargs):
        super(SimVFB_obs, self)._cycle(*args, **kwargs)

    def create_assister(self):
        self.assister = OFCEndpointAssister()

#Playback trajectories
def main_xz_CL(session_length, task_kwargs=None):
    ssm_xz = StateSpaceEndptVel2D()
    Task = experiment.make(Sim_FA_BMI, [SaveHDF])
    targets = manualcontrolmultitasks.ManualControlMulti.centerout_2D_discrete()
    task = Task(ssm_xz, targets, plant_type='cursor_14x14', session_length=session_length, **task_kwargs)
    task.run_sync()
    return task

def main_xz_CL_obstacles(session_length, task_kwargs=None):
    ssm_xz = StateSpaceEndptVel2D()
    Task = experiment.make(Sim_FA_Obs_BMI, [SaveHDF])
    targets = BMIResettingObstacles.centerout_2D_discrete_w_obstacle()
    task = Task(ssm_xz, targets, plant_type='cursor_14x14', session_length=session_length, **task_kwargs)
    task.run_sync()
    return task

#Visual Feedback session
def main_xz(session_length, task_kwargs=None):
    ssm_xz = StateSpaceEndptVel2D()
    Task = experiment.make(SimVFB, [SaveHDF])
    targets = manualcontrolmultitasks.ManualControlMulti.centerout_2D_discrete()
    C = np.random.normal(0, 2, (20, 7))
    task = Task(ssm_xz, C, targets, plant_type='cursor_14x14',
        session_length=session_length, **task_kwargs)

    task.run_sync()
    return task

def main_xz_obs(session_length, task_kwargs=None):
    ssm_xz = StateSpaceEndptVel2D()
    Task = experiment.make(SimVFB_obs, [SaveHDF])
    targets = BMIResettingObstacles.centerout_2D_discrete_w_obstacle()
    C = np.random.normal(0, 2, (20, 7))
    task = Task(ssm_xz, C, targets, plant_type='cursor_14x14',
        session_length=session_length, **task_kwargs)

    task.run_sync()
    return task

def save_stuff(task, suffix='', priv_var=None, shar_var=None):
    enc = task.encoder
    task.decoder.save()
    enc.corresp_dec = task.decoder
    enc.sampled_priv_var = priv_var
    enc.sampled_shar_var = shar_var

    #Save task info
    ct = datetime.datetime.now()
    pnm = os.path.expandvars('$FA_GROM_DATA/sims/FR_val/enc'+ ct.strftime("%m%d%y_%H%M") + suffix + '.pkl')
    pickle.dump(enc, open(pnm,'wb'))

    #Save HDF file
    new_hdf = pnm[:-4]+'.hdf'
    f = open(task.h5file.name)
    f.close()

    #Wait 
    time.sleep(1.)

    #Wait after HDF cleaned up
    task.cleanup_hdf()
    time.sleep(1.)

    #Copy temp file to actual desired location
    shutil.copy(task.h5file.name, new_hdf)
    f = open(new_hdf)
    f.close()

    #Return filename
    return pnm