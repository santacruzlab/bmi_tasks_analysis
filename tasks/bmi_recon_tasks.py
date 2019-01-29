import numpy as np
from features.simulation_features import SimHDF
from itertools import izip
import bmimultitasks, cursor_clda_tasks
from riglib.bmi import extractor, clda, feedback_controllers, goal_calculators
import os
from riglib.experiment import traits
from db import dbfunctions as dbfn
from riglib.bmi.bmi import BMILoop

from riglib.experiment import Experiment

class BMIReconstruction(BMILoop, Experiment):
    fps = 60
    def __init__(self, te, n_iter, *args, **kwargs):
        self.te = te
        self.n_iter = min(n_iter, len(te.hdf.root.task))

        try:
            self.starting_pos = te.hdf.root.task[0]['decoder_state'][0:3,0]
        except:
            # The statement above appears to not always work...
            self.starting_pos = te.hdf.root.task[0]['cursor'] # #(0, 0, 0)

        if 'plant_type' in te.params:
            self.plant_type = te.params['plant_type']
        elif 'arm_class' in te.params:
            plant_type = te.params['arm_class']
            if plant_type == 'CursorPlant':
                self.plant_type = 'cursor_14x14'            
            else:
                self.plant_type = plant_type
        else:
            self.plant_type = 'cursor_14x14'


        # self.arm_class = te.params['arm_class'] if 'arm_class' in te.params else 'cursor_14x14' 
        # print "arm class", self.arm_class

        ## Set the target radius because the old assist method changes the assist speed
        # when the cursor is inside the target
        self.target_radius = te.target_radius
        self.cursor_radius = te.cursor_radius
        self.assist_level = tuple(te.assist_level)

        self.idx = 0
        super(BMIReconstruction, self).__init__(*args, **kwargs)
        
        self.hdf = SimHDF()
        self.learn_flag = True

        task_msgs = te.hdf.root.task_msgs[:]
        self.update_bmi_msgs = task_msgs[task_msgs['msg'] == 'update_bmi']
        task_msgs = filter(lambda x: x['msg'] not in ['update_bmi'], task_msgs)
        # print task_msgs
        self.task_state = np.array([None]*n_iter)
        for msg, next_msg in izip(task_msgs[:-1], task_msgs[1:]):
            self.task_state[msg['time']:next_msg['time']] = msg['msg']

        self.update_bmi_inds = np.zeros(len(te.hdf.root.task))
        self.update_bmi_inds[self.update_bmi_msgs['time']] = 1
        self.recon_update_bmi_inds = np.zeros(len(te.hdf.root.task))

        self.target_hold_msgs = filter(lambda x: x['msg'] in ['target', 'hold'], te.hdf.root.task_msgs[:])
        self.te = te

    def init_decoder_state(self):
        '''
        Initialize the state of the decoder to match the initial state of the plant
        '''
        init_decoder_state = self.te.hdf.root.task[0]['decoder_state']
        if init_decoder_state.shape[1] > 1:
            init_decoder_state = init_decoder_state[:,0].reshape(-1,1)

        self.init_decoder_mean = init_decoder_state
        self.decoder.filt.state.mean = self.init_decoder_mean

        self.decoder.set_call_rate(self.fps)

    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.ReplaySpikeCountsExtractor(self.te.hdf.root.task, 
            source='spike_counts', units=self.decoder.units, cycle_rate=self.fps)
        self._add_feature_extractor_dtype()

    def load_decoder(self):
        '''
        Create the object for the initial decoder
        '''
        self.decoder = self.te.decoder
        self.n_subbins = self.decoder.n_subbins
        self.decoder_state = np.zeros([self.n_iter, self.decoder.n_states, self.n_subbins])

    def get_spike_counts(self):
        return self.te.hdf.root.task[self.idx]['spike_counts']

    def _update_target_loc(self):
        self.target_location = self.te.hdf.root.task[self.idx]['target']
        self.state = self.task_state[self.idx]

    def get_cursor_location(self, verbose=False):
        if self.idx % 1000 == 0 and verbose: print self.idx

        self.current_assist_level = self.te.hdf.root.task[self.idx]['assist_level'][0]
        try:
            self.current_half_life = self.te.hdf.root.task[self.idx]['half_life'][0]
        except:
            self.current_half_life = 0
        self._update_target_loc()
        
        self.call_decoder_output = self.move_plant(half_life=self.current_half_life)
        if verbose:
            print self.call_decoder_output
            print self.te.hdf.root.task[self.idx]['decoder_state']
            print
        self.decoder_state[self.idx] = self.call_decoder_output
        self.idx += 1

    def call_decoder(self, neural_obs, target_state, **kwargs):
        '''
        Run the decoder computations

        Parameters
        ----------
        neural_obs : np.array of shape (n_features, n_subbins)
            n_features is the number of neural features the decoder is expecting to decode from.
            n_subbins is the number of simultaneous observations which will be decoded (typically 1)
        target_state: np.array of shape (n_states, 1)
            The current optimal state to be in to accomplish the task. In this function call, this gets
            used when adapting the decoder using CLDA
        '''
        # Get the decoder output
        decoder_output, update_flag = self.bmi_system(neural_obs, target_state, self.state, learn_flag=self.learn_flag, **kwargs)
        if update_flag:
            # send msg to hdf file to indicate decoder update
            self.hdf.sendMsg("update_bmi")
            self.recon_update_bmi_inds[self.idx] = 1
            
        return decoder_output

    def get_time(self):
        t = self.idx * 1./self.fps
        return t

    def calc_recon_error(self, n_iter_betw_fb=100000, **kwargs):
        saved_state = self.te.hdf.root.task[:]['decoder_state']
        while self.idx < self.n_iter:
            # print self.current_assist_level
            self.get_cursor_location(**kwargs)
        
            if self.idx % n_iter_betw_fb == 0:
                if saved_state.dtype == np.float32:
                    error = saved_state[:self.idx,:,-1] - np.float32(self.decoder_state[:self.idx,:,-1])
                else:
                    error = saved_state[:self.idx,:,-1] - self.decoder_state[:self.idx,:,-1]
                print "Error after %d iterations" % self.idx, np.max(np.abs(error))



        if saved_state.dtype == np.float32:
            error = saved_state[:self.n_iter,:,-1] - np.float32(self.decoder_state[:self.n_iter,:,-1])
        else:
            error = saved_state[:self.n_iter,:,-1] - self.decoder_state[:self.n_iter,:,-1]

        return error

class FixedPPFBMIReconstruction(BMIReconstruction):
    def init_decoder_state(self):
        '''
        Initialize the state of the decoder to match the initial state of the plant
        '''
        self.decoder.set_call_rate(self.fps)

class LFPBMIReconstruction(BMIReconstruction):
    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.ReplayLFPPowerExtractor(self.te.hdf.root.task)
        self._add_feature_extractor_dtype()

class CLDAReconstruction(FixedPPFBMIReconstruction):
    def load_decoder(self):
        '''
        Create the object for the initial decoder
        '''
        self.decoder = dbfn.get_decoder(self.te.record)
        self.n_subbins = self.decoder.n_subbins
        self.decoder_state = np.zeros([self.n_iter, self.decoder.n_states, self.n_subbins])    

    def _update_target_loc(self):
        self.target_location = self.te.hdf.root.task[self.idx]['target']

        # self.state = self.task_state[self.idx]
        if self.idx in self.update_bmi_msgs['time']:
            # recon_state = self.task_state[self.idx]
            self.state = self.task_state[self.idx-1]
        else:
            self.state = 'no_target'

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(CLDAReconstruction, self).call_decoder(*args, **kwargs)            

    def create_goal_calculator(self):
        self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        data, solution_updated = self.goal_calculator(self.target_location, verbose=False, n_particles=500, n_iter=10, q_start=self.plant.get_intrinsic_coordinates())
        target_state, error = data

        return np.array(target_state).reshape(-1,1)

class ContCLDARecon(CLDAReconstruction):
    param_noise_scale = 1.
    def __init__(self, te, n_iter, *args, **kwargs):
        self.tau = float(te.params['tau'])
        super(ContCLDARecon, self).__init__(te, n_iter, *args, **kwargs)

    def init_decoder_state(self):
        '''
        Initialize the state of the decoder to match the initial state of the plant
        '''
        init_decoder_state = np.array([0., 0, 0, 0, 0, 0, 1]).reshape(-1,1)
        self.init_decoder_mean = init_decoder_state
        self.decoder.filt.state.mean = self.init_decoder_mean#np.array(self.init_decoder_mean).ravel()

        self.decoder.set_call_rate(self.fps)

    def create_learner(self):
        self.learn_flag = True

        kwargs = dict()
        dt = kwargs.pop('dt', 1./180)
        use_tau_unNat = self.tau
        self.tau = use_tau_unNat
        print "learner cost fn param: %g" % use_tau_unNat
        tau_scale = 28*use_tau_unNat/1000
        bin_num_ms = (dt/0.001)
        w_r = 3*tau_scale**2/2*(bin_num_ms)**2*26.61
        
        I = np.eye(3)
        zero_col = np.zeros([3, 1])
        zero_row = np.zeros([1, 3])
        zero = np.zeros([1,1])
        one = np.ones([1,1])
        A = np.bmat([[I, dt*I, zero_col], 
                     [0*I, 0*I, zero_col], 
                     [zero_row, zero_row, one]])
        B = np.bmat([[0*I], 
                     [dt/1e-3 * I],
                     [zero_row]])
        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = np.mat(np.diag([w_r, w_r, w_r]))
        
        F = feedback_controllers.LQRController.dlqr(A, B, Q, R)
        F_dict = dict(target=F, hold=F) 

        fb_ctrl = feedback_controllers.MultiModalLFC(A=A, B=B, F_dict=F_dict)

        batch_size = 1

        self.learner = clda.OFCLearner(batch_size, A, B, F_dict)
        # super(OFCLearner3DEndptPPF, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

        # Tell BMISystem that this learner wants the most recent output
        # of the decoder rather than the second most recent, to match MATLAB
        self.learner.input_state_index = 0

    def create_updater(self):
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder, param_noise_scale=self.param_noise_scale)

    def _update_target_loc(self):
        self.target_location = self.te.hdf.root.task[self.idx]['target']

        # self.state = self.task_state[self.idx]
        if self.idx in self.update_bmi_msgs['time']:
            # recon_state = self.task_state[self.idx]
            self.state = 'target'
        else:
            self.state = 'no_target'

    def create_assister(self):
        from tasks.bmimultitasks import OFCEndpointAssister
        self.assister = OFCEndpointAssister()
         

class KFRMLRecon(CLDAReconstruction):
    def __init__(self, te, n_iter, *args, **kwargs):
        self.batch_time = 0.1
        try:
            self.half_life = tuple(te.half_life)
        except:
            self.half_life = (120., 120.)
        super(KFRMLRecon, self).__init__(te, n_iter, *args, **kwargs)

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(KFRMLRecon, self).call_decoder(*args, **kwargs)        

    def create_assister(self):
        # Create the appropriate type of assister object
        start_level, end_level = self.assist_level
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed

        from tasks.bmimultitasks import SimpleEndpointAssister
        self.assister = SimpleEndpointAssister(**kwargs)


class KFRMLCGRecon(KFRMLRecon):
    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)                
        self.learner = cursor_clda_tasks.CursorGoalLearner2(self.batch_size, int_speed_type='decoded_speed')
        self.learn_flag=True

    def _update_target_loc(self):
        self.target_location = self.te.hdf.root.task[self.idx]['target']

        self.state = self.task_state[self.idx-1]
        ##if self.idx in self.update_bmi_msgs['time']:
        ##    recon_state = self.task_state[self.idx - 1]
        ##else:
        ##    self.state = 'no_target'

class KFRMLJointRecon(KFRMLRecon):
    def calc_recon_error(self, **kwargs):
        while self.idx < self.n_iter:
            # print self.current_assist_level
            self.get_cursor_location(**kwargs)
        
        joint_angles = self.te.hdf.root.task[:]['joint_angles']
        error = joint_angles[:self.n_iter, [1,3]] - self.decoder_state[:self.n_iter, [1,3], -1]

        return error    

    def create_learner(self):
        self.learner = clda.CursorGoalLearner2(self.batch_size, int_speed_type='decoded_speed')

class PointMassBMIReconstruction(BMIReconstruction):
    def __init__(self, te, n_iter, *args, **kwargs):
        self.te = te
        self.n_iter = min(n_iter, len(te.hdf.root.task))


        ## Set the target radius because the old assist method changes the assist speed
        # when the cursor is inside the target
        self.target_radius = te.target_radius
        self.cursor_radius = te.cursor_radius
        self.assist_level = tuple(te.assist_level)

        self.idx = 0
        super(PointMassBMIReconstruction, self).__init__(te, n_iter, *args, **kwargs)

        from tasks.point_mass_cursor import CursorPlantWithMass
        self.plant = CursorPlantWithMass()        
        
        self.hdf = SimHDF()
        self.learn_flag = True

        task_msgs = te.hdf.root.task_msgs[:]
        self.update_bmi_msgs = task_msgs[task_msgs['msg'] == 'update_bmi']
        task_msgs = filter(lambda x: x['msg'] not in ['update_bmi'], task_msgs)
        # print task_msgs
        self.task_state = np.array([None]*n_iter)
        for msg, next_msg in izip(task_msgs[:-1], task_msgs[1:]):
            self.task_state[msg['time']:next_msg['time']] = msg['msg']

        self.update_bmi_inds = np.zeros(len(te.hdf.root.task))
        self.update_bmi_inds[self.update_bmi_msgs['time']] = 1
        self.recon_update_bmi_inds = np.zeros(len(te.hdf.root.task))

        self.target_hold_msgs = filter(lambda x: x['msg'] in ['target', 'hold'], te.hdf.root.task_msgs[:])
        self.te = te    

    def create_assister(self):
        from tasks.point_mass_cursor import PointMassFBController
        from riglib.bmi.assist import FeedbackControllerAssist
        fb_ctrl = PointMassFBController()
        self.assister = FeedbackControllerAssist(fb_ctrl, style='mixing')

    def create_goal_calculator(self):
        pass

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        target_state = np.array(np.hstack([self.target_location, np.zeros(3), np.zeros(3), np.zeros(3), 1]))
        return np.tile(np.array(target_state).reshape(-1,1), [1, self.decoder.n_subbins])

class PointMassCLDAReconstruction(PointMassBMIReconstruction):
    batch_time = traits.Float(0.1, desc='The length of the batch in seconds')
    decoder_sequence = traits.String('test', desc='signifier to group together sequences of decoders')

    def __init__(self, *args, **kwargs):
        super(PointMassCLDAReconstruction, self).__init__(*args, **kwargs)
        self.half_life = self.te.half_life

    def create_learner(self):
        from tasks.point_mass_cursor import PointMassFBController
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        fb_ctrl = PointMassFBController()
        self.learner = clda.FeedbackControllerLearner(self.batch_size, fb_ctrl)
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(PointMassCLDAReconstruction, self).call_decoder(*args, **kwargs)


