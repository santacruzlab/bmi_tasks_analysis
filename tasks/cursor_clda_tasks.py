from bmimultitasks import BMIControlMulti, SimBMIControlMulti
from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife
from riglib.bmi import feedback_controllers, train
from riglib.experiment import traits
from riglib.stereo_opengl.window import WindowDispl2D
import numpy as np

################
####### Learners
################
# from riglib.bmi.clda import Learner, OFCLearner
from riglib.bmi import clda


class CursorGoalLearner2(clda.Learner):
    '''
    CLDA intention estimator based on CursorGoal/Refit-KF ("innovation 1" in Gilja*, Nuyujukian* et al, Nat Neurosci 2012)
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for CursorGoalLearner2

        Parameters
        ----------
        int_speed_type: string, optional, default='dist_to_target'
            Specifies the method to use to estimate the intended speed of the target.
            * dist_to_target: scales based on remaining distance to the target position
            * decoded_speed: use the speed output provided by the decoder, i.e., the difference between the intention and the decoder output can be described by a pure vector rotation

        Returns
        -------
        CursorGoalLearner2 instance
        '''
        int_speed_type = kwargs.pop('int_speed_type', 'dist_to_target')
        self.int_speed_type = int_speed_type
        if not self.int_speed_type in ['dist_to_target', 'decoded_speed']:
            raise ValueError("Unknown type of speed for cursor goal: %s" % self.int_speed_type)

        super(CursorGoalLearner2, self).__init__(*args, **kwargs)

        if self.int_speed_type == 'dist_to_target':
            self.input_state_index = 0

    def calc_int_kin(self, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """
        Calculate the intended kinematics and pair with the neural data
        """
        if state_order is None:
            raise ValueError("New cursor goal requires state order to be specified!")

        # The intended direction (abstract space) from the current state of the decoder to the target state for the task
        int_dir = target_state - decoder_state
        vel_inds, = np.nonzero(state_order == 1)
        pos_inds, = np.nonzero(state_order == 0)
        
        # Calculate intended speed
        if task_state in ['hold', 'origin_hold', 'target_hold']:
            speed = 0
        #elif task_state in ['target', 'origin', 'terminus']:
        else:
            if self.int_speed_type == 'dist_to_target':
                speed = np.linalg.norm(int_dir[pos_inds])
            elif self.int_speed_type == 'decoded_speed':
                speed = np.linalg.norm(decoder_output[vel_inds])
        #else:
        #    speed = np.nan

        int_vel = speed*self.normalize(int_dir[pos_inds])
        int_kin = np.hstack([decoder_output[pos_inds], int_vel, 1]).reshape(-1, 1)

        if np.any(np.isnan(int_kin)):
            int_kin = None

        return int_kin

    def __call__(self, spike_counts, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """
        Calculate the intended kinematics and pair with the neural data
        """
        if state_order is None:
            raise ValueError("CursorGoalLearner2.__call__ requires state order to be specified!")
        super(CursorGoalLearner2, self).__call__(spike_counts, decoder_state, target_state, decoder_output, task_state, state_order=state_order)
    
    @staticmethod
    def normalize(vec):
        '''
        Vector normalization. If the vector to be normalized is of norm 0, a vector of 0's is returned

        Parameters
        ----------
        vec: np.ndarray of shape (N,) or (N, 1)
            Vector to be normalized

        Returns
        -------
        norm_vec: np.ndarray of shape matching 'vec'
            Normalized version of vec
        '''
        norm_vec = vec / np.linalg.norm(vec)
        
        if np.any(np.isnan(norm_vec)):
            norm_vec = np.zeros_like(vec)
        
        return norm_vec

########################
###### CLDA cursor tasks
########################

class CLDAControlMulti(BMIControlMulti, LinearlyDecreasingHalfLife):
    '''
    BMI task that periodically refits the decoder parameters based on intended
    movements toward the targets. Inherits directly from BMIControl. Can be made
    to automatically linearly decrease assist level over set time period, or
    to provide constant assistance by setting assist_level and assist_min equal.
    '''

    batch_time = traits.Float(80.0, desc='The length of the batch in seconds')
    decoder_sequence = traits.String('test', desc='signifier to group together sequences of decoders')

    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'batch_time', 'half_life', 'half_life_time']

    def __init__(self, *args, **kwargs):
        super(CLDAControlMulti, self).__init__(*args, **kwargs)
        self.learn_flag = True

    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        self.learner = CursorGoalLearner2(self.batch_size)
        self.learn_flag = True

    def create_updater(self):
        half_life_start, half_life_end = self.half_life
        self.updater = clda.KFSmoothbatch(self.batch_time, half_life_start)

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(CLDAControlMulti, self).call_decoder(*args, **kwargs)


class CLDAControlKFCG(CLDAControlMulti):
    memory_decay_rate = traits.Float(0.45, desc='Shape parameter for the impulse response of the KF')
    ordered_traits = ['session_length', 'assist_level', 'assist_time', 'batch_time', 'half_life', 'half_life_decay_time', 'memory_decay_rate']

    def create_learner(self):
        self.learner = CursorGoalLearner2(self.batch_size, int_speed_type='decoded_speed')
        self.learn_flag = True

    def create_updater(self):
        # set defaults for C^T * Q^-1 *C
        default_gain = None #np.diag([self.memory_decay_rate, self.memory_decay_rate, self.memory_decay_rate])

        self.updater = clda.KFOrthogonalPlantSmoothbatch(
            self.batch_time, self.half_life[0], default_gain=default_gain)

class CLDAControlKFCGRML(CLDAControlMulti):
    memory_decay_rate = traits.Float(0.45, desc='Shape parameter for the impulse response of the KF')
    ordered_traits = ['session_length', 'assist_level', 'assist_time', 'batch_time', 'half_life', 'half_life_decay_time', 'memory_decay_rate']

    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)                
        self.learner = CursorGoalLearner2(self.batch_size, int_speed_type='decoded_speed')
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])


class CLDAControlKFCGRMLIVCTRIAL(CLDAControlKFCGRML):
    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)                
        self.learner = CursorGoalLearner2(self.batch_size, int_speed_type='decoded_speed', done_states=['reward', 'hold_penalty'], reset_states=['timeout_penalty'])
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.KFRML_IVC(self.batch_time, self.half_life[0])

    def _cycle(self):
        super(CLDAControlKFCGRMLIVCTRIAL, self)._cycle()
        if self.calc_state_occurrences('reward') > 16:
            self.learner.batch_size = np.inf

class CLDAConstrainedSSKFMulti(CLDAControlMulti):
    def create_updater(self):
        self.updater = clda.KFOrthogonalPlantSmoothbatch(self.batch_time, self.half_life)

class CLDARMLKF(CLDAControlMulti):
    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def init(self):
        super(CLDARMLKF, self).init()
        self.batch_time = self.decoder.binlen


class CLDARMLKFOFC(CLDARMLKF):
    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        A, B, _ = self.decoder.ssm.get_ssm_matrices()

        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = 10000*np.mat(np.eye(B.shape[1]))
        from tentaclebmitasks import OFCLearnerTentacle
        self.learner = OFCLearnerTentacle(self.batch_size, A, B, Q, R)
        self.learn_flag = True

class CLDAControlPPFContAdapt(CLDAControlMulti):
    tau = traits.Float(2.7, desc="Magic parameter for speed of OFC.")
    param_noise_scale = traits.Float(1.0, desc="Stuff")
    exclude_parent_traits = ['half_life', 'half_life_decay_time', 'batch_time']

    ordered_traits = ['session_length', 'assist_level', 'assist_time', 'tau', 'param_noise_scale']

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

        # Tell BMISystem that this learner wants the most recent output
        # of the decoder rather than the second most recent, to match MATLAB
        self.learner.input_state_index = 0
        
    def create_updater(self):
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder, param_noise_scale=self.param_noise_scale)

class CLDAControlPPFContAdapt2(CLDAControlMulti):
    exclude_parent_traits = ['half_life', 'half_life_decay_time', 'batch_time']
    param_noise_scale = traits.Float(1.0, desc="Stuff")
    cost_fn_scale = traits.Float(10000, desc="Stuff")
    ordered_traits = ['session_length', 'assist_level', 'assist_time', 'param_noise_scale']    

    def create_learner(self):
        self.batch_size = 1
        A, B, _ = self.decoder.ssm.get_ssm_matrices(update_rate=self.decoder.binlen)

        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = self.cost_fn_scale*np.mat(np.eye(B.shape[1]))
        from tasks.tentaclebmitasks import OFCLearnerTentacle
        self.learner = OFCLearnerTentacle(self.batch_size, A, B, Q, R)
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder, param_noise_scale=self.param_noise_scale)

class CLDAControlRMLPPF(CLDAControlMulti):
    exclude_parent_traits = ['batch_time']
    cost_fn_scale = traits.Float(10000, desc="Stuff")
    ordered_traits = ['session_length', 'assist_level', 'assist_time', 'param_noise_scale']
    batch_time = 1./180

    def create_learner(self):
        A, B, _ = self.decoder.ssm.get_ssm_matrices(update_rate=self.decoder.binlen)

        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = self.cost_fn_scale*np.mat(np.eye(B.shape[1]))
        fb_ctrl = feedback_controllers.LQRController(A, B, Q, R)
        self.learner = clda.FeedbackControllerLearner(1, fb_ctrl)
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.PPFRML()

class SimCLDAControlMulti(SimBMIControlMulti, CLDAControlMulti):
    def load_decoder(self):
        '''
        Create a 'seed' decoder for the simulation which is simply randomly initialized
        '''
        from riglib.bmi import state_space_models
        ssm = state_space_models.StateSpaceEndptVel2D()
        self.decoder = train.rand_KFDecoder(ssm, self.encoder.get_units())


from features.bmi_task_features import LinearlyDecreasingAttribute
class LinearlyDecreasingN(LinearlyDecreasingAttribute):
    memory_decay_rate = traits.Tuple((-1., 0.5), desc="")
    memory_decay_rate_time = traits.Float(300, desc="")

    def __init__(self, *args, **kwargs):
        super(LinearlyDecreasingAssist, self).__init__(*args, **kwargs)
        if 'memory_decay_rate' not in self.attrs:
            self.attrs.append('memory_decay_rate')


class CLDARMLKFOFCIVC(CLDARMLKFOFC):
    memory_decay_rate = traits.Float(0.5, desc="")
    def create_updater(self):
        self.updater = clda.KFRML_IVC(self.batch_time, self.half_life[0])
        self.updater.default_gain = self.memory_decay_rate

class CLDARMLKF_2DWindow(CLDARMLKF, WindowDispl2D):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(CLDARMLKF_2DWindow, self).__init__(*args, **kwargs)
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
    
    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        self.learner = CursorGoalLearner2(self.batch_size)
        self.learn_flag = True

    def create_assister(self):
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed 
        from bmimultitasks import SimpleEndpointAssister   
        self.assister = SimpleEndpointAssister(**kwargs)

    def _start_wait(self):
        self.wait_time = 0.
        super(CLDARMLKF_2DWindow, self)._start_wait()
    
    def create_goal_calculator(self):
        from riglib.bmi import goal_calculators #ZeroVelocityGoal
        self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

class CLDAKFSmoothBatch_2DWindow(CLDAControlMulti, WindowDispl2D):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(CLDAKFSmoothBatch_2DWindow, self).__init__(*args, **kwargs)

    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        self.learner = CursorGoalLearner2(self.batch_size)
        self.learn_flag = True

    def create_assister(self):
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed 
        from bmimultitasks import SimpleEndpointAssister   
        self.assister = SimpleEndpointAssister(**kwargs)

    def _start_wait(self):
        self.wait_time = 0.
        super(CLDAKFSmoothBatch_2DWindow, self)._start_wait()
    
    def create_goal_calculator(self):
        from riglib.bmi import goal_calculators #ZeroVelocityGoal
        self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    #     if self.memory_decay_rate[0] == -1:
    #         F, K = self.decoder.filt.get_sskf()
    #         n = np.mean([F[3,3], F[5,5]])
    #         self.memory_decay_rate[0] = n
    #     super(CLDARMLKFOFCIVC, self).init()

