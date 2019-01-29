'''
Tasks to create and run a BMI controlling a point mass
'''
import numpy as np
from riglib.bmi.state_space_models import State, StateSpace, offset_state
from riglib.bmi.assist import FeedbackControllerAssist
from riglib.bmi.goal_calculators import ZeroVelocityAccelGoal
from riglib.bmi.bmi import Decoder
from riglib.bmi.clda import OFCLearner, Learner, RegexKeyDict
from riglib.bmi import feedback_controllers, clda

from riglib import plants
from riglib.plants import CursorPlant
from passivetasks import EndPostureFeedbackController, MachineOnlyFilter
from bmimultitasks import BMIControlMulti
from cursor_clda_tasks import CLDAControlMulti
import os
from riglib.bmi.extractor import DummyExtractor

from features import simulation_features
from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife
from riglib.experiment import traits


class CursorPlantWithMass(CursorPlant):
    Delta = 1./60 # call rate
    hdf_attrs = [('cursor_pos', 'f8', (3,)), ('cursor_vel', 'f8', (3,))]
    def __init__(self, *args, **kwargs):
        super(CursorPlantWithMass, self).__init__(*args, **kwargs)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.mass = 1 # kg

    def drive(self, decoder):
        # decoder supplies 3-D force vector
        force = decoder['force_x', 'force_y', 'force_z']

        # run kinematics
        acceleration = 1./self.mass * force
        self.velocity += self.Delta * acceleration
        self.position += self.Delta * self.velocity + 0.5*self.Delta**2 * acceleration

        # bound position and velocity
        self.position, self.velocity = self._bound(self.position, self.velocity)
        decoder['q'] = self.position
        decoder['qdot'] = self.velocity
        decoder['hand_ax', 'hand_ay', 'hand_az'] = self.acceleration
        self.draw()

    def get_data_to_save(self):
        return dict(cursor_pos=self.position, cursor_vel=self.velocity)


###################################
##### State space model declaration
###################################
class PointForceStateSpace(StateSpace):
    def __init__(self):
        self.states = [
            State('hand_px', stochastic=False,  drives_obs=False, order=0, aux=True),
            State('hand_py', stochastic=False, drives_obs=False, order=0, aux=True),
            State('hand_pz', stochastic=False,  drives_obs=False, order=0, aux=True),

            State('hand_vx', stochastic=False,  drives_obs=False, order=1, aux=True),
            State('hand_vy', stochastic=False, drives_obs=False, order=1, aux=True),
            State('hand_vz', stochastic=False,  drives_obs=False, order=1, aux=True),

            State('hand_ax', stochastic=False,  drives_obs=False, order=2, aux=True),
            State('hand_ay', stochastic=False, drives_obs=False, order=2, aux=True),
            State('hand_az', stochastic=False,  drives_obs=False, order=2, aux=True),

            State('force_x', stochastic=True,  drives_obs=True, order=2),
            State('force_y', stochastic=False, drives_obs=False, order=2),
            State('force_z', stochastic=True,  drives_obs=True, order=2),
            offset_state]

        self.mass = 1 # kg

    def get_ssm_matrices(self, update_rate=0.1):
        I = np.mat(np.eye(3))
        Delta = update_rate
        zero_vec = np.zeros([3,1])
        D60 = 1./60
        pos_vel_int_gain = D60 + D60**2 + D60**3 + D60**4 + D60**5 + D60**6
        A = np.vstack([np.hstack([I,   Delta*I,            0.5*Delta**2*I, 0*I,   zero_vec]), 
                       np.hstack([0*I, I,                  Delta*I,        0*I,   zero_vec]), 
                       np.hstack([0*I, 0*I,                0*I,            0*I,   zero_vec]),
                       np.hstack([0*I, 0*I,                0*I,            0*I, zero_vec]),
                       np.hstack([zero_vec.T, zero_vec.T, zero_vec.T, zero_vec.T, np.ones([1,1])]),
                      ])

        B = np.vstack([0*I, 
                       0*I, 
                       Delta*1000*I, 
                       self.mass*Delta*1000*I, 
                       zero_vec.T])

        W = np.vstack([np.hstack([0*I, 0*I, 0*I, 0*I,   zero_vec]), 
                       np.hstack([0*I, 0*I, 0*I, 0*I,   zero_vec]), 
                       np.hstack([0*I, 0*I, 70*I, 0*I,   zero_vec]),
                       np.hstack([0*I, 0*I, 0*I, 70*I, zero_vec]),
                       np.hstack([zero_vec.T, zero_vec.T, zero_vec.T, zero_vec.T, np.zeros([1,1])]),
                      ]) 

        return A, B, W

###################################
##### Ideal feedback control policy
###################################
ssm = PointForceStateSpace()
class PointMassFBController(feedback_controllers.LQRController):
    def __init__(self):
        I = np.mat(np.eye(3))
        Delta = 1./60
        ssm = PointForceStateSpace()
        A, B, _ = ssm.get_ssm_matrices(update_rate=0.1)
        Q = np.mat(np.diag(np.hstack([np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3), 0])))
        R = np.mat(np.eye(3)) * 1000
        super(PointMassFBController, self).__init__(A, B, Q, R)


class BMIPointMassCursor(BMIControlMulti):
    exclude_parent_traits = ['plant_type', 'plant_visible', 'plant_hide_rate']
    def __init__(self, *args, **kwargs):
        self.plant = CursorPlantWithMass(endpt_bounds=(-14, 14, 0., 0., -14, 14))
        super(BMIPointMassCursor, self).__init__(*args, **kwargs)

    def init_decoder_state(self):
        pass

    def create_assister(self):
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


class PointMassVisualFeedback(BMIPointMassCursor):
    exclude_parent_traits = []
    assist_level = (1, 1)
    is_bmi_seed = True

    def load_decoder(self):
        self.ssm = PointForceStateSpace()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_assister(self):
        fb_ctrl = PointMassFBController()
        self.assister = FeedbackControllerAssist(fb_ctrl, style='mixing')

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()


class CLDAPointMassCursor(BMIPointMassCursor, LinearlyDecreasingHalfLife):
    batch_time = traits.Float(0.1, desc='The length of the batch in seconds')
    decoder_sequence = traits.String('test', desc='signifier to group together sequences of decoders')

    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        fb_ctrl = PointMassFBController()
        self.learner = clda.FeedbackControllerLearner(self.batch_size, fb_ctrl)
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def _cycle(self):
        super(CLDAPointMassCursor, self)._cycle()
        if self.calc_state_occurrences('reward') > 16:
            self.learner.batch_size = np.inf

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(CLDAPointMassCursor, self).call_decoder(*args, **kwargs)


class CLDABaselinePointMassCursor(CLDAPointMassCursor):
    '''
    Only re-estimate baseline firing rates
    '''
    def create_updater(self):
        self.updater = clda.KFRML_baseline(self.batch_time, self.half_life[0])
        

class SimCLDAPointMassCursor(simulation_features.SimKalmanEnc, simulation_features.SimKFDecoderShuffled, CLDAPointMassCursor):
    assist_level = (0, 0)
    assist_level_time = 60.
    half_life = (10., 450.)
    half_life_time = 300
    def __init__(self, *args, **kwargs):
        kwargs['fb_ctrl'] = PointMassFBController()
        kwargs['ssm'] = PointForceStateSpace()
        super(SimCLDAPointMassCursor, self).__init__(*args, **kwargs)


