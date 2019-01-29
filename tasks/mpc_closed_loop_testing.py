from bmimultitasks import BMIControlMulti
from cursor_clda_tasks import CLDAControlMulti
import numpy as np
from riglib.experiment import traits
from riglib.bmi import clda
from riglib.bmi.kfdecoder import KalmanFilter
from riglib.bmi import bmi
from riglib.bmi.ppfdecoder import PointProcessFilter
from riglib.bmi.assist import Assister
from bmimultitasks import SimpleEndpointAssister, OFCEndpointAssister

class CursorMPCTest(BMIControlMulti):
    sequence_generators = ['centerout_2d_discrete_mpc_switching']
    def init(self):
        self.add_dtype('r_scale', np.float64, (1,))
        super(CursorMPCTest, self).init()

    def _cycle(self):
        self.task_data['r_scale'] = self.decoder.filt.r_scale
        super(CursorMPCTest, self)._cycle()

    @staticmethod
    def centerout_2d_discrete_mpc_switching(n_switches=10, ntargets=8,
        distance=10, r_scale=20, n_blocks_between_switches=6, metablock_pseudorand_n=4):
        '''
        '''
        r_scale_vals_by_block = []
        for k in range(n_switches):
            M = metablock_pseudorand_n/2
            data = [r_scale]*M + [np.inf]*M
            np.random.shuffle(data)
            r_scale_vals_by_block += data

        data = []

        for r_ in r_scale_vals_by_block:
            trial_target_seqs = CursorMPCTest.centerout_2D_discrete(distance=distance, ntargets=ntargets, nblocks=n_blocks_between_switches)
            for tt in trial_target_seqs:
                data.append((tt, r_))
        return data

    def _parse_next_trial(self):
        self.targs, self.r_scale = self.next_trial        

    def _start_wait(self):
        super(CursorMPCTest, self)._start_wait()
        self.decoder.filt.r_scale = self.r_scale
        self.reportstats['r_scale'] = str(self.r_scale)

class CursorMPCFixed(BMIControlMulti):
    r_scale = traits.Float('20')
    def init(self):
        self.decoder.filt.r_scale = self.r_scale
        super(CursorMPCFixed, self).init()

class RScaleUpdater(clda.Updater):
    update_kwargs = dict(steady_state=False)
    def __init__(self, batch_time, half_life):
        super(RScaleUpdater, self).__init__(self.calc, multiproc=False)
        self.batch_time = batch_time
        self.half_life = half_life
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))

    def init(self, decoder):
        _, self.K = decoder.filt.get_sskf()
        self.C = decoder.filt.C
        D = decoder.filt.C_xpose_Q_inv_C
        D[:,-1] = 0
        D[-1,:] = 0

        self.G = np.linalg.pinv(D)*decoder.filt.C_xpose_Q_inv
        self.r_scale = decoder.filt.r_scale
        self.alpha = 1./(1 + self.r_scale)

    def calc(self, intended_kin=None, spike_counts=None, decoder=None, half_life=None, values=None, **kwargs):
        y = np.mat(spike_counts)
        y_t = y[:,1:]
        y_tm1 = y[:,:-1]

        K = self.K
        G = self.G
        C = self.C
        r1 = K*y_t
        r2 = K*C*G*y_tm1
        Alpha = (r1*np.linalg.pinv(r2))

        if not (half_life is None):
            rho = np.exp(np.log(0.5)/(half_life/self.batch_time))
        else:
            rho = self.rho 

        alpha_hat = np.mean(np.diag(Alpha[np.ix_([3,5], [3,5])]))
        print alpha_hat
        self.alpha = (1-self.rho)*alpha_hat + rho*self.alpha

        r_scale = (1 - self.alpha)/self.alpha
        print r_scale

        new_params = dict()
        new_params['filt.r_scale'] = r_scale
        self._new_params = new_params
        return new_params

class CLDACursorAdaptRscale(CLDAControlMulti):
    r_scale = traits.Float('20')
    def init(self):
        self.decoder.filt.r_scale = self.r_scale
        super(CLDACursorAdaptRscale, self).init()

    def create_updater(self):
        self.updater = RScaleUpdater(self.batch_time, self.half_life[0])


class CursorEllipticalRandomWalkKF(KalmanFilter):
    def _pickle_init(self):
        super(CursorEllipticalRandomWalkKF, self)._pickle_init()
        self.W_orig = self.W.copy()
        self.W_curr = self.W_orig
        self.w_scale = self.W_orig[3,3]
        self.w_aspect_ratio = 2
        self.R90 = np.array([[0, -1.],
                             [1., 0]])

    def _ssm_pred(self, state, **kwargs):
        # set self.state_noise based on v_t
        x_t = state.mean
        v_t = np.array(x_t)[[3,5],0]
        angle_vt = np.arctan2(v_t[-1], v_t[0])
        if np.linalg.norm(v_t) == 0:
            W = self.W_orig
        else:
            norm_vt = np.linalg.norm(v_t)
            v_t = v_t / norm_vt
            V_xpose = np.vstack([v_t, np.dot(self.R90, v_t)])
            V = V_xpose.T
            sca = self.w_zero_scale + (1-self.w_zero_scale)*norm_vt/self.v_max
            E = sca*np.diag([self.w_scale*self.w_aspect_ratio, self.w_scale/self.w_aspect_ratio])
            W_v = np.dot(np.dot(V_xpose, E), V)

            W = self.W.copy()
            W[np.ix_([3,5], [3,5])] = W_v

        # print np.round(W, decimals=3)
        self.W_curr = W
        self.state_noise = bmi.GaussianState(0.0, W)

        return super(CursorEllipticalRandomWalkKF, self)._ssm_pred(state, **kwargs)

class CursorEllipticalRandomWalkPPF(PointProcessFilter):
    def _pickle_init(self):
        super(CursorEllipticalRandomWalkPPF, self)._pickle_init()
        self.W_orig = self.W.copy()
        self.W_curr = self.W_orig
        self.w_scale = self.W_orig[3,3]
        self.w_aspect_ratio = 2
        self.R90 = np.array([[0, -1.],
                             [1., 0]])

    def _ssm_pred(self, state, **kwargs):
        # set self.state_noise based on v_t
        x_t = state.mean
        v_t = np.array(x_t)[[3,5],0]
        angle_vt = np.arctan2(v_t[-1], v_t[0])
        if np.linalg.norm(v_t) == 0:
            W = self.W_orig
        else:
            norm_vt = np.linalg.norm(v_t)
            v_t = v_t / norm_vt
            V_xpose = np.vstack([v_t, np.dot(self.R90, v_t)])
            V = V_xpose.T
            sca = self.w_zero_scale + (1-self.w_zero_scale)*norm_vt/self.v_max
            E = sca*np.diag([self.w_scale*self.w_aspect_ratio, self.w_scale/self.w_aspect_ratio])
            W_v = np.dot(np.dot(V_xpose, E), V)

            W = self.W.copy()
            W[np.ix_([3,5], [3,5])] = W_v

        # print np.round(W, decimals=3)
        self.W_curr = W
        self.state_noise = bmi.GaussianState(0.0, W)

        return super(CursorEllipticalRandomWalkPPF, self)._ssm_pred(state, **kwargs)

class TargetEstFakeAssister(OFCEndpointAssister):
    def __init__(self, *args, **kwargs):
        super(TargetEstFakeAssister, self).__init__(*args, **kwargs)
        self.current_state_to_target_state_map = \
            np.mat([[ 0.955,  0.   , -0.018,  0.416,  0.   ,  0.098,  0.141],
                      [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
                      [-0.035, -0.   ,  0.945,  0.002,  0.   ,  0.435, -0.111]])

    def calc_assisted_BMI_state(self, current_state, target_state, *args, **kwargs):
        # print current_state
        # print self.current_state_to_target_state_map * current_state
        # print target_state[0:3,0]
        target_state[0:3,:] = self.current_state_to_target_state_map * current_state
        res = super(TargetEstFakeAssister, self).calc_assisted_BMI_state(current_state, target_state, *args, **kwargs)
        # print res
        return res

class BMIControlCursorElip(BMIControlMulti):
    w_aspect_ratio = traits.Float(2., desc="aspect ratio for W matrix")
    w_zero_scale = traits.Float(0.5, desc="aspect ratio for W matrix")
    v_max = traits.Float(20, desc="aspect ratio for W matrix")
    def init(self):
        super(BMIControlCursorElip, self).init()
        self.decoder.filt.w_aspect_ratio = self.w_aspect_ratio
        self.decoder.filt.w_zero_scale = self.w_zero_scale
        self.decoder.filt.v_max = self.v_max

class BMIControlCursorTargetDirected(BMIControlMulti):
    def create_assister(self):
        # Create the appropriate type of assister object
        start_level, end_level = self.assist_level
        self.assister = TargetEstFakeAssister(decoding_rate=10)        
        print self.assister

class BMIControlCursorHold(BMIControlMulti):
    '''
    A cursor BMI task where reward is proportional to length of hold
    '''
    _target_color = (0, 0, 1,.5)
    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    reward_mult_factor = traits.Float(1.0, desc="reward time = reward_mult_factor * hold_time")
    max_hold = traits.Float(2.0, desc="max time for rewarded hold")

    def __init__(self, *args, **kwargs):
        super(BMIControlCursorHold, self).__init__(*args, **kwargs)
        self.hold_start_time = np.nan

    def init(self):
        self.add_dtype('reward_time', 'f8', (1,))
        super(BMIControlCursorHold, self).init()

    def _cycle(self):
        self.task_data['reward_time'] = self.reward_time
        super(BMIControlCursorHold, self)._cycle()        

    def _start_hold(self):
        self.hold_start_time = self.get_time()

    def _test_hold_complete(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius - self.cursor_radius
        
        outside_targ = d > rad
        compl = (ts > self.max_hold) or outside_targ
        if compl:
            self.reward_time += ts #(ts - self.hold_start_time)
        return compl

    def _test_trial_abort(self, ts):
        abort = super(BMIControlCursorHold, self)._test_trial_abort(ts)
        if abort:
            self.reward_time = 0
        return abort

    def _start_wait(self):
        self.reward_time = 0
        super(BMIControlCursorHold, self)._start_wait()


class CursorPDControlKF(KalmanFilter):
    def _pickle_init(self):
        self.last_displ = np.nan
        self.last_state = None
        self.switch_W = False
        super(CursorPDControlKF, self)._pickle_init()
        self.W_orig = self.W.copy()
        self.displ_mean = 0.55
        self.displ_scale = 1

    def _ssm_pred(self, state, **kwargs):
        
        if not self.last_state is None:
            self.last_displ = np.linalg.norm(np.array(self.last_state - state.mean)[0:3])
        self.last_state = state.mean

        if self.switch_W and not np.isnan(self.last_displ):
            # print "changing W!"
            # print self.last_displ
            scale_exp = self.displ_scale * (self.last_displ - self.displ_mean)
            scale_exp = min(max(scale_exp, -1), 1)
            scale = 10**(scale_exp)

            W = self.W_orig.copy() * scale
            # print W
            self.state_noise = bmi.GaussianState(0.0, W)
        else:
            self.state_noise = bmi.GaussianState(0.0, self.W_orig)

        return super(CursorPDControlKF, self)._ssm_pred(state, **kwargs)


class SigmoidalGainKF(KalmanFilter):
    attrs_to_pickle = ['A', 'W', 'C', 'Q', 'C_xpose_Q_inv', 'C_xpose_Q_inv_C', 'R', 'S', 'T', 'ESS', 'sigm_r', 'sigm_m', 'sigm_delta_gain']
    def _pickle_init(self):
        super(SigmoidalGainKF, self)._pickle_init()
        self.last_displ = np.nan
        self.last_state = None

        if not hasattr(self, 'sigm_on'):
            self.sigm_on = False

        if not hasattr(self, 'sigm_r'):
            print "param not saved!"
            self.sigm_r = 0.5

        if not hasattr(self, 'sigm_m'):
            print "param not saved!"
            self.sigm_m = 0.55

        if not hasattr(self, 'sigm_delta_gain'):
            print "param not saved!"
            self.sigm_delta_gain = 0.1

    def _ssm_pred(self, state, **kwargs):
        # track displacement change
        if not self.last_state is None:
            self.last_displ = np.linalg.norm(np.array(self.last_state - state.mean)[0:3])
        self.last_state = state.mean
        return super(SigmoidalGainKF, self)._ssm_pred(state, **kwargs)

    def _calc_kalman_gain(self, *args, **kwargs):
        K = super(SigmoidalGainKF, self)._calc_kalman_gain(*args, **kwargs)
        if self.sigm_on and not np.isnan(self.last_displ):
            delta_gain = self.sigm_delta_gain
            g0 = max(1 - delta_gain, 0)
            g1 = 1 + delta_gain
            r = self.sigm_r
            m = self.sigm_m
            gain = g0 + (g1-g0)/(1. + np.exp(-r*(self.last_displ - m)))
            K *= gain

        return K

from riglib.bmi import feedback_controllers
from riglib.bmi.clda import Updater
class SigmoidalGainUpdater(Updater):
    '''
    Calculate updates for KF parameters using the recursive maximum likelihood (RML) method
    See (Dangi et al, Neural Computation, 2014) for mathematical details.
    '''
    update_kwargs = dict(steady_state=False)
    def __init__(self, batch_time, half_life):
        '''
        Constructor for SigmoidalGainUpdater

        Parameters
        ----------
        batch_time : float
            Size of data batch to use for each update. Specify in seconds.
        half_life : float 
            Amount of time (in seconds) before parameters are half-overwritten by new data.

        Returns
        -------
        SigmoidalGainUpdater instance
        '''
        super(SigmoidalGainUpdater, self).__init__(self.calc, multiproc=False)
        # self.work_queue = None
        self.batch_time = batch_time
        # self.result_queue = None        
        self.half_life = half_life
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))

        self._new_params = None

    def init(self, decoder):
        '''
        Retrieve sufficient statistics from the seed decoder.

        Parameters
        ----------
        decoder : bmi.Decoder instance
            The seed decoder before any adaptation runs.

        Returns
        -------
        None
        '''
        self.F, self.K = decoder.filt.get_sskf()

    def calc(self, intended_kin=None, spike_counts=None, decoder=None, half_life=None, values=None, **kwargs):
        '''
        Parameters
        ----------
        intended_kin : np.ndarray of shape (n_states, batch_size)
            Batch of estimates of intended kinematics, from the learner
        spike_counts : np.ndarray of shape (n_features, batch_size)
            Batch of observations of decoder features, from the learner
        decoder : bmi.Decoder instance
            Reference to the Decoder instance
        half_life : float, optional
            Half-life to use to calculate the parameter change step size. If not specified, the half-life specified when the Updater was constructed is used.
        values : np.ndarray, optional
            Relative value of each sample of the batch. If not specified, each sample is assumed to have equal value.
        kwargs : dict
            Optional keyword arguments, ignored

        Returns
        -------
        new_params : dict
            New parameters to feed back to the Decoder in use by the task.
        '''
        if intended_kin is None or spike_counts is None or decoder is None:
            raise ValueError("must specify intended_kin, spike_counts and decoder objects for the updater to work!")

        Kyt = np.dot(self.K, spike_counts)
        e = intended_kin - Kyt
        # import pdb; pdb.set_trace()

        error_angle = (Kyt.T * e)[0,0]

        r = decoder.filt.sigm_r
        m = decoder.filt.sigm_m
        delta_gain = decoder.filt.sigm_delta_gain

        t = decoder.filt.last_displ
        
    
        if half_life is not None:
            rho = np.exp(np.log(0.5)/(half_life/self.batch_time))
        else:
            rho = self.rho 

        rho = 0.01

        if not np.isnan(t):
            x = np.exp(-r*(t-m))

            dse_dr = delta_gain*(1 + x)**(-2) * x * (t-m) * 2 * error_angle
            dse_dm = delta_gain*(1 + x)**(-2) * x * r * 2 * error_angle
            dse_ddg = 1./(1+x) * 2 * error_angle

            r = r - rho**2*dse_dr
            m = m - rho**2*dse_dm
            delta_gain = delta_gain - rho**2*dse_ddg

            # print new_params
            # import pdb; pdb.set_trace()

        new_params = dict()
        new_params['filt.sigm_r'] = r 
        new_params['filt.sigm_m'] = m
        new_params['filt.sigm_delta_gain'] = delta_gain
        return new_params


class CLDACursorSigmoidalGainKF(CLDAControlMulti):
    batch_time = 0.1
    def load_decoder(self):
        super(CLDACursorSigmoidalGainKF, self).load_decoder()
        if isinstance(self.decoder.filt, KalmanFilter):
            # Add the sigmoidal gain to the KalmanFilter
            old_filt = self.decoder.filt
            filt = SigmoidalGainKF(A=old_filt.A, W=old_filt.W, C=old_filt.C, Q=old_filt.Q)
            filt.C_xpose_Q_inv_C = old_filt.C_xpose_Q_inv_C
            filt.C_xpose_Q_inv = old_filt.C_xpose_Q_inv
            filt.sigm_on = True

            self.decoder.filt = filt

    def create_learner(self):
        F, K = self.decoder.filt.get_sskf()
        pos_gain = np.mean(K[[0,2],:]/K[[3,5],:])
        I = np.mat(np.eye(3))
        B = np.mat(np.vstack([pos_gain*I, I, np.zeros(3)]))
        A = np.mat(F)
        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = np.mat(np.eye(3)*1000)
        fb_ctrl = feedback_controllers.LQRController(A, B, Q, R)
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        self.learner = clda.FeedbackControllerLearner(self.batch_size, fb_ctrl, style='additive')

    def create_updater(self):
        half_life_start, half_life_end = self.half_life
        self.updater = SigmoidalGainUpdater(self.batch_time, half_life_start)


class BMIControlCursorHoldPD(BMIControlCursorHold):
    '''
    Switches between a fixed KF and one where the W matrix scales with stuff
    '''
    displ_scale = traits.Float(1., desc='')
    displ_mean = traits.Float(0.55, desc='')

    sequence_generators = ['centerout_2d_discrete_pd_switching']

    def load_decoder(self):
        super(BMIControlCursorHoldPD, self).load_decoder()
        old_filt = self.decoder.filt
        filt = CursorPDControlKF(A=old_filt.A, W=old_filt.W, C=old_filt.C, Q=old_filt.Q)
        filt.C_xpose_Q_inv_C = old_filt.C_xpose_Q_inv_C
        filt.C_xpose_Q_inv = old_filt.C_xpose_Q_inv

        self.decoder.filt = filt

        self.decoder.filt.displ_mean = self.displ_mean
        self.decoder.filt.displ_scale = self.displ_scale


    def init(self):
        self.add_dtype('switch_W', 'f8', (1,))
        super(BMIControlCursorHoldPD, self).init()


    def _cycle(self):
        self.task_data['switch_W'] = self.decoder.filt.switch_W
        super(BMIControlCursorHoldPD, self)._cycle()

    @staticmethod
    def centerout_2d_discrete_pd_switching(n_switches=10, ntargets=8,
        distance=10, n_blocks_between_switches=6, metablock_pseudorand_n=4):
        '''
        '''
        r_scale_vals_by_block = []
        for k in range(n_switches):
            M = metablock_pseudorand_n/2
            data = [True]*M + [False]*M
            np.random.shuffle(data)
            r_scale_vals_by_block += data

        data = []

        for r_ in r_scale_vals_by_block:
            trial_target_seqs = BMIControlCursorHoldPD.centerout_2D_discrete(distance=distance, ntargets=ntargets, nblocks=n_blocks_between_switches)
            for tt in trial_target_seqs:
                data_tt = dict(targets=tt, switch_W=r_)
                data.append(data_tt)
        return data

    def _parse_next_trial(self):
        self.targs = self.next_trial['targets']
        sw = self.next_trial['switch_W']
        if sw and not self.decoder.filt.switch_W:
            print "switching on"
        elif self.decoder.filt.switch_W and not sw:
            print "switching off"

        self.decoder.filt.switch_W = sw


    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super(BMIControlCursorHoldPD, self).update_report_stats()
        self.reportstats['switching'] = int(self.decoder.filt.switch_W)

