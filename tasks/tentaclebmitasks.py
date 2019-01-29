'''
BMI tasks in the new structure, i.e. inheriting from manualcontrolmultitasks
'''
from __future__ import division

import manualcontrolmultitasks
import bmimultitasks

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
from riglib.stereo_opengl.xfm import Quaternion
from riglib.bmi.bmi import GaussianStateHMM, Decoder, GaussianState, BMISystem

from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere
import pickle

np.set_printoptions(suppress=False)

import numpy as np
from bmimultitasks import BMIControlMulti
from cursor_clda_tasks import CLDAControlMulti
from riglib.experiment import traits
from riglib import bmi
from manualcontrolmultitasks import VirtualCircularTarget, RED, ManualControlMulti
from riglib.plants import RobotArmGen2D

RED = (1., 0, 0, 1)
BLUE = (0, 0, 1., 1)

################
####### Learners
################
from riglib.bmi.clda import OFCLearner, Learner, RegexKeyDict
from riglib.bmi import feedback_controllers
class OFCLearnerTentacle(OFCLearner):
    '''    Docstring    '''
    def __init__(self, batch_size, A, B, Q, R, *args, **kwargs):
        '''    Docstring    '''
        F = feedback_controllers.LQRController.dlqr(A, B, Q, R)
        F_dict = RegexKeyDict()
        # F_dict['target'] = F
        # F_dict['hold'] = F
        F_dict['.*'] = F
        super(OFCLearnerTentacle, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

class TentacleValueLearner(Learner):
    _mean = 24.5
    _mean_alpha = 0.99
    def __init__(self, *args, **kwargs):
        if 'kin_chain' not in kwargs:
            raise ValueError("kin_chain object must specified for TentacleValueLearner!")
        self.kin_chain = kwargs.pop('kin_chain')
        super(TentacleValueLearner, self).__init__(*args, **kwargs)

        dt = 0.1
        use_tau_unNat = 2.7
        tau = use_tau_unNat
        tau_scale = 28*use_tau_unNat/1000
        bin_num_ms = (dt/0.001)
        w_r = 3*tau_scale**2/2*(bin_num_ms)**2*26.61

        I = np.eye(3)
        zero_col = np.zeros([3, 1])
        zero_row = np.zeros([1, 3])
        zero = np.zeros([1,1])
        one = np.ones([1,1])
        A = self.A = np.bmat([[I, dt*I, zero_col], 
                     [0*I, 0*I, zero_col], 
                     [zero_row, zero_row, one]])
        B = self.B = np.bmat([[0*I], 
                     [dt/1e-3 * I],
                     [zero_row]])
        Q = self.Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = self.R = np.mat(np.diag([w_r, w_r, w_r]))

        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_int_kin(self, current_state, target_state, decoder_output, task_state, state_order=None):
        '''
        This method of intention estimation just uses the subject's output 
        '''
        return decoder_output.reshape(-1,1)

    def calc_value(self, current_state, target_state, decoder_output, task_state, state_order=None, horizon=10, **kwargs):
        '''
        Determine the 'value' of a tentacle movement (4-link arm) 
        '''
        current_state = np.array(current_state).ravel()
        target_state = np.array(target_state).ravel()
        
        joint_pos = current_state[:self.kin_chain.n_links]
        endpt_pos = self.kin_chain.endpoint_pos(joint_pos)
        J = self.kin_chain.jacobian(-joint_pos)
        joint_vel = current_state[4:8] ### TODO remove hardcoding
        endpt_vel = np.dot(J, joint_vel)
        current_state_endpt = np.hstack([endpt_pos, endpt_vel[0], 0, endpt_vel[1], 1])

        target_pos = self.kin_chain.endpoint_pos(target_state[:self.kin_chain.n_links])
        target_vel = np.zeros(len(target_pos))
        target_state_endpt = np.hstack([target_pos, target_vel, 1])


        current_state = current_state_endpt
        target_state = target_state_endpt
        current_state = np.mat(current_state).reshape(-1,1)
        target_state = np.mat(target_state).reshape(-1,1)

        F = self.F
        A = self.A 
        B = self.B
        Q = self.Q
        R = self.R

        cost = 0
        for k in range(horizon):
            u = F*(target_state - current_state)
            m = current_state - target_state
            cost += (m.T * Q * m + u.T*0*u)[0,0]
            current_state = A*current_state + B*u
        return cost

    def postproc_value(self, values):
        values = np.hstack([-np.inf, values])
        value_diff = values[:-1] - values[1:]
        value_diff[value_diff < 0] = 0
        self._mean = self._mean_alpha*self._mean + (1-self._mean_alpha)*np.mean(value_diff[value_diff > 0])
        value_diff /= self._mean
        return value_diff

    def get_batch(self):
        '''
        see Learner.get_batch for documentation
        '''
        kindata = np.hstack(self.kindata)
        neuraldata = np.hstack(self.neuraldata)
        obs_value = np.hstack(self.obs_value)
        obs_value = self.postproc_value(obs_value)

        self.reset()
        return dict(intended_kin=kindata, spike_counts=neuraldata, value=obs_value)


##############
######## Tasks
##############

class CLDAControlTentacle(CLDAControlMulti):
    def load_decoder(self):
        super(CLDAControlTentacle, self).load_decoder()
        self.batch_time = self.decoder.binlen

    def create_updater(self):
        half_life_start, half_life_end = self.half_life
        self.updater = clda.KFRML(self.batch_time, half_life_start)

    def create_learner(self):
        A, B, _ = self.decoder.ssm.get_ssm_matrices(update_rate=self.decoder.binlen)

        kin_chain = self.plant.kin_chain
        Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros(5)])))
        R = 10000*np.mat(np.eye(B.shape[1]))
        self.learner = OFCLearnerTentacle(self.batch_size, A, B, Q, R)

class CLDAControlBaselineReestimate(CLDAControlTentacle):
    def create_updater(self):
        half_life_start, half_life_end = self.half_life
        self.updater = clda.KFRML_baseline(self.batch_time, half_life_start)

class CLDAControlTentacleTrialBased(CLDAControlTentacle):
    def create_learner(self):
        A, B, _ = self.decoder.ssm.get_ssm_matrices()

        kin_chain = self.plant.kin_chain
        Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros(len(kin_chain.link_lengths)+1)])))
        R = 100000*np.mat(np.eye(B.shape[1]))
        batch_size = int(self.batch_time/self.decoder.binlen)
        self.learner = OFCLearnerTentacle(batch_size, A, B, Q, R, done_states=['reward', 'hold_penalty'], reset_states=['timeout_penalty'])

    def _cycle(self):
        super(CLDAControlTentacleTrialBased, self)._cycle()
        if self.calc_state_occurrences('reward') > 16:
            self.learner.batch_size = np.inf

class CLDATentacleRL(CLDAControlTentacle):
    n_trial_ofc_learner = traits.Float(16.0, desc='Number of rewards before switching from continuous adaptation to trial-based adaptation')
    batch_size = 0.1
    def create_learner(self):
        A, B, _ = self.decoder.ssm.get_ssm_matrices()

        kin_chain = self.plant.kin_chain
        Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros(5)])))
        R = 100000*np.mat(np.eye(B.shape[1]))
        self.ofc_learner = OFCLearnerTentacle(1, A, B, Q, R, done_states=['reward', 'hold_penalty'], reset_states=['timeout_penalty'])
        self.rl_learner = TentacleValueLearner(np.inf, done_states=['reward', 'hold_penalty'], reset_states=['timeout_penalty'], kin_chain=self.plant.kin_chain)
        self.learner = self.ofc_learner

    def _cycle(self):
        super(CLDATentacleRL, self)._cycle()
        if (self.calc_state_occurrences('reward') > self.n_trial_ofc_learner) and self.state not in ['reward', 'hold_penalty']:
            # switch learner to the trial-based learner
            self.bmi_system.learner = self.rl_learner

class CLDAControlTentaclePPF(CLDAControlTentacle):
    param_noise_scale = traits.Float(1.0, desc="Stuff")
    def create_updater(self):
        vel_gain = 1e-8
        # vel_gain *= self.param_noise_scale
        const_var = 1e-4*0.06/50
        vel_var = vel_gain*0.13
        param_noise_variances = np.array([vel_var/225, vel_var/225, vel_var/5, vel_var/5, const_var])
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder, param_noise_variances=param_noise_variances)

class BMIControlMultiTentacleAttractor(BMIControlMulti):
    def init(self):
        self.timeout_count = 0
        super(BMIControlMultiTentacleAttractor, self).init()

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine what the target state of the task is
        '''
        target_loc = np.zeros(3) 
        ## The line above is the only change between this task and the BMIControlMulti task

        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoalCached):
            task_eps = np.inf
        else:
            task_eps = 0.5
        ik_eps = task_eps/10
        data, solution_updated = self.goal_calculator(target_loc, verbose=False, n_particles=500, eps=ik_eps, n_iter=10, q_start=-self.plant.get_intrinsic_coordinates())
        target_state, error = data

        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoal) and error > task_eps and solution_updated:
            self.goal_calculator.reset()

        return np.tile(np.array(target_state).reshape(-1,1), [1, self.decoder.n_subbins])

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super(BMIControlMultiTentacleAttractor, self).update_report_stats()
        self.reportstats['Timeout counter'] = self.timeout_count
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120), decimals=2)

    def _start_timeout_penalty(self):
        super(BMIControlMultiTentacleAttractor, self)._start_timeout_penalty()
        self.timeout_count += 1

class BMIJointPerturb(BMIControlMulti):
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

    sequence_generators = BMIControlMulti.sequence_generators + ['tentacle_multi_start_config']

    pert_angles = traits.Tuple((np.pi, -3*np.pi/4, 3*np.pi/4), desc="Possible wrist angles for perturbed configurations")
    # pert_angles = traits.Tuple((np.pi, -7*np.pi/8, 7*np.pi/8, -3*np.pi/4, 3*np.pi/4), desc="Possible wrist angles for perturbed configurations")
    # pert_angles = traits.Tuple((-3*np.pi/4), desc="Possible wrist angles for perturbed configurations")
    #np.pi,
    #tuple(np.linspace(-np.pi, np.pi, 8)
    premove_time = traits.Float(.1, desc='Time before subject must start doing BMI control')
    # static_states = ['premove'] # states in which the decoder is not run

    def __init__(self, *args, **kwargs):
        super(BMIJointPerturb, self).__init__(*args, **kwargs)

    def _parse_next_trial(self):
        self.targs, self.curr_pert_angle = self.next_trial
        # self.targs, self.curr_pert_angle, self.arm_visible = self.next_trial

    # def _start_wait(self):
    #     self.curr_pert_angle = self.pert_angles[np.random.randint(0, high=len(self.pert_angles))]
    #     super(BMIJointPerturb, self)._start_wait()
    
    def _while_premove(self):
        self.decoder.filt.state.mean = self.calc_perturbed_ik(self.targs[0])

    def _end_timeout_penalty(self):
        pass

    def calc_perturbed_ik(self, endpoint_pos):
        distal_angles = np.array([self.curr_pert_angle, -np.pi/20])
        # second angle above used to be self.init_decoder_mean[3,0] for center out version of task
        joints = self.plant.perform_ik(endpoint_pos, distal_angles=-distal_angles)
        return np.mat(np.hstack([joints, np.zeros(4), 1]).reshape(-1,1))

    def _test_premove_complete(self, ts):
        return ts>=self.premove_time

    def _test_hold_complete(self,ts):
        if self.target_index==0:
            return True
        else:
            return ts>=self.hold_time

    def _test_trial_incomplete(self, ts):
        return (self.target_index<self.chain_length-1) and (self.target_index != -1) and (self.tries<self.max_attempts)

    def _test_trial_restart(self, ts):
        return (self.target_index==-1) and (self.tries<self.max_attempts)

    @staticmethod 
    def tentacle_multi_start_config(nblocks=100, ntargets=4, distance=8, startangle=45):
        elbow_angles = np.array([135, 180, 225])*np.pi/180 # TODO make this a function argument!
        startangle = 45 * np.pi/180
        n_configs_per_target = len(elbow_angles)
        target_angles = np.arange(startangle, startangle+(2*np.pi), 2*np.pi/ntargets)
        targets = distance*np.vstack([np.cos(target_angles), 0*target_angles, np.sin(target_angles)])

        seq = []
        from itertools import izip
        import random
        for i in range(nblocks):
            target_inds = np.tile(np.arange(ntargets), (n_configs_per_target, 1)).T.ravel()
            config_inds = np.tile(np.arange(n_configs_per_target), ntargets)

            sub_seq = []
            inds = np.arange(n_configs_per_target*ntargets)
            random.shuffle(inds)
            for k in inds:
                targ_ind = target_inds[k]
                config_ind = config_inds[k]

                seq_item = (np.vstack([targets[:, targ_ind], np.zeros(3)]), elbow_angles[config_ind])
                seq.append(seq_item)

        return seq

class VirtualKinChainWithToolLink(RobotArmGen2D):
	def _pickle_init(self):
		super(VirtualKinChainWithToolLink, self)._pickle_init()

		self.tool_tip_cursor = Sphere(radius=self.link_radii[-1]/2, color=RED)
		self.tool_base_cursor = Sphere(radius=self.link_radii[-1]/2, color=BLUE)

		self.graphics_models = [self.link_groups[0], self.tool_tip_cursor, self.tool_base_cursor]

	def _update_link_graphics(self):
		super(VirtualKinChainWithToolLink, self)._update_link_graphics()

		joint_angles = self.calc_joint_angles()
		spatial_joint_pos = self.kin_chain.spatial_positions_of_joints(joint_angles)
		self.tool_tip_cursor.translate(*spatial_joint_pos[:,-1], reset=True)
		self.tool_base_cursor.translate(*spatial_joint_pos[:,-2], reset=True)

class BMIControlTentacleOrientation(bmimultitasks.BMIControlMulti):
	tool_tip_target_radius = traits.Float(2.0, desc='Target radius for the endpoint')
	tool_base_target_radius = traits.Float(2.0, desc='Target radius for the 2nd to last joint')

class CLDATentacleStableSubpopulation(CLDAControlTentacleTrialBased):
    stable_decoder = traits.Instance(bmi.bmi.Decoder) 

    def load_decoder(self):
        super(CLDATentacleStableSubpopulation, self).load_decoder()
        if not np.array_equal(self.stable_decoder.units, self.decoder.units):
            print "updating seed decoder with stable parameters"
            stable_units = map(tuple, self.stable_decoder.units)
            full_units = map(tuple, self.decoder.units)

            inds = [full_units.index(unit) for unit in stable_units]

            # edit the seed decoder (self.decoder) with the stable parameters
            self.decoder.filt.C_xpose_Q_inv_C = self.stable_decoder.filt.C_xpose_Q_inv_C
            self.decoder.filt.C[inds, :] = self.stable_decoder.filt.C
            self.decoder.filt.Q[np.ix_(inds, inds)] = self.stable_decoder.filt.Q

            # edit the sufficient stats for the seed decoder parameters to use the stable parameters when appropriate
            self.decoder.filt.R = self.stable_decoder.filt.R
            self.decoder.filt.S[inds,:] = self.stable_decoder.filt.S
            self.decoder.filt.T[np.ix_(inds, inds)] = self.stable_decoder.filt.T

            self.stable_units = stable_units
            self.stable_unit_inds = inds

    def create_updater(self):
        half_life_start, half_life_end = self.half_life
        self.updater = clda.KFRML(self.batch_time, half_life_start, adapt_C_xpose_Q_inv_C=False)
        self.updater.set_stable_inds(self.stable_unit_inds, stable_inds_independent=True)


class TentacleMultiConfig(BMIJointPerturb):
    sequence_generators = ['manually_spec_config_target']

    @staticmethod
    def manually_spec_config_target(nblocks=100):
        '''
        Read the list of starting configurations and targets from a file
        '''
        import pickle
        configurations, target_locations = pickle.load(open('/storage/task_data/TentacleMultiConfig/starting_config_data.pkl'))

        from riglib.bmi import robot_arms
        from itertools import izip
        kin_chain = robot_arms.PlanarXZKinematicChain(link_lengths=[15, 15, 5, 5])
        reach_origins = [kin_chain.endpoint_pos(config) for config in configurations]
        targs = [np.vstack([origin, terminus]) for origin, terminus in izip(reach_origins, target_locations)]

        n_trials_per_block = len(targs)
        data = []
        for k in range(nblocks):
            inds = np.arange(n_trials_per_block)
            np.random.shuffle(inds)
            for i in inds:
                data_i = (configurations[i], targs[i])
                data.append(data_i)

        return data

    def _parse_next_trial(self):
        self.starting_config, self.targs = self.next_trial

    def _while_premove(self):
        starting_state = np.hstack([self.starting_config, np.zeros_like(self.starting_config), 1]).reshape(-1,1)
        self.decoder.filt.state.mean = starting_state

class TentacleObstacleAvoidance(BMIControlMultiTentacleAttractor):
    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None, hit_obstacle="obstacle_penalty"),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        obstacle_penalty = dict(obstacle_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    obstacle_radius = traits.Float(2.0, desc='Radius of cylindrical obstacle')
    obstacle_penalty = traits.Float(0.0, desc='Penalty time if the chain hits the obstacle(s)')

    def __init__(self, *args, **kwargs):
        super(TentacleObstacleAvoidance, self).__init__(*args, **kwargs)

        ## Create an obstacle object, hidden by default
        self.obstacle = Sphere(radius=self.obstacle_radius + 0.6, color=(0, 0, 1, .5)) ##Cylinder(radius=self.obstacle_radius, height=1, color=(0,0,1,1))
        self.obstacle_on = False
        self.obstacle_pos = np.ones(3)*np.nan
        self.hit_obstacle = False

        self.add_model(self.obstacle)

    def init(self):
        self.add_dtype('obstacle_on','f8', (1,))
        self.add_dtype('obstacle_pos','f8', (3,))
        super(TentacleObstacleAvoidance, self).init()

    def _cycle(self):
        self.task_data['obstacle_on'] = self.obstacle_on
        self.task_data['obstacle_pos'] = self.obstacle_pos
        super(TentacleObstacleAvoidance, self)._cycle()

    def _start_target(self):
        super(TentacleObstacleAvoidance, self)._start_target()
        if self.target_index == 1:
            target_angle = np.round(np.rad2deg(np.arctan2(self.target_location[-1], self.target_location[0])))
            try:
                obstacle_data = pickle.load(open('/storage/task_data/TentacleObstacleAvoidance/center_out_obstacle_pos.pkl'))
                self.obstacle_pos = obstacle_data[target_angle]
            except:
                self.obstacle_pos = (self.target_location/2)
            self.obstacle.translate(*self.obstacle_pos, reset=True)
            self.obstacle.attach()
            self.obstacle_on = True

    def _test_obstacle_penalty_end(self, ts):
        return ts > self.obstacle_penalty
    
    def _start_obstacle_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _end_target(self):
        self.obstacle.detach()
        self.obstacle_on = False

    def _test_hit_obstacle(self, ts):
        if self.target_index == 1:
            joint_angles = self.plant.get_intrinsic_coordinates()
            distances_to_links = self.plant.kin_chain.detect_collision(joint_angles, self.obstacle_pos)

            hit = np.min(distances_to_links) < (self.obstacle_radius + self.plant.link_radii[0])
            if hit:
                self.hit_obstacle = True
                return True
        else:
            return False

    # @staticmethod
    # def centerout_2D_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
    #     distance=10):
    #     '''

    #     Generates a sequence of 2D (x and z) target pairs with the first target
    #     always at the origin.

    #     Parameters
    #     ----------
    #     length : int
    #         The number of target pairs in the sequence.
    #     boundaries: 6 element Tuple
    #         The limits of the allowed target locations (-x, x, -z, z)
    #     distance : float
    #         The distance in cm between the targets in a pair.

    #     Returns
    #     -------
    #     pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations

    #     '''
    #     # Choose a random sequence of points on the edge of a circle of radius 
    #     # "distance"
        
    #     theta = []
    #     for i in range(nblocks):
    #         temp = np.deg2rad(np.array([0., 90, 135, 180, 225, 270, 315]))
    #         # temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
    #         np.random.shuffle(temp)
    #         theta = theta + [temp]
    #     theta = np.hstack(theta)


    #     x = distance*np.cos(theta)
    #     y = np.zeros(len(theta))
    #     z = distance*np.sin(theta)
        
    #     pairs = np.zeros([len(theta), 2, 3])
    #     pairs[:,1,:] = np.vstack([x, y, z]).T
        
    #     return pairs            

class TentacleMultiConfigObstacleAvoidance(BMIJointPerturb):
    status = dict(
        wait = dict(start_trial="premove", stop=None),
        premove=dict(premove_complete="target"),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None, hit_obstacle="obstacle_penalty"),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", trial_restart="premove"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        obstacle_penalty = dict(obstacle_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )

    obstacle_radius = traits.Float(2.0, desc='Radius of cylindrical obstacle')
    obstacle_penalty = traits.Float(0.0, desc='Penalty time if the chain hits the obstacle(s)')

    def __init__(self, *args, **kwargs):
        super(TentacleMultiConfigObstacleAvoidance, self).__init__(*args, **kwargs)

        ## Create an obstacle object, hidden by default
        self.obstacle = Sphere(radius=self.obstacle_radius + 0.6, color=(0, 0, 1, .5))
        self.obstacle_on = False
        self.obstacle_pos = np.ones(3)*np.nan
        self.hit_obstacle = False

        self.add_model(self.obstacle)

    def init(self):
        self.add_dtype('obstacle_on','f8', (1,))
        self.add_dtype('obstacle_pos','f8', (3,))
        super(TentacleMultiConfigObstacleAvoidance, self).init()

    def _cycle(self):
        self.task_data['obstacle_on'] = self.obstacle_on
        self.task_data['obstacle_pos'] = self.obstacle_pos
        super(TentacleMultiConfigObstacleAvoidance, self)._cycle()

    def _start_target(self):
        super(TentacleMultiConfigObstacleAvoidance, self)._start_target()
        if self.target_index == 1:
            self.obstacle_pos = (self.targs[0]/2)
            self.obstacle.translate(*self.obstacle_pos, reset=True)
            self.obstacle.attach()
            self.obstacle_on = True

    def _test_obstacle_penalty_end(self, ts):
        return ts > self.obstacle_penalty
    
    def _start_obstacle_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _end_target(self):
        self.obstacle.detach()
        self.obstacle_on = False

    def _test_hit_obstacle(self, ts):
        if self.target_index == 1:
            joint_angles = self.plant.get_intrinsic_coordinates()
            distances_to_links = self.plant.kin_chain.detect_collision(joint_angles, self.obstacle_pos)

            hit = np.min(distances_to_links) < (self.obstacle_radius + self.plant.link_radii[0])
            if hit:
                self.hit_obstacle = True
                return True
        else:
            return False


    @staticmethod 
    def tentacle_multi_start_config(nblocks=100, ntargets=4, distance=8, startangle=45):
        elbow_angles = np.array([135, 180, 225])*np.pi/180 # TODO make this a function argument!
        startangle = 45 * np.pi/180
        n_configs_per_target = len(elbow_angles)
        target_angles = np.arange(startangle, startangle+(2*np.pi), 2*np.pi/ntargets)
        targets = distance*np.vstack([np.cos(target_angles), 0*target_angles, np.sin(target_angles)])

        seq = []
        from itertools import izip
        import random
        for i in range(nblocks):
            target_inds = np.tile(np.arange(ntargets), (n_configs_per_target, 1)).T.ravel()
            config_inds = np.tile(np.arange(n_configs_per_target), ntargets)

            sub_seq = []
            inds = np.arange(n_configs_per_target*ntargets)
            random.shuffle(inds)
            for k in inds:
                targ_ind = target_inds[k]
                config_ind = config_inds[k]

                seq_item = (np.vstack([targets[:, targ_ind], np.zeros(3)]), elbow_angles[config_ind])
                seq.append(seq_item)

        return seq            

class TentacleCenterOutOrderChoice(BMIControlMultiTentacleAttractor):
    # status = dict(
    #     wait = dict(start_trial="target", stop=None),
    #     target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
    #     hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
    #     targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target"),
    #     timeout_penalty = dict(timeout_penalty_end="targ_transition"),
    #     hold_penalty = dict(hold_penalty_end="targ_transition"),
    #     reward = dict(reward_end="wait")
    # )
    periph_targ_radius = traits.Float(8, desc='Workspace radius')
    def __init__(self, *args, **kwargs):
        kwargs['instantiate_targets'] = False
        super(TentacleCenterOutOrderChoice, self).__init__(*args, **kwargs)
        self.targets = []
        self.n_targets = 8

        target_angles = np.arange(self.n_targets)*2*np.pi/self.n_targets
        self.target_locations = np.vstack([np.cos(target_angles), np.zeros_like(target_angles), np.sin(target_angles)]).T * self.periph_targ_radius

        for k in range(self.n_targets):
            targ = VirtualCircularTarget(target_radius=self.target_radius, target_color=RED)
            self.targets.append(targ)

        self.center_target = VirtualCircularTarget(target_radius=self.target_radius, target_color=RED)
        # self.targets.append(self.center_target)

        for target in self.targets:
            for model in target.graphics_models:
                self.add_model(model)

        for model in self.center_target.graphics_models:
            self.add_model(model)

    def init(self):
        super(TentacleCenterOutOrderChoice, self).init()

        print self.target_locations
        # print np.rad2deg(self.target_angles)

        # initialize all the targets
        for k in range(self.n_targets):
            self.targets[k].move_to_position(self.target_locations[k])

        self.center_target.move_to_position(np.zeros(3))

        self.target_available = np.ones(self.n_targets).astype(bool)

    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        if self.target_index <= 0:
            cursor_pos = self.plant.get_endpoint_pos()
            d = np.linalg.norm(cursor_pos - np.zeros(3))
            return d <= (self.target_radius - self.cursor_radius)
        else:
            cursor_pos = self.plant.get_endpoint_pos()
            d = np.array(map(np.linalg.norm, cursor_pos - self.target_locations))
            in_target = np.any(d[self.target_available] <= (self.target_radius - self.cursor_radius))
            return in_target

    # def _start_hold(self):
    #     #make next target visible unless this is the final target in the trial
    #     idx = (self.target_index + 1)
    #     if idx < self.chain_length: 
    #         target = self.targets[idx % 2]
    #         target.move_to_position(self.targs[idx])

    def _distances_to_targets(self):
        cursor_pos = self.plant.get_endpoint_pos()
        return np.array(map(np.linalg.norm, cursor_pos - self.target_locations))

    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        if self.target_index <= 0: # TODO  thought this was 0..
            return not np.linalg.norm(self.plant.get_endpoint_pos()) < (self.target_radius - self.cursor_radius)
        else:
            return not np.any(self._distances_to_targets() < (self.target_radius - self.cursor_radius))
        # cursor_pos = self.plant.get_endpoint_pos()
        # d = np.linalg.norm(cursor_pos - self.target_location)
        # rad = self.target_radius - self.cursor_radius
        # return d > rad

    def _start_hold(self):
        if self.target_index == 0:
            if not np.any(self.target_available):
                self.target_available[:] = True

            for k, target in enumerate(self.targets):
                if self.target_available[k]:
                    # set the target color to red
                    target.cue_trial_start()            
    
    def _end_hold(self):
        if self.target_index == 1:
            cursor_pos = self.plant.get_endpoint_pos()
            d = np.array(map(np.linalg.norm, cursor_pos - self.target_locations))
            target_idx = np.argmin(d)
            # change current target color to green
            # self.targets[target_idx].cue_trial_end_success()
            self.held_target_idx = target_idx

    def _start_target(self):
        self.target_index += 1

        if self.target_index == 0:
            self.center_target.cue_trial_start()
        elif self.target_index == 1:
            self.center_target.hide()
            for k, target in enumerate(self.targets):
                if self.target_available[k]:
                    # set the target color to red
                    target.cue_trial_start()            

    def _start_reward(self):
        self.targets[self.held_target_idx].cue_trial_end_success()
        self.target_available[self.held_target_idx] = False
        super(ManualControlMulti, self)._start_reward()
        self.targets[self.held_target_idx].show()

    ## FOR DEBUGGING ONLY!
    # def get_target_BMI_state(self, *args):
    #     '''
    #     Run the goal calculator to determine what the target state of the task is
    #     '''
    #     # 
    #     if self.target_index <= 0:
    #         target_loc = target_loc = np.zeros(3)
    #     else:
    #         try: # should only fail at the end of the block
    #             target_loc = self.target_locations[np.nonzero(self.target_available)[0][0]]
    #         except:
    #             target_loc = target_loc = np.zeros(3)   

    #     ## The line above is the only change between this task and the BMIControlMulti task

    #     if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoalCached):
    #         task_eps = np.inf
    #     else:
    #         task_eps = 0.5
    #     ik_eps = task_eps/10
    #     data, solution_updated = self.goal_calculator(target_loc, verbose=False, n_particles=500, eps=ik_eps, n_iter=10, q_start=-self.plant.get_intrinsic_coordinates())
    #     target_state, error = data

    #     if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoal) and error > task_eps and solution_updated:
    #         self.goal_calculator.reset()

    #     return np.tile(np.array(target_state).reshape(-1,1), [1, self.decoder.n_subbins])
