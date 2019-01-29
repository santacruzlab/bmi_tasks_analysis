'''
Tasks to create and run a BMI controlling a point mass
'''
import numpy as np
from riglib.bmi.state_space_models import State, StateSpace, offset_state
from riglib.bmi.assist import FeedbackControllerAssist
from riglib.bmi.goal_calculators import ZeroVelocityAccelGoal
from riglib.bmi.bmi import Decoder
from riglib.bmi.clda import OFCLearner, Learner, RegexKeyDict
from riglib.bmi import feedback_controllers, clda, robot_arms

from riglib import plants
from passivetasks import EndPostureFeedbackController, MachineOnlyFilter
from bmimultitasks import BMIControlMulti
from cursor_clda_tasks import CLDAControlMulti
import os
from riglib.bmi.extractor import DummyExtractor

from features import simulation_features
from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife
from riglib.experiment import traits

from riglib.stereo_opengl.primitives import Cylinder, Sphere, Cone, Cube, Chain

# import ode
from itertools import izip
import pdb

arm_color = (181/256., 116/256., 96/256., 1)
arm_radius = 0.6
pi = np.pi
cm_to_m = 0.01

#############################################
##### Joint Torque control of Planar arm ####
#############################################

##### Plant #####
class VirtualPlanarArm(plants.RobotArmGen2D):
    '''
    Arm with mass/inertia/friction modeled from morphometric data (Cheng and Scott, 2000) and controlled with joint torque commands
    '''
    def __init__(self, body_weight=6.5, base_loc=np.array([2., 0., -15])):
        '''
        body_weight : float
            Entire weight of subject in kg. Used to determine the mass of the arm segements using regression model from Cheng & Scott 2000, p221, Table 8
        '''

        # Initialize the dynamics world
        # world = ode.World() # TODO this should be part of the window, for object collision detection stuff
        # world.setGravity((0,0,0))


        # Arm link lengths----from monkey P? Numbers taken from MATLAB code originally written by Rodolphe Heliot/Amy Orsborn
        # Changed to these values because most common among old code. Cheng & Scott 2000 Table 2 has other values if change needed
        self.l_ua = 17.70 # cm
        self.l_fa = 20.35 # cm

        # Friction coefficients
        self.B = np.mat([[0.03, 0.01],
                    [0.01, 0.03]])

        # Mass of upperarm/forearm from Cheng & Scott, 2000, p221, Table 8 (based on morphometric data)
        self.m_ua = 0.001*(23 + 34.4*body_weight)
        self.m_fa = 0.001*(53 + 25.2*body_weight)

        ## Determine the inertia of each segment
        rad_of_gyration = np.array([0.247, 0.248]) # relative to the length of each segment

        # Calculate center of mass for each segment
        self.ctr_of_mass_ua = self.l_ua * rad_of_gyration[0]
        self.ctr_of_mass_fa = self.l_fa * rad_of_gyration[1]
        self.r_ua = self.ctr_of_mass_ua
        self.r_fa = self.ctr_of_mass_fa


        # Calculate moment of inertia for each segment 
        # i = 0.001 * 0.0001 * (b + m*total_body_weight), where 'b' and 'm' are from Cheng & Scott 2000, p221, Table 8
        #     0.001 * 0.0001 converts (g cm^2) ==> (kg m^2) 
        self.I_ua = 0.001*0.0001*(432 + 356.6*body_weight)
        self.I_fa = 0.001*0.0001*(2381 + 861.6*body_weight)

        self.rad_ua = np.sqrt(4*(self.I_ua/self.m_ua - 1./3*(0.01*self.l_ua)**2))
        self.rad_fa = np.sqrt(4*(self.I_fa/self.m_fa - 1./3*(0.01*self.l_fa)**2))


        # Create two bodies
        # upper_arm = ode.Body(world)
        # M = ode.Mass()
        # M.setCylinderTotal(total_mass=100*self.m_ua, direction=1, r=3, h=self.l_ua)
        # upper_arm.setMass(M)
        # upper_arm.setPosition(base_loc + np.array([self.l_ua, 0, 0]))

        # forearm = ode.Body(world)
        # M = ode.Mass()
        # M.setCylinderTotal(total_mass=100*self.m_fa, direction=1, r=3, h=self.l_fa)
        # forearm.setMass(M)
        # forearm.setPosition(base_loc + np.array([self.l_ua + self.l_fa, 0, 0]))

        # # anchor upper_arm to the world at position (2., 0, -15)
        # j1 = ode.HingeJoint(world)
        # j1.attach(upper_arm, ode.environment)
        # j1.setAnchor(base_loc)
        # j1.setAxis((0, 1, 0))


        # # anchor forearm to the distal end of upper_arm
        # j2 = ode.HingeJoint(world)
        # j2.attach(upper_arm, forearm)
        # j2.setAnchor(base_loc + np.array([self.l_ua, 0, 0]))
        # j2.setAxis((0, 1, 0))


        # self.arm_segments = [upper_arm, forearm]
        # self.joints = [j1, j2]
        # self.world = world

        # self.num_joints = len(self.joints)

        self.theta = np.array([ 0.38118002,  2.08145271])# angular
        self.omega = np.zeros(2) # angular vel 
        self.alpha = np.zeros(2) # angular acc
        self.torque = np.zeros(2)

        self.base_loc = base_loc
        
        link_radii=arm_radius
        joint_radii=arm_radius
        joint_colors=arm_color
        link_colors=arm_color

        self.link_lengths = link_lengths = [self.l_ua, self.l_fa]
        self.num_joints = len(self.link_lengths)
        self.curr_vecs = np.zeros([self.num_joints, 3]) #rows go from proximal to distal links

        self.chain = Chain(link_radii, joint_radii, link_lengths, joint_colors, link_colors)
        self.cursor = Sphere(radius=arm_radius/2, color=link_colors)
        self.graphics_models = [self.chain.link_groups[0], self.cursor]
        self.chain.translate(*self.base_loc, reset=True)

        self.kin_chain = robot_arms.PlanarXZKinematicChain2Link(link_lengths, base_loc=base_loc)
        
        self.stay_on_screen = False
        self.visible = True


    def drive(self, decoder):

        torque = decoder['torque_0', 'torque_1']
        self._handle_torque(torque)

        decoder['theta_0', 'theta_1'] = self.theta
        decoder['omega_0', 'omega_1'] = self.omega
        decoder['alpha_0', 'alpha_1'] = self.alpha

    # def _handle_torque(self, torque):
    #     ''' Handle torque using PyODE '''
    #     print torque
    #     self.world.step(1./60)
    #     for j, tau, body in izip(self.joints, torque, self.arm_segments):
    #         j.addTorque(tau)

    #     self.theta = [j.getAngle() for j in self.joints]
    #     self.omega = [j.getAngleRate() for j in self.joints]

    #     self.set_intrinsic_coordinates(self.theta)

    def calc_inertia(self, theta):
        '''
        Calculate the inertial and coriolis force matrices
        ua = upper arm, fa = forearm
        Equations from "A mathematical introduction to robotic manipulation" by Murray, Li, Sastry, p. 164

        theta : np.array of shape (2,)
            Joint angles
        omega : np.array of shape (2,)
            Joint velocities
        '''
        l_ua = self.l_ua * 0.01 # convert to m
        l_fa = self.l_fa * 0.01 # convert to m
        r_ua = self.r_ua * 0.01 # convert to m
        r_fa = self.r_fa * 0.01 # convert to m

        alpha = self.I_ua + self.I_fa + self.m_ua*r_ua**2 + self.m_fa*(l_ua**2 + r_fa**2)
        beta = self.m_fa*l_ua*r_fa
        delta = self.I_fa + self.m_fa*r_fa**2

        cos = np.cos
        sin = np.sin
        I = np.mat([[alpha + 2*beta*cos(theta[1]),  delta + beta*cos(theta[1])], 
                    [delta + beta*cos(theta[1]),    delta                     ]])         

        return I

    def calc_IN(self, theta, omega):
        '''
        Calculate the inertial and coriolis force matrices
        ua = upper arm, fa = forearm
        Equations from "A mathematical introduction to robotic manipulation" by Murray, Li, Sastry, p. 164

        theta : np.array of shape (2,)
            Joint angles
        omega : np.array of shape (2,)
            Joint velocities
        '''
        l_ua = self.l_ua * 0.01 # convert to m
        l_fa = self.l_fa * 0.01 # convert to m
        r_ua = self.r_ua * 0.01 # convert to m
        r_fa = self.r_fa * 0.01 # convert to m

        alpha = self.I_ua + self.I_fa + self.m_ua*r_ua**2 + self.m_fa*(l_ua**2 + r_fa**2)
        beta = self.m_fa*l_ua*r_fa
        delta = self.I_fa + self.m_fa*r_fa**2

        cos = np.cos
        sin = np.sin
        I = np.mat([[alpha + 2*beta*cos(theta[1]),  delta + beta*cos(theta[1])], 
                    [delta + beta*cos(theta[1]),    delta                     ]]) 

        N = np.mat([[-beta*sin(theta[1])*omega[1],  -beta*sin(theta[1])*sum(omega)],
                    [beta*sin(theta[1])*omega[0],   0                              ]])
        return I, N

    def _handle_torque(self, torque):
        # print torque
        torque = np.mat(torque).reshape(-1,1)
        Inertia, N = self.calc_IN(self.theta, self.omega)
        theta = np.mat(self.theta.reshape(-1, 1))
        omega = np.mat(self.omega.reshape(-1, 1))

        # B = np.mat([[0.03, 0.01], [0.01, 0.03]]) * 5
        # inv_Inertia = np.linalg.inv(Inertia + 1*np.mat(np.eye(2)))
        B = np.mat([[0.03, 0.01], [0.01, 0.03]])
        inv_Inertia = np.linalg.inv(Inertia)

        self.alpha = inv_Inertia * (torque - N*omega - B*omega)

        DT = 1./60

        self.theta = np.array(theta + DT*omega + DT**2 * self.alpha).ravel()
        self.omega = np.array(omega + DT*self.alpha).ravel()

        # joint limits between (0, pi)
        # self.theta[0] = max(min(self.theta[0], 0.99*np.pi), -pi/10)
        # self.theta[1] = max(min(self.theta[1], 0.75*np.pi), -pi/10)

        self.set_intrinsic_coordinates(self.theta)

    def get_endpoint_pos(self):
        return self.kin_chain.endpoint_pos(self.theta) #arm_segments[1].getPosition()

    def get_data_to_save(self):
        return dict()

##### State space #####
class StateSpaceNLinkJointTorque(StateSpace):
    def __init__(self, n_links=2):
        self.states = []
        self.n_links = n_links
        for k in range(n_links):
            theta_k = State('theta_%d' % k, stochastic=False,  drives_obs=False, order=0, aux=True)
            self.states.append(theta_k)

        for k in range(n_links):
            omega_k = State('omega_%d' % k, stochastic=False,  drives_obs=False, order=1, aux=True)
            self.states.append(omega_k)

        for k in range(n_links):            
            alpha_k = State('alpha_%d' % k, stochastic=False,  drives_obs=False, order=2, aux=True)
            self.states.append(alpha_k)

        for k in range(n_links):
            torque_k = State('torque_%d' % k, stochastic=True,  drives_obs=True, order=2)
            self.states.append(torque_k)

            # self.states += [theta_k, omega_k, alpha_k, torque_k]

        self.states.append(offset_state)

        self.inertia = np.mat(np.eye(n_links))

    def get_ssm_matrices(self, update_rate=0.1):
        I = np.mat(np.eye(self.n_links))
        Delta = update_rate
        zero_vec = np.zeros([self.n_links,1])
        A = np.vstack([np.hstack([I,   Delta*I, 0.5*Delta**2*I, 0*I,   zero_vec]), 
                       np.hstack([0*I, I,       Delta*I,        0*I,   zero_vec]), 
                       np.hstack([0*I, 0*I,     0*I,            0*I,   zero_vec]),
                       np.hstack([0*I, 0*I,     0*I,            0*I, zero_vec]),
                       np.hstack([zero_vec.T, zero_vec.T, zero_vec.T, zero_vec.T, np.ones([1,1])]),
                      ])

        B = np.vstack([0*I, 
                       0*I, 
                       Delta*1000*I, 
                       self.inertia*Delta*1000, 
                       zero_vec.T])

        W_torque = np.diag([14, 3.5]) * 0.1
        W_torque = np.diag(np.array([ 0.00044274,  0.00016395])) # from visual feedback data

        W = np.vstack([np.hstack([0*I, 0*I, 0*I, 0*I,   zero_vec]), 
                       np.hstack([0*I, 0*I, 0*I, 0*I,   zero_vec]), 
                       np.hstack([0*I, 0*I, 0*I, 0*I,   zero_vec]),
                       np.hstack([0*I, 0*I, 0*I, W_torque, zero_vec]),
                       np.hstack([zero_vec.T, zero_vec.T, zero_vec.T, zero_vec.T, np.zeros([1,1])]),
                      ]) 

        return A, B, W 
        # raise NotImplementedError("coriolis forces are actually state dependent...")

##### Feedback controller #####
class JointTorque2LController(feedback_controllers.PIDController):
    def __init__(self):
        I = np.mat(np.eye(3))
        Delta = 1./60
        self.ssm = StateSpaceNLinkJointTorque()
        # pdb.set_trace()
        super(JointTorque2LController, self).__init__(0.1, 0.02, 0, self.ssm.state_order)

    def calc_next_state(self, current_state, target_state, **kwargs):
        next_torque = super(JointTorque2LController, self).calc_next_state(current_state, target_state, **kwargs)
        ns = np.hstack([np.zeros(6), np.array(next_torque).ravel(), 1]).reshape(-1, 1)
        return ns

##### BMI task class #####
class JointTorqueBMI(BMIControlMulti):
    exclude_parent_traits = ['plant_type', 'plant_visible', 'plant_hide_rate']
    sequence_generators = BMIControlMulti.sequence_generators + ['centerout_2D_discrete_nobottom']
    def __init__(self, *args, **kwargs):
        self.plant = VirtualPlanarArm()
        super(JointTorqueBMI, self).__init__(*args, **kwargs)

    def init_decoder_state(self):
        pass

    def create_assister(self):
        fb_ctrl = JointTorque2LController()
        self.assister = FeedbackControllerAssist(fb_ctrl, style='mixing')

    def create_goal_calculator(self):
        pass

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        targ_joint_angles = self.plant.kin_chain.inverse_kinematics(self.target_location.copy()).ravel()
        target_state = np.array(np.hstack([targ_joint_angles, np.zeros(2), np.zeros(2), np.zeros(2), 1]))
        return np.tile(np.array(target_state).reshape(-1,1), [1, self.decoder.n_subbins])

    @staticmethod
    def centerout_2D_discrete_nobottom(nblocks=100, distance=10):
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

        Returns
        -------
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        pi = np.pi
        angles = np.array([0, pi/4, 2*pi/4, 3*pi/4, 4*pi/4, 5*pi/4, 7*pi/4])
        
        theta = []
        for i in range(nblocks):
            temp = angles.copy()
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs

##### Visual feedback task #####
class JointTorqueVisualFeedback(JointTorqueBMI):
    exclude_parent_traits = []
    assist_level = (1, 1)
    is_bmi_seed = True
    sequence_generators = ['centerout_2D_discrete_nobottom']

    def load_decoder(self):
        self.ssm = StateSpaceNLinkJointTorque()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()

##### CLDA task #####
class JointTorqueCLDA(JointTorqueBMI, LinearlyDecreasingHalfLife):
    batch_time = traits.Float(0.1, desc='The length of the batch in seconds')
    decoder_sequence = traits.String('test', desc='signifier to group together sequences of decoders')

    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        fb_ctrl = JointTorque2LController()
        self.learner = clda.FeedbackControllerLearner(self.batch_size, fb_ctrl)
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def _cycle(self):
        super(JointTorqueCLDA, self)._cycle()
        if self.calc_state_occurrences('reward') > 16:
            # print "switching to batch-mode updates!"
            self.learner.batch_size = np.inf

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(JointTorqueCLDA, self).call_decoder(*args, **kwargs)


#################################################
##### Muscle activation model (MAM) classes #####
#################################################
K_sh = 3.
f1 = 0.80
f2 = 0.50
f3 = 0.43
f4 = 58.


class Muscle(object):
    K_sh = 3.
    f1 = 0.80 # unitless
    f2 = 0.50 # unitless 
    f3 = 0.43 # unitless 
    f4 = 58.  # s/m

    def __init__(self, l0, F_max, theta_of, sh_moments_mat, el_moments_mat, name=''):
        '''
        name : string 
            Name of muscle, for referencing
        l0 : float
            Optimal fasicle length
        '''
        self.name = name
        self.F_max = F_max
        self.l0 = l0
        self.theta_of = np.array(theta_of).ravel()

        consts = np.array([1, 10**-2, 10**-4, 10**-5]) ## From Graham and Scott, 2003, table 3 (column headers)
        self.sh_moments_mat = np.array(sh_moments_mat).ravel() * consts
        self.el_moments_mat = np.array(el_moments_mat).ravel() * consts

        self.M_coeffs = np.vstack([self.sh_moments_mat, self.el_moments_mat])

        self.int_A_of = self.int_moment_arm(self.theta_of)

    def get_active_force(self, activ, l, dl):
        '''
        active : float 
            Muscle activation level 
        len : float 
            Musce length
        dl : float
            Muscle velocity
        '''
        F_g = activ * self.F_max * (1 - 4*((l - self.l0)/self.l0)**2)
        # make sure F_g is nonnegative
        # F_g = max(F_g, 0)
        F_a = F_g * (self.f1 + self.f2 * np.arctan(self.f3 + self.f4*dl))
        return F_a

    def active_force_length_gain(self, l):
        return (1 - 4*((l - self.l0)/self.l0)**2)

    def active_force_velocity_gain(self, dl):
        return (self.f1 + self.f2 * np.arctan(self.f3 + self.f4*dl))

    def get_passive_force(self, l):
        '''
        len : float 
            Musce length
        '''
        F_p = (self.F_max/(np.exp(self.K_sh) - 1)) * (np.exp((self.K_sh*(l - self.l0))/(0.5*self.l0)) - 1)
        F_p = max(F_p, 0)
        # return F_p
        return 0

    def moment_arm(self, theta):
        '''
        Calculate the mechanical advantage of the muscle at a particular joint configuration
        '''
        theta = np.rad2deg(theta)
        theta_hat = np.vstack([np.ones(2), theta, theta**2, theta**3])
        A = np.diag(np.dot(self.M_coeffs, theta_hat))
        return A        

    def int_moment_arm(self, theta):
        '''
        Calculate integrated moment arm, used for determining the length of the muscle
        '''
        theta = np.rad2deg(theta)
        theta_hat = np.vstack([theta, 1./2 * theta**2, 1./3 * theta**3, 1./4 * theta**4])
        int_A = np.diag(np.dot(self.M_coeffs, theta_hat))
        return int_A

    def calc_length(self, theta):
        '''
        Determine the current length of the muscle. Assumes zero muscle pennation angle
        '''
        # import pdb; pdb.set_trace()
        length = self.l0 - np.sum(self.int_moment_arm(theta) - self.int_A_of) * np.pi/180
        # print length
        return length

    def calc_velocity(self, theta, omega):
        A = self.moment_arm(theta)
        dl = -np.dot(A, omega) * 0.01  # convert from cm to m
        return dl

    def __repr__(self):
        return self.name

class MSKModel(object):
    def __init__(self, muscles):
        '''
        muscles : iterable
            List (or similar) of all the muscles in the model
        '''
        self.muscles = muscles

    def get_moment_arm_matrix(self, theta):
        A = []
        for muscle in self.muscles:
            # muscle = self.muscle[muscle_name]
            A_musc = muscle.moment_arm(theta)
            A.append(A_musc)
        return np.vstack(A)

    def __iter__(self):
        return self.muscles.__iter__()

    def __len__(self):
        return len(self.muscles)

    def get_muscle_lengths(self, theta):
        return np.array([muscle.calc_length(theta) for muscle in self.muscles])

    def get_muscle_velocities(self, theta, omega):
        return np.array([muscle.calc_velocity(theta, omega) for muscle in self.muscles])        

    def get_passive_force(self, theta):
        l = self.get_muscle_lengths(theta)
        F_p = np.array([muscle.get_passive_force(l_m) for muscle, l_m in izip(self.muscles, l)])
        return F_p

    def __repr__(self):
        return str([muscle.name for muscle in self.muscles])

    @property
    def l0(self):
        return np.array([muscle.l0 for muscle in self.muscles])

    @property 
    def F_max(self):
        return np.array([muscle.F_max for muscle in self.muscles])

    def get_active_force(self, activ, l, dl):
        return np.array([muscle.get_active_force(a_muscle, l_muscle, dl_muscle) for muscle, a_muscle, l_muscle, dl_muscle in izip(self.muscles, activ, l, dl)])


body_weight = 6.5

# Fmax = scaling factor * PCSA; from Cheng & Scott 2000, p214
# scaling = 22.5 ?
# Spector 1980 : 2.3 kg /cm2 -> 22.56 N / cm2
# Lucas 1987 : 2.43 kg /cm2 -> 23.83 N / cm2
Fmax_sc_factor = 23.8383;

#Initializing muscles with l0, F_max, theta_of, sh_moments_mat, el_moments_mat
#    % Moments -- Graham & Scott 2003, p306
#    % L0 -- optimal fascicle length from Graham & Scott 2003, p308
#    % theta_of -- Theta optimal force
#    F_max -- maximum force that can be generated by the muscle

# BL -- Bicep Long Head (consider removing shoulder moment matrix if BL monoarticular)
BLH  = Muscle(name='BLH', l0=6.1,  F_max=Fmax_sc_factor * (1.14 + 0.44 * body_weight**(2./3)), theta_of=np.deg2rad([-2, 89]),   sh_moments_mat=[1.12, -3.13, 3.78, 0],      el_moments_mat=[1.39, -3.42, 11.40, -0.74])
# BS -- Biceps Short Head
BSH  = Muscle(name='BSH', l0=6.7,  F_max=Fmax_sc_factor * (1.14 + 0.17 * body_weight**(2./3)), theta_of=np.deg2rad([15, 100]),  sh_moments_mat=[1.47,  0, 0, 0],            el_moments_mat=[0.41, 2.99, -1.26, 0])
# B -- Brachialis
B    = Muscle(name='B', l0=4.5,  F_max=Fmax_sc_factor * (1.34 + 0.28 * body_weight**(2./3)), theta_of=np.deg2rad([0, 79]),    sh_moments_mat=[0, 0, 0, 0],                el_moments_mat=[0.31, 0.96, 0, 0])
# Br -- Brachioradialis
Br   = Muscle(name='Br', l0=10.8, F_max=Fmax_sc_factor * (0.30 + 0.17 * body_weight**(2./3)), theta_of=np.deg2rad([0, 85]),    sh_moments_mat=[0, 0, 0, 0],                el_moments_mat=[8.68, -39.40, 71.80, -3.79])
# DA -- Deltoid (Anterior)
DA   = Muscle(name='DA', l0=4.9,  F_max=Fmax_sc_factor * (0.99 + 0.12 * body_weight**(2./3)), theta_of=np.deg2rad([-1, 0]),    sh_moments_mat=[1.47, -1.56, 20.58, -2.98], el_moments_mat=[0, 0, 0, 0]);
# DM -- Deltoid (Medial)
DM   = Muscle(name='DM', l0=2.3,  F_max=Fmax_sc_factor * (0.83 + 0.54 * body_weight**(2./3)), theta_of=np.deg2rad([6, 0]),     sh_moments_mat=[-0.72, -3.02, 5.44, 0],     el_moments_mat=[0,0, 0, 0])
# DP -- Deltoid (Posterior)
DP   = Muscle(name='DP', l0=5.1,  F_max=Fmax_sc_factor * (0.15 + 0.23 * body_weight**(2./3)), theta_of=np.deg2rad([2, 0]),     sh_moments_mat=[-2.75, 2.62, 0, 0],         el_moments_mat=[0, 0, 0, 0])
# ECRL -- Extensor carpi radialis longus
ECRL = Muscle(name='ECRL', l0=5.4,  F_max=Fmax_sc_factor * (2.05 + 0.08 * body_weight**(2./3)), theta_of=np.deg2rad([2, 0]),     sh_moments_mat=[0, 0, 0, 0],                el_moments_mat=[0.59, 1.97, 0, 0])
# De -- Dorsoepitrochlearis
De   = Muscle(name='De', l0=6.3,  F_max=Fmax_sc_factor * (0.17 + 0.16 * body_weight**(2./3)), theta_of=np.deg2rad([-19, 22]),  sh_moments_mat=[-2.99, 4.15, -3.52, 0],     el_moments_mat=[-0.44, -0.91, 0, 0])
# PM(C) -- Pectoralis Major (Clavicular head)
PMC  = Muscle(name='PMC', l0=7.6,  F_max=Fmax_sc_factor * (2.73 + 0.61 * body_weight**(2./3)), theta_of=np.deg2rad([29, 0]),    sh_moments_mat=[2.13, -1.71, 20.87, -2.84], el_moments_mat=[0, 0, 0, 0])
# TLa -- Triceps Lateral
TLa  = Muscle(name='TLa', l0=4.2,  F_max=Fmax_sc_factor * (2.34 + 0.68 * body_weight**(2./3)), theta_of=np.deg2rad([0, 105]),   sh_moments_mat=[0, 0, 0, 0],                el_moments_mat=[1.33, -6.49, 3.98, 0])
# TLo -- Triceps Long
TLo  = Muscle(name='TLo', l0=3.9,  F_max=Fmax_sc_factor * (1.73 + 1.10 * body_weight**(2./3)), theta_of=np.deg2rad([5, 101]),   sh_moments_mat=[-2.69, -2.98, 7.31, 0],     el_moments_mat=[0.30, -1.67, 0, 0])
# TME -- Triceps medial?
TMe  = Muscle(name='TMe', l0=4.4,  F_max=Fmax_sc_factor * (2.51 + 0.25 * body_weight**(2./3)), theta_of=np.deg2rad([0, 117]),   sh_moments_mat=[0, 0, 0, 0],                el_moments_mat=[-1.75, 0, 0, 0])


muscles = [BLH, BSH, TLo, PMC, DM, TLa]
# muscles = [BLH, BSH, B, Br, DA, DM, DP, ECRL, De, PMC, TLa, TLo,  TMe]
mskmodel = MSKModel(muscles)


class VirtualPlanarArmMAM(plants.RobotArmGen2D):
    def __init__(self, *args, **kwargs):

        # Initialize the dynamics world
        # world = ode.World() # TODO this should be part of the window, for object collision detection stuff
        # world.setGravity((0,0,0))


        # Arm link lengths----from monkey P? Numbers taken from MATLAB code originally written by Rodolphe Heliot/Amy Orsborn
        # Changed to these values because most common among old code. Cheng & Scott 2000 Table 2 has other values if change needed
        self.l_ua = 0.01 * 17.70 # m
        self.l_fa = 0.01 * 20.35 # m

        # Friction coefficients
        self.B = np.mat([[0.03, 0.01],
                         [0.01, 0.03]])

        # Mass of upperarm/forearm from Cheng & Scott, 2000, p221, Table 8 (based on morphometric data)
        self.m_ua = 0.001*(23 + 34.4*body_weight) # kg (regression data specified in grams)
        self.m_fa = 0.001*(53 + 25.2*body_weight) # kg 

        ## Determine the inertia of each segment
        rad_of_gyration = np.array([0.247, 0.248]) # relative to the length of each segment

        # Calculate center of mass for each segment
        self.r_ua = self.ctr_of_mass_ua = self.l_ua * rad_of_gyration[0]
        self.r_fa = self.ctr_of_mass_fa = self.l_fa * rad_of_gyration[1]


        # Calculate moment of inertia for each segment 
        # i = 0.001 * 0.0001 * (b + m*total_body_weight), where 'b' and 'm' are from Cheng & Scott 2000, p221, Table 8
        #     0.001 * 0.0001 converts (g cm^2) ==> (kg m^2) 
        self.I_ua = 0.001*0.0001*(432 + 356.6*body_weight)
        self.I_fa = 0.001*0.0001*(2381 + 861.6*body_weight)


        self.num_joints = 2

        self.theta = np.zeros(2) # angular
        self.omega = np.zeros(2) # angular vel 
        self.alpha = np.zeros(2) # angular acc
        self.torque = np.zeros(2)

        base_loc = np.array([2., 0., -15])
        self.base_loc = base_loc
        
        link_radii=arm_radius
        joint_radii=arm_radius
        joint_colors=arm_color
        link_colors=arm_color

        self.link_lengths = link_lengths = np.array([self.l_ua, self.l_fa])
        self.curr_vecs = np.zeros([self.num_joints, 3]) #rows go from proximal to distal links

        self.chain = Chain(link_radii, joint_radii, list(link_lengths * 100), joint_colors, link_colors)
        self.cursor = Sphere(radius=arm_radius/2, color=link_colors)
        self.graphics_models = [self.chain.link_groups[0], self.cursor]
        self.chain.translate(*self.base_loc, reset=True)

        self.kin_chain = robot_arms.PlanarXZKinematicChain2Link(link_lengths * 100, base_loc=base_loc) # Multiplying by 100 to convert from m to cm
        
        self.stay_on_screen = False
        self.visible = True


        self.n_muscles = len(muscles)
        self.muscle_vel = np.zeros(self.n_muscles)

        self.call_count = 0

    def get_data_to_save(self):
        return dict()

    def get_endpoint_pos(self):
        return self.kin_chain.endpoint_pos(self.theta) #arm_segments[1].getPosition()

    def calc_IN(self, theta, omega):
        '''
        Calculate the inertial and coriolis force matrices
        ua = upper arm, fa = forearm
        Equations from "A mathematical introduction to robotic manipulation" by Murray, Li, Sastry, p. 164

        theta : np.array of shape (2,)
            Joint angles
        omega : np.array of shape (2,)
            Joint velocities
        '''
        l_ua = self.l_ua #* 0.01 # convert to m
        l_fa = self.l_fa #* 0.01 # convert to m
        r_ua = self.r_ua #* 0.01 # convert to m
        r_fa = self.r_fa #* 0.01 # convert to m

        alpha = self.I_ua + self.I_fa + self.m_ua*r_ua**2 + self.m_fa*(l_ua**2 + r_fa**2)
        beta = self.m_fa*l_ua*r_fa
        delta = self.I_fa + self.m_fa*r_fa**2

        cos = np.cos
        sin = np.sin
        I = np.mat([[alpha + 2*beta*cos(theta[1]),  delta + beta*cos(theta[1])], 
                    [delta + beta*cos(theta[1]),    delta                     ]]) 

        N = np.mat([[-beta*sin(theta[1])*omega[1],  -beta*sin(theta[1])*sum(omega)],
                    [beta*sin(theta[1])*omega[0],   0                              ]])
        return I, N

    def _handle_torque(self, torque):
        # print torque
        torque = np.mat(torque).reshape(-1,1)
        Inertia, N = self.calc_IN(self.theta, self.omega)
        theta = np.mat(self.theta.reshape(-1, 1))
        omega = np.mat(self.omega.reshape(-1, 1))
        B = np.mat([[0.03, 0.01], [0.01, 0.03]])
        self.alpha = Inertia.I * (torque - N*omega - B*omega)
        # pdb.set_trace()

        DT = 1./60

        self.theta = np.array(theta + DT*omega + DT**2 * self.alpha).ravel()
        self.omega = np.array(omega + DT*self.alpha).ravel()

        # joint limits between (0, pi)
        # self.theta[0] = max(min(self.theta[0], np.pi), 0)
        # self.theta[1] = max(min(self.theta[1], np.pi), 0)

        self.set_intrinsic_coordinates(self.theta)

    def drive(self, decoder):
        theta = np.array(self.theta).ravel()

        muscle_activ_state_names = tuple(['%s_activ' % muscle.name for muscle in mskmodel])
        a = decoder[muscle_activ_state_names]

        # Force muscle activations to be positive
        a[a < 0] = 0
        a[a > 1] = 1
        decoder[muscle_activ_state_names] = a
        print a

        if self.call_count % 6 == 0:
            torque = self._handle_a(a)
            decoder['torque_0', 'torque_1'] = torque

            theta = np.array(self.theta).ravel()
            for muscle in muscles:
                l = muscle.calc_length(theta)
                A = muscle.moment_arm(theta)
                dl = -np.dot(A, self.omega) * cm_to_m 
                decoder['%s_len' % muscle.name] = l
                decoder['%s_vel' % muscle.name] = dl

            decoder['theta_0', 'theta_1'] = np.array(self.theta).ravel()
            decoder['omega_0', 'omega_1'] = np.array(self.omega).ravel()
            decoder['alpha_0', 'alpha_1'] = np.array(self.omega).ravel()

        self.call_count += 1        

    def _handle_a(self, a):
        F = []
        A_full = []
        theta = np.array(self.theta).ravel()
        omega = np.array(self.omega).ravel()

        l = mskmodel.get_muscle_lengths(theta)
        dl = mskmodel.get_muscle_velocities(theta, omega)
        F_passive = mskmodel.get_passive_force(theta)
        F_active = mskmodel.get_active_force(a, l, dl)
        F = F_active + F_passive 

        A = 0.01 * mskmodel.get_moment_arm_matrix(theta)
        torque = np.dot(A.T, F) # one factor is to convert from cm to m
        # print "activations received to plant"
        # print a
        # if np.any(a < 0):
        #     pdb.set_trace()
        # if np.any(a > (1 + 1e-3)):
        #     pdb.set_trace()
        # print "generated torque"
        # print theta, omega
        # print torque
        # pdb.set_trace()

        self._handle_torque(torque)
        # print self.theta
        return torque


class StateSpaceMAM(StateSpace):
    def __init__(self):
        self.states = []
        n_links = 2
        self.n_links = n_links
        for k in range(n_links):
            theta_k = State('theta_%d' % k, stochastic=False,  drives_obs=False, order=0, aux=True)
            self.states.append(theta_k)

        for k in range(n_links):
            omega_k = State('omega_%d' % k, stochastic=False,  drives_obs=False, order=1, aux=True)
            self.states.append(omega_k)

        for k in range(n_links):
            alpha_k = State('alpha_%d' % k, stochastic=False,  drives_obs=False, order=2, aux=True)
            self.states.append(alpha_k)

        for k in range(n_links):
            torque_k = State('torque_%d' % k, stochastic=False,  drives_obs=False, order=2, aux=True)
            self.states.append(torque_k)

        for muscle in muscles:
            musc_activ_k = State('%s_activ' % muscle.name, stochastic=True, drives_obs=True, order=2, aux=False, min_val=0)
            self.states.append(musc_activ_k)

        for muscle in muscles:
            musc_len_k = State('%s_len' % muscle.name, stochastic=False, drives_obs=False, order=2, aux=True)              
            self.states.append(musc_len_k)

        for muscle in muscles:
            musc_vel_k = State('%s_vel' % muscle.name, stochastic=False, drives_obs=False, order=2, aux=True)
            self.states.append(musc_vel_k)


        self.states.append(offset_state)

        self.inertia = np.mat(np.eye(n_links))

    def get_ssm_matrices(self, update_rate=0.1):
        if not update_rate == 0.1:
            raise NotImplementedError
        n_states = len(self.states)
        A = np.mat(np.zeros([n_states, n_states]))
        B = 0
        W = np.mat(np.zeros([n_states, n_states]))

        n_muscles = len(mskmodel)
        # W diagonal values taken from VFB data
        # np.array([  1.21986555e+02,   5.10752323e+06,   1.11013526e+01,   5.30080459e+03,   1.51024704e+01,   2.15531563e+03])
        W[6:6+n_muscles, 6:6+n_muscles] = 0.1 * np.diag(np.ones(6))
        return A, B, W


    # def calc_next_state(self, current_state, target_state, **kwargs):
    #     next_torque = super(JointTorque2LController, self).calc_next_state(current_state, target_state, **kwargs)
    #     ns = np.hstack([np.zeros(6), np.array(next_torque).ravel(), 1]).reshape(-1, 1)
    #     return ns

class MSKController(feedback_controllers.PIDController):
    '''
    Inv muscle dynamics using method of Yamaguchi, Moran and Si, 1995
    '''
    def __init__(self):
        I = np.mat(np.eye(3))
        Delta = 1./60
        self.ssm = StateSpaceMAM()
        # super(MSKController, self).__init__(1, 0.5, 0, self.ssm.state_order)
        super(MSKController, self).__init__(0.2, 0.02, 0, self.ssm.state_order)

    def calc_next_state(self, current_state, target_state, **kwargs):
        current_state = current_state.reshape(-1,1)
        target_state = target_state.reshape(-1, 1)
        next_torque = super(MSKController, self).calc_next_state(current_state, target_state, **kwargs)
        theta = np.array(current_state[0:2,0]).ravel()
        omega = np.array(current_state[2:4,0]).ravel()
        muscle_activ = self.inv_muscle_dynamics(next_torque, theta, omega)
        # uncomment the next line to drive the arm purely using passive force (should go to some equilibrium)
        # muscle_activ = np.zeros_like(muscle_activ)

        print "muscle activ"
        print muscle_activ

        n_muscles = len(mskmodel)
        next_state = np.array(np.hstack([np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), muscle_activ, np.zeros(n_muscles), np.zeros(n_muscles), 1]))
        return next_state.reshape(-1,1)

    def inv_muscle_dynamics(self, torque, theta, omega, eps=0.01):
        '''
        Determine muscle forces to generate the required torque
        '''
        # print "desired torque"
        # print torque
        

        F_muscle = np.ones(len(mskmodel)).reshape(-1, 1) * -1
        A = 0.01 * mskmodel.get_moment_arm_matrix(theta).T  ## TODO determine if this scale factor is actually necessary...
        F_p = mskmodel.get_passive_force(theta)
        torque_passive = np.dot(A, F_p.reshape(-1,1))
        torque_active = torque - torque_passive

        l = mskmodel.get_muscle_lengths(theta)
        dl = mskmodel.get_muscle_velocities(theta, omega)
        l0 = mskmodel.l0


        F_max_l0 = mskmodel.F_max
        length_gain = 1 - 4*((l - l0)/l0)**2
        length_gain[length_gain < 0] = 0
        velocity_gain = f1 + f2*np.arctan(f3 + f4*dl)
        gain = length_gain * velocity_gain        
        F_max = F_max_l0 * gain


        inactive_muscles, = np.nonzero((l > 1.5*l0) + (l < 0.5*l0))
        A[:, inactive_muscles] = 0
        A[:, gain == 0] = 0

        if 0:
            # constrained optimization to keep activations positive
            import scipy.optimize
            bounds = [(0, 100000*x) for x in F_max]
            x0 = np.zeros(A.shape[1])
            F_muscle = scipy.optimize.fmin_slsqp(lambda x: np.linalg.norm(np.dot(A, x) - np.array(torque_active).ravel()), x0, bounds=bounds)
            # F_muscle = data[0]
            # print np.round(F_muscle, decimals=3)
            # print np.array(torque).ravel() - np.dot(A, F_muscle) - np.array(torque_passive).ravel()
            # print "what the inverse muscle dynamics thinks the generated torque will be"
            # print np.dot(A, F_muscle) + np.array(torque_passive).ravel()
            # print "passive torque"
            # print np.array(torque_passive).ravel()
            F_muscle[F_muscle < eps] = 0
        elif 1:
            F_muscle = np.ones(A.shape[1]) * -1
            A_ = A.copy()
            while np.any((F_muscle < 0) * (np.abs(F_muscle) > eps)):
                F_muscle = np.dot(np.linalg.pinv(A_), np.array(torque_active))
                # import pdb; pdb.set_trace()
                negative_forces, _ = np.nonzero((F_muscle < 0) * (np.abs(F_muscle) > eps))
                # if len(negative_forces) > 0:
                #     idx = np.argmin(F_muscle[:,0])
                A_[:, negative_forces] = 0
                # import pdb; pdb.set_trace()
                F_muscle = np.array(F_muscle).ravel()
            # import pdb; pdb.set_trace()
            F_muscle[F_muscle < eps] = 0
        else:
            F_muscle = np.dot(np.linalg.pinv(A), torque_active)
            F_muscle = np.array(F_muscle).ravel()
            

        a = np.zeros_like(F_muscle)

        a[gain > 0] = F_muscle[gain > 0] / (F_max[gain > 0])

        # if np.any((F_muscle - mskmodel.get_active_force(a, l, dl)) > 1e-5):
        #     print "error converting muscle force to activation"
        #     import pdb; pdb.set_trace()


        # print "running forward dynamics"
        F_passive = mskmodel.get_passive_force(theta)
        F_active = mskmodel.get_active_force(a, l, dl)
        F = F_active + F_passive 

        A = 0.01 * mskmodel.get_moment_arm_matrix(theta)
        pred_torque_gen = np.dot(A.T, F) # one factor is to convert from cm to m
        # print "torque diff"
        # print pred_torque_gen - np.array(torque).ravel()

        return a


##### MAM tasks #####
class MAMBMI(BMIControlMulti):
    exclude_parent_traits = ['plant_type', 'plant_visible', 'plant_hide_rate']
    sequence_generators = BMIControlMulti.sequence_generators + ['centerout_2D_discrete_nobottom']

    def __init__(self, *args, **kwargs):
        self.plant = VirtualPlanarArmMAM()
        super(MAMBMI, self).__init__(*args, **kwargs)

    def init_decoder_state(self):
        pass

    def create_assister(self):
        fb_ctrl = MSKController()
        self.assister = FeedbackControllerAssist(fb_ctrl, style='mixing')

    def create_goal_calculator(self):
        pass

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        targ_joint_angles = self.plant.kin_chain.inverse_kinematics(self.target_location.copy()).ravel()

        n_muscles = len(muscles)
        target_state = np.array(np.hstack([targ_joint_angles, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(n_muscles), np.zeros(n_muscles), np.zeros(n_muscles), 1]))
        return np.tile(np.array(target_state).reshape(-1,1), [1, self.decoder.n_subbins])

    @staticmethod
    def centerout_2D_discrete_nobottom(nblocks=100, distance=10):
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

        Returns
        -------
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        pi = np.pi
        angles = np.array([0, pi/4, 2*pi/4, 3*pi/4, 4*pi/4, 5*pi/4, 7*pi/4])
        
        theta = []
        for i in range(nblocks):
            temp = angles.copy()
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs


class MAMVisualFeedback(MAMBMI):
    exclude_parent_traits = []
    assist_level = (1, 1)
    is_bmi_seed = True
    sequence_generators = ['centerout_2D_discrete_nobottom']

    def load_decoder(self):
        self.ssm = StateSpaceMAM()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=1./10)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()

class MAMCLDA(MAMBMI, LinearlyDecreasingHalfLife):
    batch_time = traits.Float(0.1, desc='The length of the batch in seconds')
    decoder_sequence = traits.String('MAM', desc='signifier to group together sequences of decoders')

    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        fb_ctrl = MSKController()
        self.learner = clda.FeedbackControllerLearner(self.batch_size, fb_ctrl, style='mixing')
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def _cycle(self):
        super(MAMCLDA, self)._cycle()
        if self.calc_state_occurrences('reward') > 16:
            self.learner.batch_size = np.inf

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(MAMCLDA, self).call_decoder(*args, **kwargs)
