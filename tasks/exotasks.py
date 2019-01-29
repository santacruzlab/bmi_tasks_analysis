'''
Tasks for the active and passive exoskeletons
'''
from __future__ import division
import numpy as np
from collections import OrderedDict
import time

from riglib.experiment import traits, Sequence, Experiment, FSMTable, StateTransitions

from riglib import plants

from riglib.stereo_opengl.window import Window, WindowWithExperimenterDisplay
from riglib.stereo_opengl.primitives import Chain, Sphere, Cylinder, Cone
from riglib.stereo_opengl.xfm import Quaternion


from riglib.bmi.bmi import BMILoop, MachineOnlyFilter
from riglib.bmi import feedback_controllers, state_space_models
from riglib.bmi.robot_arms import KinematicChain
from riglib.bmi.state_space_models import StateSpace, offset_state
from riglib.bmi.assist import FeedbackControllerAssist
from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter
from riglib.bmi.extractor import DummyExtractor
from riglib.bmi import clda

from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife
from tasks.manualcontrolmultitasks import ManualControlMulti, VirtualCircularTarget

from riglib.positioner import PositionerTaskController

import math
import traceback
import socket
import struct
import robot
import select

import matplotlib.pyplot as plt

SIM = 0

#####################
##### Constants #####
#####################
n_bytes_per_double = 8
pi = np.pi
arm_color = (181/256., 116/256., 96/256., 1)
arm_radius = 0.6
pi = np.pi

link_lengths = np.array([7., 7.25]) * 1 # [14, 14.5]
exo_chain_graphics_base_loc = np.array((0, 0, 0))

##############################
##### State space models #####
##############################
class StateSpaceJointVelocityActiveExo(StateSpace):
    '''
    State space to represent the upper-arm exoskeleton in active mode
    '''
    def __init__(self):
        '''
        '''
        super(StateSpaceJointVelocityActiveExo, self).__init__(
            state_space_models.State('theta_sh_flex', stochastic=False, drives_obs=False, order=0, aux=True),
            state_space_models.State('theta_sh_abd', stochastic=False, drives_obs=False, order=0, aux=True),
            state_space_models.State('theta_sh_introt', stochastic=False, drives_obs=False, order=0, aux=True),
            state_space_models.State('theta_el_flex', stochastic=False, drives_obs=False, order=0, aux=True),
            state_space_models.State('theta_pron', stochastic=False, drives_obs=False, order=0, aux=True),

            state_space_models.State('omega_sh_flex', stochastic=True, drives_obs=True, order=1, min_val=-0.2, max_val=0.2),
            state_space_models.State('omega_sh_abd', stochastic=True, drives_obs=True, order=1, min_val=-0.2, max_val=0.2),
            state_space_models.State('omega_sh_introt', stochastic=True, drives_obs=True, order=1, min_val=-0.2, max_val=0.2),
            state_space_models.State('omega_el_flex', stochastic=True, drives_obs=True, order=1, min_val=-0.2, max_val=0.2),
            state_space_models.State('omega_pron', stochastic=True, drives_obs=True, order=1, min_val=-0.2, max_val=0.2),

            state_space_models.State('torque_sh_flex', stochastic=False, drives_obs=False, order=2, aux=True),
            state_space_models.State('torque_sh_abd', stochastic=False, drives_obs=False, order=2, aux=True),
            state_space_models.State('torque_sh_introt', stochastic=False, drives_obs=False, order=2, aux=True),
            state_space_models.State('torque_el_flex', stochastic=False, drives_obs=False, order=2, aux=True),
            state_space_models.State('torque_pron', stochastic=False, drives_obs=False, order=2, aux=True),

            offset_state
        )

        self.n_links = 5

    def get_ssm_matrices(self, update_rate=0.1):
        I = np.mat(np.eye(self.n_links))
        Delta = update_rate
        zero_vec = np.zeros([self.n_links,1])
        A = np.vstack([np.hstack([I,   Delta*I, 0*I, zero_vec]), 
                       np.hstack([0*I,       I, 0*I, zero_vec]), 
                       np.hstack([0*I,     0*I, 0*I, zero_vec]),
                       np.hstack([zero_vec.T, zero_vec.T, zero_vec.T, np.ones([1,1])]),
                      ])

        B = np.vstack([0*I,
                       Delta*1000*I, 
                       0*I,
                       zero_vec.T])

        # TODO estimate W_vel from machine-control data
        W = np.mat(np.zeros_like(A))
        W[5:10, 5:10] = np.diag(np.array([ 0.00171236,  0.0001653 ,  0.00045551,  0.00048973,  0.00090953]))

        return A, B, W 

class ActiveExoJointVelFeedbackController(feedback_controllers.LQRController):
    def __init__(self, **kwargs):
        ssm = StateSpaceJointVelocityActiveExo()
        A, B, _ = ssm.get_ssm_matrices()
        Q = np.mat(np.diag(np.hstack([np.ones(5), 0.5*np.ones(5), np.zeros(5), 0])))
        R = np.mat(np.diag([3., 3, 0.3, 0.3, 0.3])) * 100000 * 2
        super(ActiveExoJointVelFeedbackController, self).__init__(A, B, Q, R, **kwargs)        

joint_vel_fb_ctrl = ActiveExoJointVelFeedbackController()


##########################
##### Exo kinematics #####
##########################
class ActiveExoChain(KinematicChain):
    def __init__(self, *args, **kwargs):
        coordinate_origin = kwargs.pop('coordinate_origin', 'task')
        assert coordinate_origin in ['task', 'shoulder']
        self.coordinate_origin = coordinate_origin
        super(ActiveExoChain, self).__init__(*args, **kwargs)
        self.joint_limits = [(-1.25, 1.25), (0, 1.7), (-0.95, 0.9), (0, 1.4), (-1.5, 1.5)]

    def _init_serial_link(self):
        pi = np.pi

        x_rot_angle = -65.5
        #### TODO should these be input parameters to this function instead of being hard-coded?
        dh_params_sh_to_pos = [
            (-15.5,    0, 0, np.deg2rad(67)),
            (0,     77.5, 0, 0),
            (0,        0, 0, np.deg2rad(-67 + x_rot_angle)),
        ]

        dh_params_exo = [
            (0,     0, -pi/2, pi),
            (0,     0, -pi/2, pi),
            (-14.,  0, -pi/2, pi),
            (0,     0, -pi/2, pi),
            (-14.5, 0, 0,     0)
        ]

        if self.coordinate_origin == 'task':
            dh_params = dh_params_sh_to_pos + dh_params_exo
        elif self.coordinate_origin == 'shoulder':
            dh_params = dh_params_exo


        # # first 3 sets of params are for the shoulder-center to positioner origin transformation
        # d = np.array([0,-24.2, 0,                                        0., 0., -14.0, 0, -14.5])  ## cm
        # a = np.array([0, 76, 0,                                          0, 0, 0, 0, 0])
        # alpha = np.array([0, 0, 0,                                       -pi/2, -pi/2, -pi/2, -pi/2, 0])
        # offsets = np.array([np.deg2rad(67), 0, np.deg2rad(-67-65),       pi, pi, pi, pi, 0])

        ##### Construct the links & SerialLink object        
        n_links = len(dh_params)
        # assert len(d) == len(a)
        # assert len(d) == len(alpha)
        # assert len(d) == len(offsets)
        links = []
        for d, a, alpha, offset in dh_params:
            link = robot.Link(a=a, d=d, alpha=alpha, offset=offset)
            links.append(link)            

        # for k in range(n_links):
        #     link = robot.Link(a=a[k], d=d[k], alpha=alpha[k], offset=offsets[k])
        #     links.append(link)

        
        r = robot.SerialLink(links)

        # Length of "stylus" with force sensor, relative to most distal coordinate frame
        tool_offset = np.array([-.01, -3.34, -17.65]) # cm        
        r.tool[0:3, -1] = tool_offset.reshape(-1,1)

        # save attributes
        self.robot = r
        self.link_lengths = d

    def calc_full_joint_angles(self, theta):
        if self.coordinate_origin == 'task':
            return np.hstack([np.zeros(3), theta])
        elif self.coordinate_origin == 'shoulder':
            return theta

    def plot_3d(self, theta, ax=None):
        '''

        '''
        joint_locs = self.spatial_positions_of_joints(theta)

        if self.coordinate_origin == 'shoulder':
            n_fake_joints = 0
        elif self.coordinate_origin == 'task':
            n_fake_joints = 3 # number of "joints" (D-H matrices) used to just translate/rotate from positioner origin to exo origin

        shoulder_center = joint_locs[:, n_fake_joints]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', aspect=1, azim=170, elev=20)

        if self.coordinate_origin == 'task':
            targ_x_min = 0
            targ_x_max = 80.5
            targ_y_min = 0
            targ_y_max = 52
            targ_z_min = -40
            targ_z_max = 0

            # only plot the positioner wire frame if the chain is set up to use task coordinates
            ax.plot([targ_x_min, targ_x_min, targ_x_max, targ_x_max, targ_x_min], [targ_y_min, targ_y_max, targ_y_max, targ_y_min, targ_y_min], [targ_z_min]*5, color='gray', linewidth=2)
            ax.plot([targ_x_min, targ_x_min, targ_x_max, targ_x_max, targ_x_min], [targ_y_min, targ_y_max, targ_y_max, targ_y_min, targ_y_min], [targ_z_max]*5, color='gray', linewidth=2)

            ax.plot([targ_x_min, targ_x_min, targ_x_max, targ_x_max, targ_x_min], [targ_y_min]*5, [targ_z_min, targ_z_max, targ_z_max, targ_z_min, targ_z_min], color='gray', linewidth=2)
            ax.plot([targ_x_min, targ_x_min, targ_x_max, targ_x_max, targ_x_min], [targ_y_max]*5, [targ_z_min, targ_z_max, targ_z_max, targ_z_min, targ_z_min], color='gray', linewidth=2)

            ax.plot([targ_x_min]*5, [targ_y_min, targ_y_min, targ_y_max, targ_y_max, targ_y_min], [targ_z_min, targ_z_max, targ_z_max, targ_z_min, targ_z_min], color='gray', linewidth=2)
            ax.plot([targ_x_max]*5, [targ_y_min, targ_y_min, targ_y_max, targ_y_max, targ_y_min], [targ_z_min, targ_z_max, targ_z_max, targ_z_min, targ_z_min], color='gray', linewidth=2)

            # plot a green dot to indicate the origin of the task coordinate frame
            ax.scatter(0, 0, 0, color='green', s=60)

        # plot blue and green lines to represent the upper arm and forearm, respectively
        upper_arm_locs = joint_locs[:,n_fake_joints:n_fake_joints+3]
        lower_arm_locs = joint_locs[:,n_fake_joints+3:]
        ax.plot(upper_arm_locs[0], upper_arm_locs[1], upper_arm_locs[2], color='blue', linewidth=3)
        ax.plot(lower_arm_locs[0], lower_arm_locs[1], lower_arm_locs[2], color='green', linewidth=3)

        # plot a black dot to indicate the origin of the shoulder coordinate frame (shoulder center)
        ax.scatter(shoulder_center[0], shoulder_center[1], shoulder_center[2], color='black', s=60)
        
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        print 'endpoint location'
        print lower_arm_locs[:,1]
        plt.show()
        plt.draw()   


    def plot_2d(self, theta):
        import plotutil
        joint_locs = self.spatial_positions_of_joints(theta)

        targ_x_min = 0
        targ_x_max = 80.5
        targ_y_min = 0
        targ_y_max = 52
        targ_z_min = -40
        targ_z_max = 0

        n_fake_joints = 3 # number of "joints" used to just translate/rotate from positioner origin to exo origin
        shoulder_center = joint_locs[:, n_fake_joints]


        plt.figure(figsize=(9, 3))
        axes = plotutil.subplots2(1, 3, aspect=1, x=0.05)
        axes[0,0].plot(joint_locs[0,n_fake_joints:], joint_locs[1,n_fake_joints:])
        axes[0,0].scatter(shoulder_center[0], shoulder_center[1])
        # draw x-y positioner box
        axes[0,0].plot([targ_x_min, targ_x_min, targ_x_max, targ_x_max, targ_x_min], [targ_y_min, targ_y_max, targ_y_max, targ_y_min, targ_y_min], color='gray', linewidth=2)


        axes[0,1].plot(joint_locs[0,n_fake_joints:], joint_locs[2,n_fake_joints:])
        axes[0,1].scatter(shoulder_center[0], shoulder_center[2])
        # draw x-z positioner box
        axes[0,1].plot([targ_x_min, targ_x_min, targ_x_max, targ_x_max, targ_x_min], [targ_z_min, targ_z_max, targ_z_max, targ_z_min, targ_z_min], color='gray', linewidth=2)


        axes[0,2].plot(joint_locs[1,n_fake_joints:], joint_locs[2,n_fake_joints:])
        axes[0,2].scatter(shoulder_center[1], shoulder_center[2])
        # draw x-z positioner box
        axes[0,2].plot([targ_y_min, targ_y_min, targ_y_max, targ_y_max, targ_y_min], [targ_z_min, targ_z_max, targ_z_max, targ_z_min, targ_z_min], color='gray', linewidth=2)

        plotutil.clean_up_ticks(axes)
        plotutil.set_title(axes[0,0], 'Top view (x-y)')
        plotutil.set_title(axes[0,1], 'Front view (x-z)')
        plotutil.set_title(axes[0,2], 'Side view (y-z)')

        plotutil.xlabel(axes[0,0], 'x-axis')
        plotutil.ylabel(axes[0,0], 'y-axis')
        plotutil.xlabel(axes[0,1], 'x-axis')
        plotutil.ylabel(axes[0,1], 'z-axis')
        plotutil.xlabel(axes[0,2], 'y-axis')
        plotutil.ylabel(axes[0,2], 'z-axis')

        plt.show()
        plt.draw()        


############################
##### Plant interfaces #####
############################
class ActiveExoPlant(plants.Plant):
    
    n_joints = 5
    force_sensor_offset = 1544.

    def __init__(self, *args, **kwargs):
        if SIM:
            self.rx_port = ('localhost', 60000)
            self.tx_port = ('localhost', 60001)
        else:
            self.rx_port = ('10.0.0.1', 60000)
            self.tx_port = ('10.0.0.14', 60001)            

        self.has_force_sensor = kwargs.pop('has_force_sensor', True)

        self.hdf_attrs = [('joint_angles', 'f8', (5,)), ('joint_velocities', 'f8', (5,)), ('joint_applied_torque', 'f8', (5,)),]
        if self.has_force_sensor and not ('endpt_force' in self.hdf_attrs):
            self.hdf_attrs.append(('endpt_force', 'f8', (1,)))

        # Initialize sockets for transmitting velocity commands / receiving sensor data
        tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.tx_sock = tx_sock

        ## kinematic chain
        self.kin_chain = ActiveExoChain()


        # joint limits in radians, based on mechanical limits---some configurations 
        # may still be outside the comfortable range for the subject
        self.kin_chain.joint_limits = [(-1.25, 1.25), (0, 1.7), (-0.95, 0.9), (0, 1.4), (-1.5, 1.5)]

        ## Graphics, for experimenter only
        self.link_lengths = link_lengths
        self.cursor = Sphere(radius=arm_radius/2, color=arm_color)

        self.upperarm_graphics = Cylinder(radius=arm_radius, height=self.link_lengths[0], color=arm_color)
        # self.upperarm_graphics.xfm.translate(*exo_chain_graphics_base_loc)

        self.forearm_graphics = Cone(radius1=arm_radius, radius2=arm_radius/3, height=self.link_lengths[1], color=arm_color)
        self.forearm_graphics.xfm.translate(*exo_chain_graphics_base_loc)

        self.graphics_models = [self.upperarm_graphics, self.forearm_graphics, self.cursor]
        self.enabled = True

        super(ActiveExoPlant, self).__init__(*args, **kwargs)

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def start(self):
        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_sock.bind(self.rx_port)
        super(ActiveExoPlant, self).start()

    def _get_sensor_data(self):
        if not hasattr(self, 'rx_sock'):
            raise Exception("You seem to have forgotten to 'start' the plant!")
        tx_sock, rx_sock = self.tx_sock, self.rx_sock

        tx_sock.sendto('s', self.tx_port)
        self._read_sock()

    def _check_if_ready(self):
        tx_sock, rx_sock = self.tx_sock, self.rx_sock
        socket_list = [self.rx_sock]

        tx_sock.sendto('s', self.tx_port)

        time.sleep(0.5)
        
        # Get the list sockets which are readable
        read_sockets, write_sockets, error_sockets = select.select(socket_list , [], [], 0)

        return self.rx_sock in read_sockets

    def _read_sock(self):
        n_header_bytes = 4
        n_bytes_per_int = 2

        n_rx_bytes = n_header_bytes*3 + 3*self.n_joints*n_bytes_per_double
        fmt = '>IdddddIdddddIddddd'
        if self.has_force_sensor:
            n_rx_bytes += n_bytes_per_int
            fmt += 'H'

        bin_data = self.rx_sock.recvfrom(n_rx_bytes)
        data = np.array(struct.unpack(fmt, bin_data[0]))

        if self.has_force_sensor:
            force_adc_data = data[-1]
            frac_of_max_force = (force_adc_data - self.force_sensor_offset)/(2**14 - self.force_sensor_offset) 
            force_lbs = frac_of_max_force * 10 
            force_N = force_lbs * 4.448221628254617
            self.force_N = max(force_N, 0) # force must be positive for this sensor
            data = data[:-1]
        data = data.reshape(3, self.n_joints + 1)
        data = data[:,1:]
        self.joint_angles, self.joint_velocities, self.joint_applied_torque = data

    def get_data_to_save(self):
        if not hasattr(self, 'joint_angles'):
            print "No data has been acquired yet!"
            return dict()
        data = dict(joint_angles=self.joint_angles, joint_velocities=self.joint_velocities, joint_applied_torque=self.joint_applied_torque)
        if self.has_force_sensor:
            data['endpt_force'] = self.force_N
        return data

    def _set_joint_velocity(self, vel):
        if not len(vel) == self.n_joints:
            raise ValueError("Improper number of joint velocities!")

        if self.enabled:
            vel = vel.ravel()
        else:
            vel = np.zeros(5)
        self.tx_sock.sendto(struct.pack('>I' + 'd'*self.n_joints, self.n_joints, vel[0], vel[1], vel[2], vel[3], vel[4]), self.tx_port)
        self._read_sock()

    def stop_vel(self):
        self._set_joint_velocity(np.zeros(5))

    def stop(self):
        self.rx_sock.close()
        print "RX socket closed!"

    def set_intrinsic_coordinates(self, theta):
        '''
        Set the joint by specifying the angle in radians. Theta is a list of angles. If an element of theta = NaN, angle should remain the same.
        '''
        joint_locations = self.kin_chain.spatial_positions_of_joints(theta)

        vec_sh_to_elbow = joint_locations[:,2]
        vec_elbow_to_endpt = joint_locations[:,4] - joint_locations[:,2]

        self.upperarm_graphics.xfm.rotate = Quaternion.rotate_vecs((0,0,1), vec_sh_to_elbow)
        # self.upperarm_graphics.xfm.translate(*exo_chain_graphics_base_loc)
        self.forearm_graphics.xfm.rotate = Quaternion.rotate_vecs((0,0,1), vec_elbow_to_endpt)
        self.forearm_graphics.xfm.translate(*vec_sh_to_elbow, reset=True)

        self.upperarm_graphics._recache_xfm()
        self.forearm_graphics._recache_xfm()

        self.cursor.translate(*self.get_endpoint_pos(), reset=True)

    def get_intrinsic_coordinates(self, new=False):
        if new or not hasattr(self, 'joint_angles'):
            self._get_sensor_data()
        return self.joint_angles

    def get_endpoint_pos(self):
        if not hasattr(self, 'joint_angles'):
            self._get_sensor_data()        
        return self.kin_chain.endpoint_pos(self.joint_angles)

    def draw_state(self):
        self._get_sensor_data()
        self.set_intrinsic_coordinates(self.joint_angles)

    def drive(self, decoder):
        # import pdb; pdb.set_trace()
        joint_vel = decoder['qdot']
        joint_vel[joint_vel > 0.2] = 0.2
        joint_vel[joint_vel < -0.2] = -0.2

        joint_vel[np.abs(joint_vel) < 0.02] = 0

        # send the command to the robot
        self._set_joint_velocity(joint_vel)

        # set the decoder state to the actual joint angles
        decoder['q'] = self.joint_angles

        self.set_intrinsic_coordinates(self.joint_angles)

    def vel_control_to_joint_config(self, fb_ctrl, target_config, sim=True, control_rate=10, tol=np.deg2rad(10)):
        '''
        Parameters
        ----------
        control_rate : int
            Control rate, in Hz
        '''
        target_state = np.hstack([target_config, np.zeros_like(target_config), np.zeros_like(target_config), 1])
        target_state = np.mat(target_state.reshape(-1,1))

        if not sim:
            self._get_sensor_data()
        else:
            # assume that self.joint_angles has been automagically set
            pass
        current_state = np.hstack([self.joint_angles, np.zeros_like(target_config), np.zeros_like(target_config), 1])
        current_state = np.mat(current_state.reshape(-1,1))

        N = 250
        traj = np.zeros([current_state.shape[0], N]) * np.nan

        for k in range(250):
            print k
            current_config = np.array(current_state[0:5, 0]).ravel()
            # print current_config
            
            if np.all(np.abs(current_config - target_config) < tol):
                print np.abs(current_config - target_config)
                print tol
                break

            current_state = fb_ctrl.calc_next_state(current_state, target_state)
            
            traj[:,k] = np.array(current_state).ravel()

            if sim:
                pass
            else:
                current_vel = np.array(current_state[5:10,0]).ravel()
                self._set_joint_velocity(current_vel)
                
                # update the current state using the joint encoders
                current_state = np.hstack([self.joint_angles, np.zeros_like(target_config), np.zeros_like(target_config), 1])
                current_state = np.mat(current_state.reshape(-1,1))
                
                time.sleep(1./control_rate)

        return traj


#################
##### Tasks #####
#################
class ExoBase(object):
    plant = ActiveExoPlant()

    @classmethod 
    def pre_init(cls, **kwargs):
        '''
        See riglib.experiment.Experiment.pre_init for documentation
        Before calling the next pre_init in the MRO, checks if the plant is ready to send data
        '''
        plant = ActiveExoPlant()
        plant.start()
        if not plant._check_if_ready():
            raise Exception("Exo not ready to send data!")

        super(ExoBase, cls).pre_init(**kwargs)

    def init(self):
        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

        self.plant.start()
        super(ExoBase, self).init()

    def run(self):
        '''
        Tell the plant to stop (free the UDP socket) when the task ends. 
        '''
        try:
            super(ExoBase, self).run()
        finally:
            self.plant.stop_vel()
            self.plant.stop()

    def _cycle(self):
        self.move_plant()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(ExoBase, self)._cycle()

    def terminate(self):
        # self.plant.stop()
        self.plant.rx_sock.close()
        super(ExoBase, self).terminate()


class BMIControlExoJointVel(ExoBase, BMILoop, Window):
    background = (.5,.5,.5,1) # Set the screen background color to grey
    plant = ActiveExoPlant()
    def __init__(self, *args, **kwargs):
        print "initializing task..."
        super(BMIControlExoJointVel, self).__init__(*args, **kwargs)

        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        print "finished initializing task!"

    def create_assister(self):
        self.assister = FeedbackControllerAssist(joint_vel_fb_ctrl, style='mixing')

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        if hasattr(self, '_gen_target_config'):
            target_config = self._gen_target_config
        else:
            target_config = self.plant.joint_angles
        target_state = np.hstack([target_config, np.zeros_like(target_config), np.zeros_like(target_config), 1])
        target_state = target_state.reshape(-1, 1)
        return target_state


class EndpointManualControl(ExoBase, PositionerTaskController):
    status = dict(
        go_to_origin = dict(microcontroller_done='wait', stop=None),
        wait = dict(start_trial='move_target', stop=None),
        move_target = dict(microcontroller_done='reach'),
        reach = dict(force_applied='reward', new_target_set_remotely='move_target', skip='wait', stop=None),
        reward = dict(time_expired='wait', stop=None),
    )

    trial_end_states = ['reward']

    min_force_on_target = traits.Float(1., desc='Force that needs to be applied, in Newtons')
    reward_time = traits.Float(3., desc='reward time for solenoid')
    

    def move_plant(self):
        self.plant._get_sensor_data()

    def _test_force_applied(self, *args, **kwargs):
        return self.plant.force_N > self.min_force_on_target

    def _end_move_target(self):
        # send command to kill motors
        steps_actuated = self.pos_uctrl_iface.end_continuous_move(stiff=True)
        self._integrate_steps(steps_actuated, self.pos_uctrl_iface.motor_dir)

    def _cycle(self):
        # print "_cycle"
        super(EndpointManualControl, self)._cycle()

    def init(self):
        import pygame
        pygame.init()
        super(EndpointManualControl, self).init()

    def _test_skip(self, *args, **kwargs):
        import pygame
        keys = pygame.key.get_pressed()
        return keys[pygame.K_RIGHT]


class MachineControlExoJointVel(Sequence, BMIControlExoJointVel):
    '''
    Task to automatically go between different joint configurations
    '''
    sequence_generators = ['exo_joint_space_targets']
    current_assist_level = 1

    status = FSMTable(
        wait=StateTransitions(start_trial='move'),
        move=StateTransitions(reached_config='reward'),
        reward=StateTransitions(time_expired='wait'),
    )

    state = 'wait'

    reward_time = traits.Float(3, desc='Time that reward solenoid is open')
    config_tolerances_deg = traits.Tuple((10, 10, 10, 7.5, 10), desc='Time that reward solenoid is open')
    
    def __init__(self, *args, **kwargs):
        super(MachineControlExoJointVel, self).__init__(*args, **kwargs)
        self.config_tolerances = np.deg2rad(np.array(self.config_tolerances_deg))
        self.config_tolerances[-1] = np.inf

    @staticmethod 
    def exo_joint_space_targets(n_blocks=10):
        # neutral_config = np.array([-0.64732247,  0.79,  0.19634043,  0.97628754, -0.02114062])
        target1 = np.array([-1.05, 0.7, 0.4, 1, 0]) 
        target2 = np.array([-0.25, 0.7, 0.4, 1, 0])
        target3 = np.array([-0.85, 1.3 , 0.4, 1, 0])
        target4 = np.array([-0.65, 1.3, 0.4, 0.2, 0])



        trial_target_ls = []
        for k in range(n_blocks):
            configs = np.random.permutation([target1, target2, target3, target4]) # generate the target sequence for each block
            for config in configs:
                trial_target_ls.append(dict(target_config=config))
        return trial_target_ls


    def _test_reached_config(self, *args, **kwargs):
        target_endpt = self.plant.kin_chain.endpoint_pos(self._gen_target_config)
        current_endpt = self.plant.kin_chain.endpoint_pos(self.plant.joint_angles)
        pos_diff = np.linalg.norm(current_endpt - target_endpt)
        joint_diff = np.abs(self.plant.joint_angles - self._gen_target_config)
        # print pos_diff 
        # print np.round(np.rad2deg(joint_diff), decimals=2)
        # print 
        return (pos_diff < 3) or np.all(joint_diff < self.config_tolerances)
        

    def load_decoder(self):
        self.ssm = StateSpaceJointVelocityActiveExo()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        from riglib.bmi.extractor import DummyExtractor

        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()

    def _start_reward(self):
        print "trial complete!"




class MachineControlExoJointVel_w_Positioner(BMIControlExoJointVel, PositionerTaskController):
    '''
    Task to automatically go between different joint configurations with the target positioner following
    '''
    sequence_generators = ['exo_joint_space_targets']
    current_assist_level = 1

    status = FSMTable(
        wait=StateTransitions(start_trial='move'),
        move=StateTransitions(reached_config='reward'),
        reward=StateTransitions(time_expired='wait'),
    )

    state = 'wait'

    reward_time = traits.Float(3, desc='Time that reward solenoid is open')
    config_tolerances_deg = traits.Tuple((10, 10, 10, 7.5, 10), desc='Time that reward solenoid is open')
    
    def __init__(self, *args, **kwargs):
        super(MachineControlExoJointVel, self).__init__(*args, **kwargs)
        self.config_tolerances = np.deg2rad(np.array(self.config_tolerances_deg))
        self.config_tolerances[-1] = np.inf

    @staticmethod 
    def exo_joint_space_targets(n_blocks=10):
        # neutral_config = np.array([-0.64732247,  0.79,  0.19634043,  0.97628754, -0.02114062])
        target1 = np.array([-1.05, 0.7, 0.4, 1, 0]) 
        target2 = np.array([-0.25, 0.7, 0.4, 1, 0])
        target3 = np.array([-0.85, 1.3 , 0.4, 1, 0])
        target4 = np.array([-0.65, 1.3, 0.4, 0.2, 0])



        trial_target_ls = []
        for k in range(n_blocks):
            configs = np.random.permutation([target1, target2, target3, target4]) # generate the target sequence for each block
            for config in configs:
                trial_target_ls.append(dict(target_config=config))
        return trial_target_ls


    def _test_reached_config(self, *args, **kwargs):
        target_endpt = self.plant.kin_chain.endpoint_pos(self._gen_target_config)
        current_endpt = self.plant.kin_chain.endpoint_pos(self.plant.joint_angles)
        pos_diff = np.linalg.norm(current_endpt - target_endpt)
        joint_diff = np.abs(self.plant.joint_angles - self._gen_target_config)
        # print pos_diff 
        # print np.round(np.rad2deg(joint_diff), decimals=2)
        # print 
        return (pos_diff < 3) or np.all(joint_diff < self.config_tolerances)
        

    def load_decoder(self):
        self.ssm = StateSpaceJointVelocityActiveExo()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        from riglib.bmi.extractor import DummyExtractor

        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()

    def _start_reward(self):
        print "trial complete!"




class ExoAssist(LinearlyDecreasingAssist):
    def get_current_assist_level(self):
        if self.state in ['reward', 'init_exo']:
            return 1
        else:
            return self.current_assist_level # set by linearly decreasing attribute

class BMIControlExoEndpt(ExoAssist, BMIControlExoJointVel, PositionerTaskController):
    status = dict(
        go_to_origin = dict(microcontroller_done='wait', stop=None),
        wait = dict(start_trial='init_exo', stop=None),
        init_exo = dict(exo_ready='move_target', stop=None),
        move_target = dict(microcontroller_done='pause'),
        pause = dict(time_expired='reach', stop=None),
        reach = dict(force_applied='reward', new_target_set_remotely='move_target', skip='wait', stop=None),
        reward = dict(time_expired='wait', stop=None),
    )
    state = 'go_to_origin'
    pause_time = 2
    

    trial_end_states = ['reward']

    sequence_generators = ['exo_endpt_targets']

    min_force_on_target = traits.Float(1., desc='Force that needs to be applied, in Newtons')
    reward_time = traits.Float(3., desc='reward time for solenoid')
    config_tolerances_deg = traits.Tuple((10, 10, 10, 7.5, 10), desc='Time that reward solenoid is open')


    def __init__(self, *args, **kwargs):
        super(BMIControlExoEndpt, self).__init__(*args, **kwargs)
        self.config_tolerances = np.deg2rad(np.array(self.config_tolerances_deg))
        self.config_tolerances[-1] = np.inf

        # fixed target location, for now
        self._gen_int_target_pos = np.array([ 27.3264, 47.383, -22.79])
        #36.4222245 ,  26.27462972, -11.38728596
        self.neutral_config = np.array([-0.64732247,  0.79,  0.19634043,  0.97628754, -0.02114062])

        self._gen_target_config = np.array([-0.50579373,  1.28357092,  0.66706522,  0.6, -0.02114062])
        self.current_assist_level = 1

    def init(self):
        self.add_dtype('starting_config', 'f8', (5,))
        super(BMIControlExoEndpt, self).init()

    def init_decoder_state(self):
        pass

    @staticmethod 
    def exo_endpt_targets(n_blocks=1):
        final_config = np.array([-0.50579373,  1.28357092,  0.66706522,  0.6, -0.02114062])
        configs = [final_config]
        trial_target_ls = []

        init_configs = [
            np.array([-0.64732247,  0.79,  0.19634043,  0.97628754, -0.02114062]),
            np.array([-0.64732247,  1.5,  0.19634043,  0.97628754, -0.02114062]),                
            np.array([-1.05,  0.79,  0.19634043,  0.97628754, -0.02114062]),
            np.array([-1.05,  0.79,  -0.07,  0.97628754, -0.02114062]),            
        ]

        for k in range(n_blocks):
            for config in init_configs:
                trial_target_ls.append(dict(target_config=final_config, init_config=config))
        return trial_target_ls        

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        if self.state in ['reach', 'reward']:
            target_config = self._gen_target_config
        elif hasattr(self, '_gen_init_config'):
            target_config = self._gen_init_config
        else: # before the wait state
            target_config = self.neutral_config
            
        target_state = np.hstack([target_config, np.zeros_like(target_config), np.zeros_like(target_config), 1])
        target_state = target_state.reshape(-1, 1)

        self.print_to_terminal(self.decoder.get_state())
        self.print_to_terminal(self.plant.joint_angles)
        self.print_to_terminal(target_state[0:5,0].ravel())
        self.print_to_terminal('')
        return target_state

    def _test_exo_ready(self, *args, **kwargs):
        target_endpt = self.plant.kin_chain.endpoint_pos(self._gen_init_config)
        current_endpt = self.plant.kin_chain.endpoint_pos(self.plant.joint_angles)
        pos_diff = np.linalg.norm(current_endpt - target_endpt)
        joint_diff = np.abs(self.plant.joint_angles - self._gen_init_config)
        return (pos_diff < 3) or np.all(joint_diff < self.config_tolerances)

    def _cycle(self):
        if hasattr(self, '_gen_init_config'):
            starting_config = self._gen_init_config
        else: # before the wait state
            starting_config = self.neutral_config

        self.task_data['starting_config'] = starting_config
        super(BMIControlExoEndpt, self)._cycle()

    def _end_go_to_origin(self):
        steps_actuated = self.pos_uctrl_iface.end_continuous_move(stiff=True)

        self.loc = np.zeros(3)
        self.steps_from_origin = np.zeros(3)

    def _end_move_target(self):
        # send command to kill motors
        steps_actuated = self.pos_uctrl_iface.end_continuous_move(stiff=True)
        self._integrate_steps(steps_actuated, self.pos_uctrl_iface.motor_dir)

    def _test_force_applied(self, *args, **kwargs):
        return self.plant.force_N > self.min_force_on_target


class MachineControlExoEndpt(BMIControlExoEndpt):
    current_assist_level = 1
    is_bmi_seed = True

    def load_decoder(self):
        self.ssm = StateSpaceJointVelocityActiveExo()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        from riglib.bmi.extractor import DummyExtractor

        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()

    
class CLDAControlExoEndpt(BMIControlExoEndpt, LinearlyDecreasingHalfLife):
    batch_time = traits.Float(0.1, desc='The length of the batch in seconds')
    decoder_sequence = traits.String('exo', desc='signifier to group together sequences of decoders')

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def init(self):
        super(CLDAControlExoEndpt, self).init()
        self.batch_time = self.decoder.binlen
        self.updater.init(self.decoder)        

    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        self.learner = clda.FeedbackControllerLearner(self.batch_size, joint_vel_fb_ctrl, reset_states=['go_to_origin', 'wait', 'init_exo', 'move_target', 'pause', 'reward' ])
        self.learn_flag = True

        
class RecordEncoderData(BMIControlExoJointVel):
    '''
    Task which just records encoder values/motor sensor feedback and displays the configuration of the arm on screen
    '''
    def move_plant(self):
        self.plant.draw_state()

    def _cycle(self):
        # print "task is running!"
        super(RecordEncoderData, self)._cycle()

    def load_decoder(self):
        self.ssm = StateSpaceJointVelocityActiveExo()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()


