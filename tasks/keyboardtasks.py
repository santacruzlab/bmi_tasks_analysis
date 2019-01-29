'''
Tasks for experimenters (e.g., for debugging)
'''
from __future__ import division
import numpy as np
from collections import OrderedDict
import time

from riglib import reward
from riglib.experiment import traits, Sequence

from riglib.stereo_opengl.window import Window, FPScontrol, WindowDispl2D
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from plantlist import plantlist

from riglib.stereo_opengl import ik

import math
import traceback


class ArmPlant(Window):

    '''
    This task creates a RobotArm object and allows it to move around the screen based on either joint or endpoint
    positions. There is a spherical cursor at the end of the arm. The links of the arm can be visible or hidden.
    '''
    
    background = (0,0,0,1)
    
    arm_visible = traits.Bool(True, desc='Specifies whether entire arm is displayed or just endpoint')
    
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    cursor_color = (.5,0,.5,1)

    arm_class = traits.OptionsList(*plantlist, bmi3d_input_options=plantlist.keys())
    starting_pos = (5, 0, 5)

    def __init__(self, *args, **kwargs):
        super(ArmPlant, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        # Initialize the arm
        self.arm = ik.test_3d
        self.arm_vis_prev = True

        if self.arm_class == 'CursorPlant':
            pass
        else:
            self.dtype.append(('joint_angles','f8', (self.arm.num_joints, )))
            self.dtype.append(('arm_visible','f8',(1,)))
            self.add_model(self.arm)

        ## Declare cursor
        self.dtype.append(('cursor', 'f8', (3,)))
        self.cursor = Sphere(radius=self.cursor_radius, color=self.cursor_color)
        self.add_model(self.cursor)
        self.cursor.translate(*self.arm.get_endpoint_pos(), reset=True)

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second by default.
        '''
        ## Run graphics commands to show/hide the arm if the visibility has changed
        if self.arm_class != 'CursorPlant':
            if self.arm_visible != self.arm_vis_prev:
                self.arm_vis_prev = self.arm_visible
                self.show_object(self.arm, show=self.arm_visible)

        self.move_arm()
        self.update_cursor()
        if self.cursor_visible:
            self.task_data['cursor'] = self.cursor.xfm.move.copy()
        else:
            #if the cursor is not visible, write NaNs into cursor location saved in file
            self.task_data['cursor'] = np.array([np.nan, np.nan, np.nan])

        if self.arm_class != 'CursorPlant':
            if self.arm_visible:
                self.task_data['arm_visible'] = 1
            else:
                self.task_data['arm_visible'] = 0
        
        super(ArmPlant, self)._cycle()

    ## Functions to move the cursor using keyboard/mouse input
    def get_mouse_events(self):
        import pygame
        events = []
        for btn in pygame.event.get((pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)):
            events = events + [btn.button]
        return events

    def get_key_events(self):
        import pygame
        return pygame.key.get_pressed()

    def move_arm(self):
        '''
        allows use of keyboard keys to test movement of arm. Use QW/OP for joint movements, arrow keys for endpoint movements
        '''
        import pygame

        keys = self.get_key_events()
        joint_speed = (np.pi/6)/60
        hand_speed = .2

        x,y,z = self.arm.get_endpoint_pos()

        if keys[pygame.K_RIGHT]: 
            x = x - hand_speed
            self.arm.set_endpoint_pos(np.array([x,0,z]))
        if keys[pygame.K_LEFT]: 
            x = x + hand_speed
            self.arm.set_endpoint_pos(np.array([x,0,z]))
        if keys[pygame.K_DOWN]: 
            z = z - hand_speed
            self.arm.set_endpoint_pos(np.array([x,0,z]))
        if keys[pygame.K_UP]: 
            z = z + hand_speed
            self.arm.set_endpoint_pos(np.array([x,0,z]))
        
        if self.arm.num_joints==2:
            xz, xy = self.get_arm_joints()
            e = np.array([xz[0], xy[0]])
            s = np.array([xz[1], xy[1]])
 
            if keys[pygame.K_q]:
                s = s - joint_speed
                self.set_arm_joints([e[0],s[0]], [e[1],s[1]])
            if keys[pygame.K_w]: 
                s = s + joint_speed
                self.set_arm_joints([e[0],s[0]], [e[1],s[1]])
            if keys[pygame.K_o]: 
                e = e - joint_speed
                self.set_arm_joints([e[0],s[0]], [e[1],s[1]])
            if keys[pygame.K_p]: 
                e = e + joint_speed
                self.set_arm_joints([e[0],s[0]], [e[1],s[1]])

        if self.arm.num_joints==4:
            jts = self.get_arm_joints()
            keyspressed = [keys[pygame.K_q], keys[pygame.K_w], keys[pygame.K_e], keys[pygame.K_r]]
            for i in range(self.arm.num_joints):
                if keyspressed[i]:
                    jts[i] = jts[i] + joint_speed
                    self.set_arm_joints(jts)

    def get_cursor_location(self):
        return self.arm.get_endpoint_pos()

    def set_arm_endpoint(self, pt, **kwargs):
        self.arm.set_endpoint_pos(pt, **kwargs)

    def set_arm_joints(self, angle_xz, angle_xy):
        self.arm.set_intrinsic_coordinates(angle_xz, angle_xy)

    def get_arm_joints(self):
        return self.arm.get_intrinsic_coordinates()

    def update_cursor(self):
        '''
        Update the cursor's location and visibility status.
        '''
        pt = self.get_cursor_location()
        if pt is not None:
            self.move_cursor(pt)

    def move_cursor(self, pt):
        ''' Move the cursor object to the specified 3D location. '''
        if not hasattr(self.arm, 'endpt_cursor'):
            self.cursor.translate(*pt[:3],reset=True)

