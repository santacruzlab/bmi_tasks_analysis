from __future__ import division
import numpy as np
from collections import OrderedDict
import time

from riglib import reward
from riglib.experiment import traits, Sequence, Experiment, LogExperiment, Pygame

from riglib.stereo_opengl.window import Window, FPScontrol
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import ssao, stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex
from riglib.stereo_opengl.ik import RobotArm

class ArmPositionTraining(LogExperiment):
    status = dict(
        wait = dict(in_position="hold", stop=None),
        hold = dict(hold_complete="reward", leave_position="wait", stop=None),
        reward = dict(reward_end="wait")
        )

    hold_time_range = traits.Tuple(2, 6, desc="Range of randomly selected hold time before reward occurs in seconds")
    distance = traits.Float(2, desc="Distance from target location hand must be within for reward in cm")
    reward_time = traits.Float(.2, desc="Length of reward")

    target_position = [214,7,380]
    hand_position = [0,0,0]

    def _start_None(self):
        pass
        
    def update_hand_position(self):
        pt = self.motiondata.get()
        if len(pt) > 0:
            pt = pt[:, 0, :]
            pt = pt[~np.isnan(pt).any(1)]        
        if len(pt) > 0:
            self.hand_position = pt.mean(0)[0:3]
            #print self.hand_position

    def _while_wait(self):
        self.update_hand_position()
        time.sleep(1/60)

    def _while_hold(self):
       self.update_hand_position()

    def _test_in_position(self, ts):
        d = np.sqrt((self.hand_position[0]-self.target_position[0])**2 + (self.hand_position[1]-self.target_position[1])**2 + (self.hand_position[2]-self.target_position[2])**2)
        return d <= self.distance*10

    def _test_leave_position(self, ts):
        d = np.sqrt((self.hand_position[0]-self.target_position[0])**2 + (self.hand_position[1]-self.target_position[1])**2 + (self.hand_position[2]-self.target_position[2])**2)
        return d > self.distance*10

    def _test_hold_complete(self, ts):
        return ts >= np.random.rand()*(self.hold_time_range[1] - self.hold_time_range[0]) + self.hold_time_range[0]

    def _test_reward_end(self, ts):
        return ts >= self.reward_time

    def _start_reward(self):
        pass

class FreeMap(Pygame):
    status = dict(
        wait = dict(stop=None)
        )

    def __init__(self, *args, **kwargs):
        super(FreeMap, self).__init__(**kwargs)
        # Set up font.
        import pygame
        pygame.font.init()
        self.font1 = pygame.font.SysFont(None,100)
        self.font2 = pygame.font.SysFont(None,300)

    def update_marker_nums(self):
        # Get current motion data.
        pt = self.motiondata.get()
        mask = np.array([])
        text = []
        coords1 = []
        coords2 = []

        # Identify marker numbers present and filter out those with bad
        # condition numbers.
        if len(pt) > 0:
            conds = pt[:,:,-1]
            mask = (conds>=0) & (conds!=4)
            mask = np.sum(mask,0)
            mask = mask>0

        # For all good markers, add their numbers to a text object to be
        # displayed on screen.
        for i, boo in enumerate(mask):
            if boo:
                text.append(self.font1.render(str(i), True, (0, 255, 0),
                                             (0,0,0)))
            else:
                text.append(self.font1.render(str(i), True, (255,0,0), (0,0,0)))
                
            coords1.append((2000 + (i * 100), 900))
            coords2.append((100 + (i * 100), 900))

        # Blit text to display
        [self.surf.blit(txt, crd) for txt, crd in zip(text, coords1)]
        [self.surf.blit(txt, crd) for txt, crd in zip(text, coords2)]

    def _start_wait(self):
        pass

    def _while_wait(self):
        # Clear screen
        self.surf.fill(self.background)
        # Update text
        self.update_marker_nums()
        # Display text
        self.flip_wait()
                
    def _start_None(self):
        pass

class NumberMap(Sequence, FreeMap):
    status = dict(
        wait = dict(stop=None, start_trial="numdisplay"),
        numdisplay = dict(stop=None, cont="wait", reward_trig="reward"),
        reward = dict(reward_end="wait")
        )

    reward_time = traits.Float(.5, desc="Length of juice reward")

    def __init__(self, *args, **kwargs):
        super(NumberMap, self).__init__(*args, **kwargs)
        self.trial_count = 0

    def _while_wait(self):
        super(NumberMap, self)._while_wait()

    def _start_numdisplay(self):
        self.trial_count += 1

    def _while_numdisplay(self):
        # Clear screen
        self.surf.fill(self.background)
        # Update marker text
        self.update_marker_nums()

        # Create text object for map number and blit.
        mapnum = self.font2.render(str(self.next_trial), True, (255, 255, 255),
                                   (0,0,0))
        self.surf.blit(mapnum, (2800, 300))
        self.surf.blit(mapnum, (900, 300))

        # Render some text to display the current trial number.
        prog_text = self.font1.render('Trial # ' + str(self.trial_count), True,
                                     (0,0,255), (0,0,0))
        self.surf.blit(prog_text, (2000, 800))
        self.surf.blit(prog_text, (100, 800))

        # Display
        self.flip_wait()

    def _test_cont(self, ts):
        return self.event in [4, 8]
    
    def _test_reward_trig(self, ts):
        return self.event in [1, 2]

    def _test_reward_end(self, ts):
        return ts > self.reward_time


def gen_taps(length=640,
             numlist=[1,2,5,6,9,10,13,14,17,18,21,22,23,27,28,29,33,34,35,39, \
                      40,41,45,46,47,51,52,53,57,58,63,64]):

    from random import shuffle

    output = []

    for i in range(length//len(numlist)):
        lst = list(numlist)
        shuffle(lst)
        output = output + lst

    if length%len(numlist) > 0:
        lst = list(numlist)
        shuffle(lst)
        output = output + lst[:length%len(numlist)]

    return output
