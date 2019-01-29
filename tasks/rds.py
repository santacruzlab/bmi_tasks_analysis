import time
import numpy as np

from riglib import reward
from riglib.experiment import TrialTypes, traits
try:
    import pygame
    from riglib.experiment import Pygame
except:
    import warnings
    warnings.warn("rds.py: Pygame not imported")
    Pygame = object

class RDS(TrialTypes, Pygame):
    ndots = traits.Int(250, desc="Number of dots on sphere")
    sphere_radius = traits.Float(250, desc="Radius of virtual sphere")
    dot_radius = traits.Int(5, desc="Radius of dots drawn on screen")
    sphere_speed = traits.Float(0.005*np.pi, desc="Speed at which the virtual sphere turns")
    disparity = traits.Float(.05, desc="Amount of disparity")

    trial_types = ["cw", "ccw"]

    def init(self):
        super(RDS, self).init()
        self.screen_init()
        self.width, self.height = self.surf.get_size()
        self.loff = self.width / 4., self.height / 2.
        self.roff = self.width * 0.75, self.height / 2.
    
    def _init_sphere(self):
        u, v = np.random.rand(2, self.ndots)
        self._sphere = np.array([2*np.pi*u, np.arccos(2*v-1)])

    def _project_sphere(self, offset=True):
        theta, phi = self._sphere
        x = self.sphere_radius * np.cos(theta) * np.sin(phi)
        y = self.sphere_radius * np.sin(theta) * np.sin(phi)
        z = self.sphere_radius * np.cos(phi)
        d = y * self.disparity

        return np.array([x+d*(-1,1)[offset], z]).T
    
    def _draw_sphere(self):
        import pygame
        self.surf.fill(self.background)
        for pt in (self.loff + self._project_sphere(True)).astype(int):
            pygame.draw.circle(self.surf, (255, 255, 255), pt, self.dot_radius)
        
        for pt in (self.roff + self._project_sphere(False)).astype(int):
            pygame.draw.circle(self.surf, (255, 255, 255), pt, self.dot_radius)
        self.flip_wait()
    
    def _start_cw(self):
        print 'cw'
        self._init_sphere()

    def _start_ccw(self):
        print 'ccw'
        self._init_sphere()
    
    def _while_cw(self):
        self._sphere[0] += self.sphere_speed
        self._draw_sphere()
    
    def _while_ccw(self):
        self._sphere[0] -= self.sphere_speed
        self._draw_sphere()
    
    def _test_cw_correct(self, ts):
        return self.event in [4, 8]
    
    def _test_cw_incorrect(self, ts):
        return self.event in [1, 2]
    
    def _test_ccw_correct(self, ts):
        return self.event in [1, 2]
    
    def _test_ccw_incorrect(self, ts):
        return self.event in [4, 8]


class RDS_half(RDS):
    def _project_sphere(self, offset=True):
        theta, phi = self._sphere
        x = self.sphere_radius * np.cos(theta) * np.sin(phi)
        y = self.sphere_radius * np.sin(theta) * np.sin(phi)
        z = self.sphere_radius * np.cos(phi)
        d = y * self.disparity

        return np.array([x+d*(-1,1)[offset], z]).T[y > 0]
