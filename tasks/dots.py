import time
import random

import numpy as np

from riglib import reward
from riglib.experiment import LogExperiment, TrialTypes, traits

try:
    import pygame
    from riglib.experiment import Pygame
except:
    import warnings
    warnings.warn("rds.py: Pygame not imported")
    Pygame = object

def checkerboard(size=(500,500), n=6):
    square = np.ones(np.array(size) / n)
    line = np.hstack([np.hstack([square, 0*square])]*(n/2))
    return np.vstack([line,line[:,::-1]]*(n/2)).astype(bool)

def squaremask(size=(500,500), square=200):
    data = np.zeros(size, dtype=bool)
    top, bottom = size[0]/2-square/2, size[0]/2+square/2
    left, right = size[1]/2-square/2, size[1]/2+square/2
    data[top:bottom, left:right] = True
    return data

def generate(mask, offset=10):
    data = (np.random.random(mask.shape) > 0.5).astype(int)
    left, right = data.copy(), data.copy()
    leftmask = np.roll(mask, offset, axis=1)
    rightmask = np.roll(mask, -offset, axis=1)
    left[mask] = data[leftmask] 
    right[mask] = data[rightmask]
    left[np.logical_and(np.roll(mask, -offset*2, axis=1), mask)] *= 2
    right[np.logical_and(np.roll(mask, offset*2, axis=1), mask)] *= 2
    return left, right, data

class Dots(TrialTypes, Pygame):
    trial_types = ["flat", "depth"]
    saturation = traits.Float(1.)

    def init(self):
        super(Dots, self).init()
        
        self.width, self.height = self.surf.get_size()
        mask = squaremask()
        mid = self.height / 2 - mask.shape[0] / 2
        lc = self.width / 4 - mask.shape[1] / 2
        rc = 3*self.width / 4 - mask.shape[1] / 2

        self.mask = mask
        self.coords = (lc, mid), (rc, mid)
    
    def _start_depth(self):
        c = 255*(1-self.saturation)
        left, right, flat = generate(self.mask)
        sleft = pygame.surfarray.make_surface(left.T)
        sright = pygame.surfarray.make_surface(right.T)
        sleft.set_palette([(0,0,0,255), (c,c,255,255), (c,255,c,255)])
        sright.set_palette([(0,0,0,255), (c,c,255,255), (c,255,c,255)])
        self.sleft, self.sright = sleft, sright

    def _start_flat(self):
        left, right, flat = generate(self.mask)
        sflat = pygame.surfarray.make_surface(flat.T)
        sflat.set_palette([(0,0,0,255), (255,255,255,255)])
        self.sflat = sflat
    
    def _while_depth(self):
        self.surf.blit(self.sleft, self.coords[0])
        self.surf.blit(self.sright, self.coords[1])
        self.flip_wait()
    
    def _while_flat(self):
        self.surf.blit(self.sflat, self.coords[0])
        self.surf.blit(self.sflat, self.coords[1])
        self.flip_wait()
    
    def _test_flat_correct(self, ts):
        return self.event in [1, 2]
    
    def _test_flat_incorrect(self, ts):
        return self.event in [4, 8]
    
    def _test_depth_correct(self, ts):
        return self.event in [4, 8]
    
    def _test_depth_incorrect(self, ts):
        return self.event in [1, 2]
