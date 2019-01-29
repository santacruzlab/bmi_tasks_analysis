import time
import cPickle

import pygame
import numpy as np

from riglib import reward
from riglib.experiment import traits, Sequence, Pygame
from riglib.calibrations import crossval, ThinPlateEye

from riglib.db.json_param import Parameters

def randcoords(exp, bbox=(0,0,1920,1080)):
    h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
    while True:
        yield ((np.random.rand(2) * (w,h))+bbox[:2]).astype(int)

def gencoords(length, bbox=(0,0,1920, 1080)):
    h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
    return np.random.rand(length, 2)*(w,h) + bbox[:2]

shrinklen = 1.
init_dot = 100.
freq = 100

class RedGreen(Sequence, Pygame):
    status = dict(
        wait = dict(start_trial="pretrial", premature="penalty", stop=None),
        trial = dict(correct="reward", timeout="penalty"),
        reward = dict(post_reward="wait"),
        penalty = dict(post_penalty="wait"),
    )

    colors = traits.Array(shape=(2, 3), value=[[255,0,0],[0,255,0]],
        desc="Tuple of colors (c1, c2) where c* = [r,g,b] between 0 and 1")
    dot_radius = traits.Int(100, desc='dot size')
    delay_range = traits.Tuple((0.5, 5.), 
        desc='delay before switching to second color will be drawn from uniform distribution in this range')

    def _while_pretrial(self):
        self.surf.fill(self.background)
        right = [self.next_trial[0] + 1920, self.next_trial[1]]
        ts = time.time() - self.start_time
        dotsize = (init_dot - self.dot_radius) * (shrinklen - min(ts, shrinklen)) + self.dot_radius
        if (np.mod(np.round(ts*1000),freq) < freq/2):
            pygame.draw.circle(self.surf, self.colors[0], self.next_trial, int(dotsize))
            pygame.draw.circle(self.surf, self.colors[0], right, int(dotsize))
        self.flip_wait()
    
    def _while_trial(self):
        self.surf.fill(self.background)
        right = [self.next_trial[0] + 1920, self.next_trial[1]]
        ts = time.time() - self.start_time
        if (np.mod(np.round(ts*1000),freq) < freq/2):
            pygame.draw.circle(self.surf, self.colors[1], self.next_trial, self.dot_radius)
            pygame.draw.circle(self.surf, self.colors[1], right, self.dot_radius)
        self.flip_wait()
    
    def _start_pretrial(self):
        self._wait_time = np.random.rand()*abs(self.delay_range[1]-self.delay_range[0]) + self.delay_range[0]
    
    def _test_correct(self, ts):
        return self.event is not None
    
    def _test_go(self, ts):
        return ts > self._wait_time + shrinklen

    def _test_premature(self, ts):
        return self.event is not None

class EyeCal(RedGreen):
    def __init__(self, *args, **kwargs):
        super(EyeCal, self).__init__(*args, **kwargs)
        self.actual = []
        self.caldata = []
    
    def _start_trial(self):
        self._last_data = self.eyedata.get()[-20:].mean(0)

    def _start_reward(self):
        self.actual.append(self.next_trial)
        self.caldata.append(self._last_data)
        super(EyeCal, self)._start_reward()
    
    def _start_None(self):
        super(EyeCal, self)._start_None()
        #Log the data for now
        cPickle.dump(dict(actual=self.actual, cal=self.caldata), open("/tmp/eyecal.pkl", "w"), 2)

    def cleanup(self, database, saveid, subject=None, **kwargs):
        assert len(self.actual) == len(self.caldata)
        #divide to be in normalized screen coordinates, -1->0->1
        self.actual = np.array(self.actual) / [1920., 1080.] * 2 - 1
        #Crossvalidate for smoothness parameter, return calibration
        calibration, smooth, ccs = crossval(ThinPlateEye, self.caldata, self.actual)
        
        params = Parameters.from_dict(calibration.__getstate__())
        database.save_cal(subject, calibration.system, "ThinPlateEye", params.to_json())