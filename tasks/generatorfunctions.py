import numpy as np

from utils.constants import *

'''
All target-capture generators should return arrays of ntrials x ntargspertrial x 3. For example:
a 2D task where each trial contains a string of 3 targets should be of shape n x 3 x 3, with
the second element of the last dimension always 0 to fix the targets in a vertical plane.
All units should be in cm.
'''





def seq_to_gen(seq):
    n_trials = len(seq)
    for k in range(n_trials):
        yield seq[k]

