# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:04:08 2021

@author: yc25258

In this file, it shows how to use neo package to read Ripple recordings.

NOTE: all the recording files in the same folder, even if I only specify 
      one file format (e.g., .nev), it will read all the recording files.

Example usage to obtain the data corresponding to Ripple Neural Interface Processor:
    block.segments[0].analogsignals[0]       --- Analog input or LFP or Raw data stream
    block.segments[0].spiketrains[Ch].times  --- Spike trains
    block.segments[0].events[0].times        --- Digital input (parallel)

"""
from neo import io
import numpy as np
import os

#%% Load the file (Example file: datafile0012.XXX)
path = os.getcwd()
r = io.BlackrockIO(path + "\datafile0012.nev")
block = r.read_block()

#%% Get the spiking channels
Ch_spk = [block.segments[0].spiketrains[i].annotations['channel_id'] for i in range(len(block.segments[0].spiketrains))]
Ch_spk = np.array(Ch_spk)
