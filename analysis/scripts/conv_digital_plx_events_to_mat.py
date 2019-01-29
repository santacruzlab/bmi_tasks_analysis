#!/usr/bin/python
import numpy as np
'''
Save the HDF row times to .mat format
'''

fn = sys.argv[1]

data = np.load(fn)



def get_taskdata_key(reg):
    returnkey = None
    for key, system in reg.items():
        if system[0] in ['task', 'ask']:
            returnkey = key
    return returnkey

plx = plexfile.openFile(fn)
events = plx.events[:].data
reg = parse.registrations(events)
key = get_taskdata_key(reg)
assert (key is not None), "No task data registered in plx file!"
rowtimes = parse.rowbyte(events)[get_taskdata_key(reg)][:,0]

scipy.io.savemat(fn.rstrip('.plx')+'.mat', {'rowtimes':rowtimes})
