from riglib.plexon import plexfile
from riglib.nidaq import parse
import scipy.io

import sys
fn = sys.argv[1]


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