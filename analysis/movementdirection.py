'''Docstring for movementdirection'''

import tables
import numpy as np
import scipy.signal as sig

def load():
    seq = np.load("/Users/HGM/code/data/20120502_1616.npy")[:,[0,2], 0] / .35
    h5 = tables.openFile('/Users/HGM/code/data/20120508_17-17.hdf')
    return seq, h5

def success_rate(numbins, h5):
    inds = np.nonzero([r['msg'] == 'origin' for r in h5.root.motiontracker_msgs])[0]
    inds2 = np.nonzero([r['msg'] == 'reward' for r in h5.root.motiontracker_msgs])[0] - 2
    success = np.zeros(len(inds))
    success[np.searchsorted(inds,inds2)] = 1
    rate = []
    binsize = np.round(len(inds)/numbins)
    for b in range(numbins):
        rate.append(np.sum(success[b*binsize:(b+1)*binsize])/float(binsize))
    return rate

def get_data(targets, h5):
    inds = np.nonzero([r['msg'] == 'origin' for r in h5.root.motiontracker_msgs])[0]
    inds2 = np.nonzero([r['msg'] == 'reward' for r in h5.root.motiontracker_msgs])[0] - 2
    t = np.searchsorted(inds,inds2, side='left')
    success = np.zeros(len(inds))
    success[t] = 1
    fail = np.zeros(len(inds))
    count = 0
    for i, s in enumerate(success):
        if s == 0:
            count += 1
        else:
            count = 0
        if count == 10:
            fail[i] = 1
            count = 0
        else:
            fail[i] = 0
    newtargs = success + fail
    np.hstack([1, newtargs[0:-2]])
    targs = np.zeros(targets.shape)
    count = 0
    for i, v in enumerate(newtargs):
        targs[i,:] = targets[count]
        if v == 1:
            count +=1
    targets = targs 
    inds3 = np.searchsorted(inds,inds2)
    origins = zip(h5.root.motiontracker_msgs[inds]['time'], h5.root.motiontracker_msgs[inds+1]['time'])

    mask = np.zeros(len(targets), dtype=bool)
    mask[inds3] = True
    o = []
    for t, i, (s,e) in zip(targets, mask, origins):
        if e-s > 8 and i:
            signal = sig.resample(h5.root.motiontracker[s:e, 6][:,[0,2]], (e-s)/4., axis=0)
            o.append((t, signal))
    return o

def calc_diff(data):
    return np.hstack([np.sqrt((np.diff(d, axis=0)**2).sum(1)) for target, d in data])

def make_hist(origins):
    angles = []
    move_angles = []
    targ_angles = []
    start = []
    for target, d in origins:
        sp = d[0,:]
        start.append(sp)
        dist2 = np.sqrt((d[1:,:] - sp)[:,0]**2 + (d[1:,:] - sp)[:,1]**2)
        if np.max(dist2)>50:
            cutoff = np.nonzero(dist2>50)[0][0]
        else:
            cutoff = len(dist2-1)
        #dist = np.sqrt((np.diff(d, axis=0)**2).sum(1))
        diffs = np.diff(d, axis=0)[:cutoff][np.logical_and(20 < dist2[:cutoff], dist2[:cutoff] < 50)]
        tdiff = (target - d[:-1])[:cutoff][np.logical_and(20 < dist2[:cutoff], dist2[:cutoff] < 50)]
        a1 = np.arctan2(diffs[:,1],diffs[:,0])
        a2 = np.arctan2(tdiff[:,1],tdiff[:,0])
        angles.append(a1-a2)
        move_angles.append(a1)
        targ_angles.append(a2)
    angles = np.hstack(angles)
    angles[angles < -np.pi] += 2*np.pi
    angles[angles > np.pi] -= 2*np.pi
    move_angles = np.hstack(move_angles)
    targ_angles = np.hstack(targ_angles)
    return angles, move_angles, targ_angles

if __name__ == "__main__":
    seq, h5 = load()
    nseq = seq.copy()
    np.random.shuffle(nseq)
    origins = get_data(seq, h5)
    angles, a1, a2 = make_hist(origins)