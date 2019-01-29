
'''
This module contains functions to extract and analyze the visual and tactile
receptive fields of individual spiking units.

'''

from __future__ import division
import numpy as np
from plexon import plexfile
import tables
from riglib.nidaq import parse
from basicanalysis import *

def tap_times(plxfile, min_interval=200, min_duration=100, max_duration=2000):
    '''

    Returns the epochs where the tactile stimulator switch is depressed
    (encoded on the plx AI01 channel).

    Parameters
    ----------
    plxfile : plexon file
        The plexon file containing the data.
    min_interval : int (optional)
        Minimum time in ms since previous tap required.
    min_duration : int (optional)
        Minimum length in ms of tap required.
    max_duration : int (optional)
        Maximum length in ms of tap required.

    Returns
    -------
    taps : list of tuples of floats
        The epochs in the plx file when the switch is depressed.

    '''

    # Extract data and timestamps from AI01 (microswitch) channel.
    ai = plxfile.analog[0:plxfile.length,0].data
    times = plxfile.analog[0:plxfile.length,0].time

    # Mask out times when switch is not depressed and get list of on and off
    # transition indices.
    mask = (ai < -1000).astype(float)
    #mask = (ai > 1000).astype(float) #If voltage direction of switch is reversed
    raw_onsets = np.nonzero((np.diff(mask, axis=0)>0).astype(float))[0]
    raw_offsets = np.nonzero((np.diff(mask, axis=0)<0).astype(float))[0]

    # Make sure there are equal numbers of onsets and offsets.
    assert len(raw_onsets)==len(raw_offsets), \
        "Problem matching up switch on and off times."

    # Filter out small gaps in presses that are within 50ms. (Not true releases,
    # usually caused by inconsistent pressure on switch.) 
    onmask = np.append(np.array([True]),
                       (raw_onsets[1:] - raw_offsets[:-1]) >= 50)
    onsets = raw_onsets[onmask]
    offmask = np.append((raw_onsets[1:] - raw_offsets[:-1]) >= 50,
                        np.array([True]))
    offsets = raw_offsets[offmask]

    # Get rid of taps that don't meet the min interval criteria.
    mask = np.append(np.array([True]),
                     (onsets[1:] - offsets[:-1]) >= min_interval)
    onsets = onsets[mask]
    offsets = offsets[mask]

    # Get rid of taps that don't meet the min and max duration criteria.
    mask = np.logical_and((offsets - onsets) >= min_duration,
                          (offsets - onsets) <= max_duration)
    onsets = onsets[mask]
    offsets = offsets[mask]

    # Get the timestamps of the selected indices.
    taps = zip(times[onsets], times[offsets])
    return taps


def marker_locations(hdffile, timestamps, marker_nums):
    '''

    Pulls out the 3D coordinates for the specified markers at each specified
    time.

    Parameters
    ----------
    hdffile : hdf5 file
        The hdf5 file containing the data.
    timestamps : list of ints
        A list of hdf file row indices corresponding to the times of interest.
    marker_nums : list of ints
        A list of marker indices to include.

    Returns
    -------
    locations : array
        A 3D array (timestamp x marker x spatial coordinate) of marker locations
        at specified times.

    '''

    # Check to make sure the requested markers exist in the file.
    assert np.max(marker_nums) < hdffile.root.motiontracker.shape[1], \
        "File does not contain all requested markers."

    # Get the raw phasespace data and interpolate it.
    raw_data = hdffile.root.motiontracker[:,marker_nums,:]
    data = interp_missing(raw_data)

    locations = data[timestamps, :, :]
    return locations


def map_stimuli(arm_data, stimuli_locations):
    '''

    Maps 3D points around the arm onto the arm's surface.

    Parameters
    ----------
    arm_data : array
        An n x 8 x 3 (stimulus time x arm marker x coordinate) array of arm
        marker locations at each stimulus time.
    stimuli_locations : array
        An n x 3 (stimulus time x coordinate) array of stimulus locations.

    Returns
    -------
    remapped_points : array
        An n x 2 (stimulus time x coordinate) array of the stimuli_locations
        mapped to the closest point on the surface of the arm. The first
        coordinate is the distance along the arm (0 is shoulder, increases
        toward wrist) and the second coordinate is the angle around the arm's
        circumference.

    '''

    from riglib.stereo_opengl.xfm import Quaternion

    # Make sure there is no missing data.
    assert np.sum(np.isnan(stimuli_locations)) == 0, "Nans in data!"

    # Calculate joint positions.
    joints = get_jointpos(arm_data)

    # Sort points onto upper arm vs. forearm.
    mask = np.zeros([len(stimuli_locations)])
    for i, (position, stim) in enumerate(zip(joints, stimuli_locations)):
        wrist = position[0,:]
        elbow = position[1,:]
        shoulder = position[2,:]

        # Calculate shortest distance from stim to forearm. Based on code found
        # here: http://nodedangles.wordpress.com/2010/05/16/measuring-distance-
        # from-a-point-to-a-line-segment/
        fa_length = np.sqrt((wrist[0] - elbow[0])**2 + \
            (wrist[1] - elbow[1])**2 + (wrist[2] - elbow[2])**2)
        u = ((stim[0] - elbow[0]) * (wrist[0] - elbow[0]) + \
            (stim[1] - elbow[1]) * (wrist[1] - elbow[1]) + \
            (stim[2] - elbow[2]) * (wrist[2] - elbow[2]))/fa_length**2
        if (u <= 0.0) or (u >= 1.0):
            end_dist1 = np.sqrt((stim[0] - elbow[0])**2 + \
                (stim[1] - elbow[1])**2 + (stim[2] - elbow[2])**2)
            end_dist2 = np.sqrt((stim[0] - wrist[0])**2 + \
                (stim[1] - wrist[1])**2 + (stim[2] - wrist[2])**2)
            if end_dist1 < end_dist2:
                fa_dist = end_dist1
            else:
                fa_dist = end_dist2
        else:
            intersectx = elbow[0] + u * (wrist[0] - elbow[0])
            intersecty = elbow[1] + u * (wrist[1] - elbow[1])
            intersectz = elbow[2] + u * (wrist[2] - elbow[2])
            fa_dist = np.sqrt((stim[0] - intersectx)**2 + \
                (stim[1] - intersecty)**2 + (stim[2] - intersectz)**2)

        # Calculate shortest distance from stim to upper arm. 
        ua_length = np.sqrt((elbow[0] - shoulder[0])**2 + \
            (elbow[1] - shoulder[1])**2 + (elbow[2] - shoulder[2])**2)
        u = ((stim[0] - shoulder[0]) * (elbow[0] - shoulder[0]) + \
            (stim[1] - shoulder[1]) * (elbow[1] - shoulder[1]) + \
            (stim[2] - shoulder[2]) * (elbow[2] - shoulder[2]))/ua_length**2
        if (u <= 0.0) or (u >= 1.0):
            end_dist1 = np.sqrt((stim[0] - shoulder[0])**2 + \
                (stim[1] - shoulder[1])**2 + (stim[2] - shoulder[2])**2)
            end_dist2 = np.sqrt((stim[0] - elbow[0])**2 + \
                (stim[1] - elbow[1])**2 + (stim[2] - elbow[2])**2)
            if end_dist1 < end_dist2:
                ua_dist = end_dist1
            else:
                ua_dist = end_dist2
        else:
            intersectx = shoulder[0] + u * (elbow[0] - shoulder[0])
            intersecty = shoulder[1] + u * (elbow[1] - shoulder[1])
            intersectz = shoulder[2] + u * (elbow[2] - shoulder[2])
            ua_dist = np.sqrt((stim[0] - intersectx)**2 + \
                (stim[1] - intersecty)**2 + (stim[2] - intersectz)**2)
        # Mask is true if point closer to upper arm, false if closer to forearm.
        if ua_length < fa_length:
            mask[i] = 1
        else:
            mask[i] = 0

    # Compute locations for forearm.
    forearm_points = np.zeros([sum(mask==0), 2])
    for i, (position, stim) in enumerate(zip(joints[mask==0],
                                             stimuli_locations[mask==0])):

        # Calculate rotation to orient arm link along z axis and use it to
        # rotate stimulation point.
        rot = Quaternion.rotate_vecs(position[0] - position[1], (0,0,1))
        stim = rot * (stim - position[1])

        # First dimension is distance from elbow toward wrist along forearm.
        forearm_points[i,0] = stim[2]
        
        # The second dimension is the angular position around the circumference
        # of the arm.
        forearm_points[i,1] = np.arctan2(stim[1] - position[0,1],
                                 stim[0] - position[0,0])


    # Compute locations for upper arm.
    upperarm_points = np.zeros([sum(mask==1), 2])
    for i, (position, stim) in enumerate(zip(joints[mask==1],
                                             stimuli_locations[mask==1])):

        # Calculate rotation to orient arm link along z axis and use it to
        # rotate stimulation point.
        rot = Quaternion.rotate_vecs(position[1] - position[2], (0,0,1))
        stim = rot * (stim - position[2])

        # First dimension is distance from shoulder toward elbow.
        upperarm_points[i,0] = stim[2]
        
        # The second dimension is the angular position around the circumference
        # of the arm.
        upperarm_points[i,1] = np.arctan2(stim[1] - position[1,1],
                                 stim[0] - position[1,0])
    #print upperarm_points[:,0].max(), upperarm_points[:,0].min()

    # Shift forearm points along length axis by length of upper arm.
    ua_length = np.mean(np.sqrt((joints[:,1,0] - joints[:,2,0])**2 + \
        (joints[:,1,1] - joints[:,2,1])**2 + \
        (joints[:,1,2] - joints[:,2,2])**2))
    forearm_points[:,0] = forearm_points[:,0] + ua_length

    # Put points back in original order.
    remapped_points = np.zeros([len(stimuli_locations), 2])
    remapped_points[mask==0] = forearm_points
    remapped_points[mask==1] = upperarm_points
        
    return remapped_points

def match_taps(seq, numtimes, taptimes):
    '''

    Matches switch press stimuli with the corresponding trial by using the
    timestamps of the state changes and switch presses. If there are multiple
    stimuli for a single trial the last one is chosen.

    Parameters
    ----------
    seq: list of ints
        stimulus zone numbers from sequence file
    numtimes: list of floats (must be same length as seq)
        plx timestamps for the start of each 'numdisplay' state
    taptimes: list of tuples of floats
        start and end plx timestamps of switch press epochs

    Returns
    -------
    stims: list of floats
        stimulus onset times
    seq2: list of ints
        corresponding zone numbers 

    '''
    staps = np.array([tap[0] for tap in taptimes]) #get start times of taps
    stims = []
    seq2 = []
    for i, num in enumerate(seq[:-1]): #for each stimulus in sequence
        stime = numtimes[i] #time corresponding number is displayed
        etime = numtimes[i+1] #time next number is displayed
        ind1 = np.searchsorted(staps, stime) #find indices of taps occurring between two number states
        ind2 = np.searchsorted(staps, etime, side='right')
        taps = taptimes[ind1:ind2] #list of tap epochs corresponding to current stim in sequence
        if len(taps)>0: #if there was at least one tap
            stims = stims + [taps[-1]] #add last one to output list
            seq2 = seq2 + [seq[i]] #add current seq element to output list
    return stims, seq2


def sort_stimuli(seq):
    '''

    Sorts a sequence of ints by value.

    Parameters
    ----------
    seq: list of ints
        stimulus zone numbers from sequence file

    Returns
    -------
    indices: array-like
        Shape is [Unique values x number of values in a group], elements are the
        indices in the original sequence. The number of columns is equal to the
        size of the smallest group after sorting (extra values are ignored).
    order: list of ints
        Corresponding unique values for each row of indices. 

    '''
    order = list(set(seq))
    min_count = 500
    indices = np.zeros([len(order),500])
    for i, zone in enumerate(order):
        temp = np.nonzero(np.array(seq)==zone)[0]
        if len(temp)<min_count:
            min_count = len(temp)
        indices[i,:min_count] = temp[:min_count]
    indices = indices[:,:min_count]
    return indices, order


def zone_to_coords(zonenums):

    # old sleeve
    # mapping = {1:np.array([[0.923,0.0],[0.923,0.25],[1.0,0.25],[1.0,0.0]]),
    #            2:np.array([[0.923,0.25],[0.923,0.5],[1.0,0.5],[1.0,0.25]]),
    #            5:np.array([[0.846,0.0],[0.846,0.25],[0.923,0.25],[0.923,0.0]]),
    #            6:np.array([[0.846,0.25],[0.846,0.5],[0.923,0.5],[0.923,0.25]]),
    #            9:np.array([[0.769,0.0],[0.769,0.25],[0.846,0.25],[0.846,0.0]]),
    #            10:np.array([[0.769,0.25],[0.769,0.5],[0.846,0.5],[0.846,0.25]]),
    #            13:np.array([[0.692,0.0],[0.692,0.25],[0.769,0.25],[0.769,0.0]]),
    #            14:np.array([[0.692,0.25],[0.692,0.5],[0.769,0.5],[0.769,0.25]]),
    #            17:np.array([[0.615,0.0],[0.615,0.25],[0.692,0.25],[0.692,0.0]]),
    #            18:np.array([[0.615,0.25],[0.615,0.5],[0.692,0.5],[0.692,0.25]]),
    #            21:np.array([[0.538,0.0],[0.538,0.167],[0.615,0.167],[0.615,0.0]]),
    #            22:np.array([[0.538,0.167],[0.538,0.333],[0.615,0.333],[0.615,0.167]]),
    #            23:np.array([[0.538,0.333],[0.538,0.5],[0.615,0.5],[0.615,0.333]]),
    #            27:np.array([[0.462,0.0],[0.462,0.167],[0.538,0.167],[0.538,0.0]]),
    #            28:np.array([[0.462,0.167],[0.462,0.333],[0.538,0.333],[0.538,0.167]]),
    #            29:np.array([[0.462,0.333],[0.462,0.5],[0.538,0.5],[0.538,0.333]]),
    #            33:np.array([[0.385,0.0],[0.385,0.167],[0.462,0.167],[0.462,0.0]]),
    #            34:np.array([[0.385,0.167],[0.385,0.333],[0.462,0.333],[0.462,0.167]]),
    #            35:np.array([[0.385,0.333],[0.385,0.5],[0.462,0.5],[0.462,0.333]]),
    #            39:np.array([[0.308,0.0],[0.308,0.167],[0.385,0.167],[0.385,0.0]]),
    #            40:np.array([[0.308,0.167],[0.308,0.333],[0.385,0.333],[0.385,0.167]]),
    #            41:np.array([[0.308,0.333],[0.308,0.5],[0.385,0.5],[0.385,0.333]]),
    #            45:np.array([[0.231,0.0],[0.231,0.167],[0.308,0.167],[0.308,0.0]]),
    #            46:np.array([[0.231,0.167],[0.231,0.333],[0.308,0.333],[0.308,0.167]]),
    #            47:np.array([[0.231,0.333],[0.231,0.5],[0.308,0.5],[0.308,0.333]]),
    #            51:np.array([[0.154,0.0],[0.154,0.167],[0.231,0.167],[0.231,0.0]]),
    #            52:np.array([[0.154,0.167],[0.154,0.333],[0.231,0.333],[0.231,0.167]]),
    #            53:np.array([[0.154,0.333],[0.154,0.5],[0.231,0.5],[0.231,0.333]]),
    #            57:np.array([[0.077,0.0],[0.077,0.167],[0.154,0.167],[0.154,0.0]]),
    #            58:np.array([[0.077,0.167],[0.077,0.333],[0.154,0.333],[0.154,0.167]]),
    #            63:np.array([[0.0,0.0],[0.0,0.167],[0.077,0.167],[0.077,0.0]]),
    #            64:np.array([[0.0,0.167],[0.0,0.333],[0.077,0.333],[0.077,0.167]])}

    # return np.array([(np.mean(mapping[num][:,0]),np.mean(mapping[num][:,1])) for num in zonenums])


    mapping = {1:(24, 3), 2:(24, 5), 3:(22,5), 4:(22,3), 5:(20,3), 6:(20,5),
                7:(18,5), 8:(18,3), 9:(16,2), 10:(16,4), 11:(16,6), 12:(14,6),
                13:(14,4), 14:(14,2), 15:(12,2), 16:(12,4), 17:(12,6),
                18:(10,7), 19:(10,5), 20:(10,3), 21:(10,1), 22:(8,1),
                23:(8,3), 24:(8,5), 25:(8,7), 26:(6,7), 27:(6,5), 28:(6,3),
                29:(6,1), 30:(4,1), 31:(4,3), 32:(4,5), 33:(4,7), 34:(2,8),
                35:(2,6), 36:(2,4), 37:(2,2), 38:(2,0), 39:(0,0), 40:(0,2),
                41:(0,4), 42:(0,6), 43:(0,8), 44:(4,9), 45:(6,9), 46:(8,9),
                47:(10,9), 48:(12,8), 49:(14,8), 50:(16,8), 51:(18,7),
                52:(20,7), 53:(22,7), 54:(24,7)}

    return np.array([mapping[num] for num in zonenums])
    
    



def process_session(filename):
    #plx, hdf, ts = open_files(('/Users/HGM/Downloads/' + filename + '.plx'), ('/Users/HGM/Downloads/' + filename + '.hdf'))
    plx, hdf, ts = basicanalysis.load_session('cart20120829_01')
    taptimes_raw, baselines = get_tap_times(plx)
    taps, arm, taptimes = calc_tap_locations(taptimes_raw, hdf.root.motiontracker[:,0:8,:], hdf.root.motiontracker[:,8:-1,:], ts)
    #results = norm_forearm(arm, taps)
    #u,s,v = map_pts_to_arm(arm, taps)
    return plx, hdf, ts, taptimes_raw, baselines, taps, arm, taptimes, results


def kde(pts, wts, sigma = .05, **kwargs):
    from stats import mvn_norm
    from matplotlib import pyplot as plt
    funcs = [mvn_norm(pt, np.diag([sigma, 1.4 / (2*np.pi) * sigma])) for pt in pts]
    grid = np.mgrid[-np.pi:np.pi:512j, 0:1.4:512j].T
    out = np.zeros((512, 512))
    for i, (wt, f) in enumerate(zip(wts, funcs)):
        out += wt*f(grid)
        print i
    plt.clf(); plt.subplot(1, 2, 1)
    plt.imshow(out.T, aspect='auto', origin='lower', extent=(-np.pi, np.pi, 0, 1.4), **kwargs)
    plt.subplot(1,2,2)
    plt.scatter(pts[:,0], pts[:,1], c=wts, vmin=0, vmax=6, **kwargs)
    plt.colorbar(); plt.xlim(-np.pi, np.pi); plt.ylim(0,1.4)
    return out.T




from scipy.spatial import cKDTree as KDTree
    # http://docs.scipy.org/doc/scipy/reference/spatial.html

class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
    invdisttree = Invdisttree( X, z )  -- data points, values
    interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
        interpolates z from the 3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3)
            / (1/d1 + 1/d2 + 1/d3)
            = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        q may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        p: use 1 / distance**p
        weights: optional multipliers for 1 / distance**p, of the same shape as q
        stat: accumulate wsum, wn for average weights

    How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

    p=1, p=2 ?
        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3)
            / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.

    Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std() .

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.

    """
# anykernel( dj / av dj ) is also scale-free
# error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]

#...............................................................................
if __name__ == "__main__":
    import sys

    N = 10000
    Ndim = 2
    Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = 1  # weights ~ 1 / distance**p
    cycle = .25
    seed = 1

    exec "\n".join( sys.argv[1:] )  # python this.py N= ...
    np.random.seed(seed )
    np.set_printoptions( 3, threshold=100, suppress=True )  # .3f

    print "\nInvdisttree:  N %d  Ndim %d  Nask %d  Nnear %d  leafsize %d  eps %.2g  p %.2g" % (
        N, Ndim, Nask, Nnear, leafsize, eps, p)

    def terrain(x):
        """ ~ rolling hills """
        return np.sin( (2*np.pi / cycle) * np.mean( x, axis=-1 ))

    known = np.random.uniform( size=(N,Ndim) ) ** .5  # 1/(p+1): density x^p
    z = terrain( known )
    ask = np.random.uniform( size=(Nask,Ndim) )

#...............................................................................
    invdisttree = Invdisttree( known, z, leafsize=leafsize, stat=1 )
    interpol = invdisttree( ask, nnear=Nnear, eps=eps, p=p )

    print "average distances to nearest points: %s" % \
        np.mean( invdisttree.distances, axis=0 )
    print "average weights: %s" % (invdisttree.wsum / invdisttree.wn)
        # see Wikipedia Zipf's law
    err = np.abs( terrain(ask) - interpol )
    print "average |terrain() - interpolated|: %.2g" % np.mean(err)

    # print "interpolate a single point: %.2g" % \
    #     invdisttree( known[0], nnear=Nnear, eps=eps )