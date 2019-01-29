'''
Basic functions for analyzing BMI3D data.

'''

import numpy as np
from plexon import plexfile, psth
import tables
from riglib.nidaq import parse
from pylab import specgram
import os.path

# Make sure these paths are up to date!
#plx_path = '/media/A805-7C6F/split data/'#'/storage/bmi3d/plexon/'
#hdf_path = '/media/A805-7C6F/split data/'#'/storage/bmi3d/rawdata/hdf/'

plx_path = '/storage/bmi3d/plexon/'
hdf_path = '/storage/bmi3d/rawdata/hdf/'
binned_spikes_path = '/storage/bmi3d/binned_spikes/'


def load_session(session_name):
    '''

    Load all files associated with a recording session and extract timestamps.

    Parameters
    ----------
    session_name : string
        The name of the session of interest without file extension.

    Returns
    -------
    plx : plexon file
        The loaded plexon file.
    hdf : hdf file
        The loaded hdf5 file.
    ts_func : function
        A function that translates plexon timestamps to hdf row indices or vice
        versa for this session.

        Parameters:
        input_times : list of either plx timestamps (floats) or hdf timestamps
        (ints) to translate
        output_type : string ['hdf', 'plx'] specifying which type the output
        should be (should NOT be the same as the input type)

        Returns:
        output : list of either plx or hdf timestamps corresponding to input


    '''

    hdf = tables.openFile(hdf_path + session_name + '.hdf')
    plx = plexfile.openFile(plx_path + session_name + '.plx')

    def sys_eq(sys1, sys2):
        return sys1 in [sys2, sys2[1:]]

    events = plx.events[:].data
    # get system registrations
    reg = parse.registrations(events)
    syskey = None

    # find the key for the task data
    for key, system in reg.items():
        if sys_eq(system[0], 'task'):
            syskey = key
            break

    ts = parse.rowbyte(events)[syskey] 

    # Use checksum in ts to make sure there are the right number of rows in hdf.
    if len(hdf.root.task)<len(ts):
        ts = ts[1:]
    assert np.all(np.arange(len(ts))%256==ts[:,1]), \
        "Dropped frames detected!"

    if len(ts) < len(hdf.root.task):
        print "Warning! Frames missing at end of plx file. Plx recording may have been stopped early."

    ts = ts[:,0]

    # Define a function to translate plx timestamps to hdf and vice versa for
    # this session.
    def ts_func(input_times, output_type):

        if output_type == 'plx':
            if len(input_times)>len(ts):
                input_times = input_times[:len(ts)]
            output = [ts[time] for time in input_times]

        if output_type == 'hdf':
            output = [np.searchsorted(ts, time) for time in input_times]

        return np.array(output)

    # Check for previously saved binned spike file, save one if doesn't exist
    filename = binned_spikes_path+session_name
    #if not os.path.isfile(filename+'.npz'):
    #    save_binned_spike_data(plx, hdf, ts_func, filename)

    return plx, hdf, ts_func#, np.load(filename+'.npz')['spike_counts']

def save_binned_spike_data(plx, hdf, ts, name):
    plxbins = ts(np.array(range(len(hdf.root.task))),'plx')[1:]
    sb = psth.SpikeBin(plx.units, binlen=1/60.)
    bspikes = np.array([bin for bin in plx.spikes.bin(plxbins, sbin=sb)])
    np.savez(name, units=np.array(plx.units), spike_counts=bspikes)

def create_dict(plxfile):
    '''

    Returns dictionary for looking up the index of a particular unit by name.
    
    Parameters
    ----------
    plxfile : plexon file
        The file containing the data.

    Returns
    -------
    udict : dictionary
        The keys are the channel-unit tuples in plxfile.units, and the values
        are the indices of those units in the list. This is to create an easy
        way of locating the index of a specific unit.

    '''

    udict = dict()
    for i, unit in enumerate(plxfile.units):
        udict[tuple(unit)] = i
    return udict

def trials_per_minute(hdf):
    step_s = 60.0 # number of seonds to slide window forward
    step = int(step_s*60)
    window = 5*60*60
    states = hdf.root.task_msgs
    # ignore anything after last reward state
    sessionend = [s['time'] for s in states if s['msg']=='reward'][-1]
    laststate = [i for i, s in enumerate(states) if s['msg']=='reward'][-1]
    states = states[:laststate+1]
    sessionlen = sessionend/60.0
    bins = range(step,sessionend,step)
    output = np.zeros(len(bins))
    reward_times = [msg['time'] for msg in states if msg[0]=="reward"]
    for i, bin in enumerate(bins):
        if bin<window:
            output[i] = len([reward for reward in reward_times if (reward>=0 and reward<bin)])*(window/bin)
        else:
            output[i] = len([reward for reward in reward_times if (reward>=bin-window and reward<bin)])
    inds = np.nonzero(output>.3*np.mean(output))[0]
    output = output[:inds[-1]+1]
    return output/(window/3600), ((np.array(bins)-step)/3600.)[:inds[-1]+1]

def reach_time(hdf, taskname='_multi'):
    step = 10 #trials
    window = 100 #trials
    states = hdf.root.task_msgs
    # ignore anything after last reward state
    sessionend = [s['time'] for s in states if s['msg']=='reward'][-1]
    laststate = [i for i, s in enumerate(states) if s['msg']=='reward'][-1]
    states = states[:laststate+1]
    sessionlen = sessionend/60.0
    reward_times = [msg[1] for msg in states if msg[0]=="reward"]
    reward_inds = [i for i, msg in enumerate(states) if msg['msg']=='reward']

    if '_multi' in taskname:
        targ_states = [i for i, s in enumerate(states) if s['msg']=='target']
        complete_targ_states = [i for i in targ_states if states[i+1]['msg']=='hold']
        reachtimes = np.array([states[i+1]['time'] for i in complete_targ_states]) - np.array([states[i]['time'] for i in complete_targ_states])
        reachtimes = reachtimes/60.
        ntrials = len(reachtimes)

    else:
        ntrials = len(reward_times)   
        reward_inds = [i for i,m in enumerate(states[:]) if m[0]=="reward"]
        reachtimes = np.zeros([len(reward_inds)])
        for i,r in enumerate(reward_inds):
            reachtimes[i] = (states[r-1][1] - states[r-2][1])/60.

    bins = range(step,ntrials,step)
    output = np.zeros(len(bins))
    for i, bin in enumerate(bins):
        if bin<window:
            output[i] = np.mean(np.array([reachtimes[j] for j in range(len(reachtimes)) if (j>=0 and j<bin)]))
        else:
            output[i] = np.mean(np.array([reachtimes[j] for j in range(len(reachtimes)) if (j>=bin-window and j<bin)]))
    return output, (np.array(bins) - step)

def percent_success(hdf,taskname='_multi'):
    step = 10 #trials
    window = 100 #trials
    states = hdf.root.task_msgs[:]
    # ignore anything after last reward state
    sessionend = [s['time'] for s in states if s['msg']=='reward'][-1]
    laststate = [i for i, s in enumerate(states) if s['msg']=='reward'][-1]
    states = states[:laststate+1]

    if '_multi' in taskname:
        pass

    else:
        trial_inds = [i for i, msg in enumerate(states) if msg[0]=="origin_hold"][:-1]
        success_inds = [i for i, trial in enumerate(trial_inds) if states[trial+3][0]=="reward"]
        ntrials = len(trial_inds)
        bins = range(step,ntrials,step)
        output = np.zeros(len(bins))
        for i, bin in enumerate(bins):
            if bin<window:
                output[i] = len([ind for ind in success_inds if (ind>=0 and ind<bin)])*(float(window)/bin)
            else:
                output[i] = len([ind for ind in success_inds if (ind>=bin-window and ind<bin)])
        return output/float(window), (np.array(bins) - step)
    
def angular_error(hdf):
    step = 10 #trials
    window = 100 #trials

    target = hdf.root.task[:]['target']
    cursor = hdf.root.task[:]['cursor']
    # cursor positions of exactly 0,0,0 mean there was no data available
    cursor[cursor[:,0]==0]= np.nan
    cursor_vecs = np.diff(cursor,axis=0)
    target_vecs = target[:-1] - cursor[:-1]

    states = hdf.root.task_msgs[:]
    hasterminus2 = np.any(np.array([s[0] for s in states])=='terminus2')
    rewards = [i for i, s in enumerate(states) if s[0]=='reward']
    if hasterminus2:
        start_inds = [s[1] for s in states[np.array(rewards)-4]]
        end_inds = [s[1] for s in states[np.array(rewards)-1]]
    else:
        start_inds = [s[1] for s in states[np.array(rewards)-2]]
        end_inds = [s[1] for s in states[np.array(rewards)-1]]
    epochs = np.array([start_inds,end_inds]).T

    errors = np.zeros([epochs.shape[0]])
    errorsraw=np.array([])
    cursor_angs_raw=np.array([])
    target_angs_raw=np.array([])
    weightederrors=np.array([])
    magsall=np.array([])

    for i,e in enumerate(epochs):
        cv = cursor_vecs[e[0]:e[1],[0,2]]
        tv = target_vecs[e[0]:e[1],[0,2]]
        cv_ang = np.arctan2(cv[:,1], cv[:,0])
        tv_ang = np.arctan2(tv[:,1], tv[:,0])
        diff_angs =np.abs(tv_ang - cv_ang)
        diff_angs[diff_angs>np.pi] = 2*np.pi - diff_angs[diff_angs>np.pi]

        mags = np.sqrt(cv[:,0]**2+cv[:,1]**2)

        goodinds = ~np.isnan(diff_angs)

        errorsraw = np.concatenate((errorsraw, diff_angs[goodinds]))
        cursor_angs_raw=np.concatenate((cursor_angs_raw, cv_ang[goodinds]))
        target_angs_raw=np.concatenate((target_angs_raw, tv_ang[goodinds]))
        weightederrors=np.concatenate((weightederrors, diff_angs[goodinds]*mags[goodinds]))
        magsall=np.concatenate((magsall, mags[goodinds]))
        errors[i] = np.sum(diff_angs[goodinds]*mags[goodinds])/sum(mags[goodinds])

    #normalize error measure by dividing by mean magnitude and scaling so random movements would have error of 1
    return errors*(2/np.pi)#, errorsraw, cursor_angs_raw,target_angs_raw,weightederrors,magsall



def interp_missing(raw_data, threshold=.5):
    '''

    Interpolates missing motiontracker data and discards bad values.

    Parameters
    ----------
    raw_data : array
        An n x m x 4 (timepoint x marker x coord/condition val) array of raw
        motiontracker data.
    threshold : float (optional)
        The maximum window in sec of missing data to interpolate over. For gaps
        greater than the threshold, nans are inserted.

    Returns
    -------
    processed_data : array
        An n x m x 3 (timepoint x marker x coord) array of interpolated data.

    '''

    import scipy.interpolate as interp

    processed_data = np.zeros([raw_data.shape[0], raw_data.shape[1], 3])
    times = np.arange(len(raw_data))

    for i in range(raw_data.shape[1]):

        # Get the list of condition values for one marker.
        cond = raw_data[:, i, -1]

        # Mask out bad condition values and interpolated values.
        mask = np.logical_and(cond > 0, cond != 4)

        # If marker never has a good value, fill in nans in data.
        if sum(mask) == 0:
            processed_data[:,i, :] = np.nan
        else:
            t, = np.nonzero(mask)
            for dim in range(3):
                # Fit spline to available data.
                spline = interp.UnivariateSpline(t, raw_data[:,i,dim][mask])
                # Interpolate with spline.
                processed_data[:,i,dim] = spline(times)

        # Create vector of ones with length threshold * sample rate (adjusted
        # by 1 to be odd).
        window = np.ones(np.round(threshold*120)*2+1)

        # Use convolutions to mark strings of 0s longer than the threshold.
        mask2 = np.convolve(np.convolve(mask, window, 'same') == 0,
                            window, 'same')
        # Get indices of values to delete and insert nans.
        t, = np.nonzero(mask2)
        processed_data[t,i,:] = np.nan # Insert nans for gaps in good data.

    return processed_data


def get_jointpos(arm_data, ref="elbow_wrist.npz"):
    '''

    Returns the approximate pivot centers of the wrist and elbow joints. Based
    on the reference positions given by ref. There should not be any nans in
    arm markers 1-7 (0 is ok because it's not used here).

    Parameters
    ----------
    arm_data : array
        An n by 8 x 3 (timepoints x arm marker x coord) array of arm position
        data.
    ref : (optional)
        Arm reference file.

    Returns
    -------
    joints : array
        An n x 3 x 3 (timepoints x joint x coord) array of joint positions.
        Joint order is wrist, elbow, shoulder.

    '''

    from riglib.stereo_opengl.xfm import Quaternion

    # Check to make sure all required points exist in arm_data.
    assert np.sum(np.isnan(arm_data[:,[1,4,7]])) == 0, "Nans in data!"

    ref = np.load(ref)
    refarm = ref['arm']

    joints = np.zeros([len(arm_data), 3, 3])

    for i, pts in enumerate(arm_data):
        # Calculate wrist position.
        rot1 = Quaternion.rotate_vecs(refarm[1] - refarm[2], pts[1] - pts[2])
        rot2 = Quaternion.rotate_vecs(rot1 * (refarm[3] - refarm[2]),
                                      pts[3]-pts[2])
        wrist = rot2*rot1*(ref['wrist']-refarm[2]) + pts[2]
        #wrist = pts[1] # Temporary fix for lack of data

        # Calculate elbow position.
        rot3 = Quaternion.rotate_vecs(refarm[4]-refarm[5], pts[4]-pts[5])
        rot4 = Quaternion.rotate_vecs(rot1*(refarm[6]-refarm[5]), pts[6]-pts[5])
        elbow = rot4*rot3*(ref['elbow']-refarm[5]) + pts[5]
        #elbow = pts[6] # Temporary fix for lack of data

        # Calculate shoulder position.
        shoulder = pts[7]

        joints[i,:,:] = np.concatenate((wrist[:,None],
                                       elbow[:,None],
                                       shoulder[:,None]),
                                       axis=1)
    
    ref.close()

    return joints


def find_nan_rows(data):
    '''

    Returns indices of rows not containing nans.

    Parameters:
    -----------
    data : array
        An array of any size with trial # as the first dimension.

    Returns:
    --------
    inds : list
        A list of indices of rows in data not containing any nans.

    '''

    numdim = len(data.shape)
    nanlist = np.isnan(data)
    for i in range(numdim-1):
        nanlist = np.sum(nanlist,axis=1)

    inds = np.nonzero(nanlist==0)[0]

    return inds

def generate_epochs(times, timebefore, timeafter):
    '''

    Returns a list of time windows around the specified timestamps.

    Parameters:
    -----------
    times: array-like, floats
        1D array of timestamps of interest in seconds
    timebefore: float
        Length of time to include before time of interest in seconds
    timeafter: float
        Length of time to include after time of interest in seconds

    Returns:
    --------
    epochs: array-like
        n x 2 array of start and end times of windows in seconds

    '''

    epochs = np.zeros([len(times),2])
    epochs[:,0] = times - timebefore
    epochs[:,1] = times + timeafter
    return epochs

def unit_filter(units):
    '''

    Returns a function that filters spike data for specific unit labels.

    Parameters
    ----------
    units : list of tuples of ints
        List of tuples designating channel-unit combinations. Channels must be
        1-256.

    Returns
    -------
    filt : function
        Filtering function that takes a record array of spike data
        (plx.spikes[timeslice]) and output type as input and returns a list of
        arrays of spike times or waveforms. Each array in the list corresponds
        to a unit in the units parameter.

    '''

    def filt(data,type='spiketimes'):
        if type not in ("spiketimes", "waveforms"):
            raise ValueError("Invalid type name")

        ret = []
        spikes = data.data
        for ch, un in units:
            mask = np.logical_and(spikes['chan'] == ch, spikes['unit'] == un)
            if type=='spiketimes':
                ret.append(spikes[mask]['ts'])
            elif type=='waveforms':
                ret.append(data.waveforms[mask])
        return ret
    return filt


def psth_analysis(plxfile, units, epochs, nbins):
    '''

    Creates PSTHs for units within specified epochs of interest.

    Parameters
    ----------
    plxfile : plexon file
        The plexon file containing the data.
    units : list of tuples of ints
        Each tuple of the list is one channel-unit label (chans go from 1-256).
    epochs : array like
        n x 2 array of start and end times for epochs of interest
    nbins : int
        The number of bins for the final PSTH.

    Returns
    -------
    firing_rates : np array
        A nbins x nunits x nepochs array containing the mean spike counts for
        each bin.

    '''
    if ~np.all(np.diff(epochs)==np.diff(epochs)[0]):
        raise ValueError("All epochs must be the same length")
    firing_rates = np.zeros([nbins, len(units), epochs.shape[0]])
    for i, (start, stop) in enumerate(epochs):
        times = np.linspace(start, stop, nbins)
        sb = psth.SpikeBin(units, binlen=np.diff(times).mean())
        firing_rates[:, :, i] = np.array([bin for bin in plxfile.spikes.bin(times, sbin=sb)])

    return firing_rates

def lfp_epochs(plxfile, channel, epochs):
    '''

    Pulls out LFP data for specified channel within specified epochs of interest.

    Parameters
    ----------
    plxfile : plexon file
        The plexon file containing the data.
    channel : list or array
        Channel number (0-255).
    epochs : array like
        n x 2 array of start and end times for epochs of interest

    Returns
    -------
    lfp : np array
        A time x nunits x nepochs array containing the LFP signal for each channel

    '''

    nsamples = len(plxfile.lfp[epochs[0,0]:epochs[0,1]])
    lfp = np.zeros([nsamples,epochs.shape[0]])
    for i, (start, stop) in enumerate(epochs):
        print start,stop
        lfp[:,i] = plxfile.lfp[start:stop+.01].data[:nsamples,channel]
    return lfp


def baseline_fr(plxfile, units, epoch, binsize=1.0):
    '''

    Calculates the mean firing rate and standard deviation for specified units
    across a long epoch.

    Parameters
    ----------
    plxfile : plexon file
        The plexon file containing the data.
    units : list of tuples of ints
        Each tuple of the list is one channel-unit label (chans go from 1-256).
    epoch : array
        Contains a start and end time for epoch of interest. Must be longer
        than bin size (and should be much longer).
    binsize : float (optional)
        The length of each bin used in the standard deviation calculation.

    Returns
    -------
    mean_firing_rates : np array
        A 1D array containing the mean firing rate in hz for each unit across
        all bins.
    sd : array
        A 1D array containing the standard deviation in hz of each unit's
        firing rate across all epochs
    firing_rates : array
        A 2D array containing all FR measurements for each bin (rows) by unit (cols)
        
    '''


    # Truncate epoch so last bin isn't short.
    starttime = epoch[0]
    endtime = epoch[1] - (epoch[1] % binsize)

    fr = psth_analysis(plxfile, units, epoch[None,:], (endtime-starttime)/binsize)
    fr = fr/binsize
    
    sd = np.squeeze(np.std(fr,axis=0))
    mean_firing_rates = np.squeeze(np.mean(fr,axis=0))

    return mean_firing_rates, sd, np.squeeze(fr)


    # epochs = []
    # for bin in range(int((endtime - starttime) / binsize)):
    #     edge1 = bin*binsize
    #     edge2 = (bin+1)*binsize
    #     epochs.append((starttime + edge1, starttime + edge2))

    # filt = unit_filter(units)
    # firing_rates = np.zeros([len(units), len(epochs)])
    # for i, epoch in enumerate(epochs):
    #     for j, spikes in enumerate(filt(plxfile.spikes[epoch[0]:epoch[1]])):
    #         firing_rates[j,i] = len(spikes)/binsize
    # sd = np.squeeze(np.std(firing_rates,axis=1))
    # mean_firing_rates = np.squeeze(np.mean(firing_rates,axis=1))
    # return mean_firing_rates, sd, firing_rates


def mean_fr(plxfile, units, epochs):
    '''

    Calculates the mean firing rate for specified units across during specified
    epochs.

    Parameters
    ----------
    plxfile : plexon file
        The plexon file containing the data.
    units : list of tuples of ints
        Each tuple of the list is one channel-unit label (chans go from 1-256).
    epochs : list of tuples of floats
        Each tuple in the list contains a start and end time for an epoch of
        interest (plx timestamps).

    Returns
    -------
    firing_rates : np array
        A 2D array (unit x epoch) containing the mean firing rate in hz for
        each unit in each epoch.

    '''

    filt = unit_filter(units)
    firing_rates = np.zeros([len(units), len(epochs)])
    for i, epoch in enumerate(epochs):
        for j, spikes in enumerate(filt(plxfile.spikes[epoch[0]:epoch[1]])):
            firing_rates[j,i] = len(spikes)/(epoch[1]-epoch[0])
    return firing_rates


def waveforms(plxfile, units):
    filt = unit_filter(units)
    result = np.zeros([32,len(units),2])
    for i, spikes in enumerate(filt(plxfile.spikes[10:310],type='waveforms')):
        result[:,i,0] = np.mean(spikes, axis=0)
        result[:,i,1] = np.std(spikes, axis=0)
        if spikes.shape[0]<20:
            print('Warning: less than 20 waveforms found for ' + str(units[i]))
    return result


def gen_spcgrm(lfpdat,srate=1000.,cutoffs=(0,250),binsize=50):

    for i in range(lfpdat.shape[1]):
        pxxx,freqs,bins,im=specgram(lfpdat[:,i],Fs=srate,NFFT=binsize,noverlap=0)
        if i==0:
            spec = pxxx.copy()
        spec = pxxx + spec
    
    return spec/lfpdat.shape[1],freqs,bins

