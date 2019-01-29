import tables
import numpy as np
import basicanalysis
from riglib.nidaq import parse
from plexon import plexfile
import scipy


def normed_movement_response(plx,hdf, task):
    print "parsing hdf"
    oh,m,th,pen = get_movement_periods(hdf, task)
    print "parsing plx"
    oh2,m2,th2,pen2 = get_movement_periods(plx, task)
    targets = np.array([hdf.root.task[p[0]]['target'][[0,2]] for p in m])

    print "getting spike data"
    spikedata = [plx.spikes[p[0]-.1:p[0]+.5].data for p in m2]

    # Find normalized firing rates for each unit and each movement direction
    print "getting firing rates"
    rates = np.zeros([len(targets),len(plx.units)])
    for i in range(len(targets)):
        trial = i
        triallen = m2[trial][1] - m2[trial][0]
        for j in range(len(plx.units)):
            unit = plx.units[j]
            spcount = len(spikedata[trial][np.logical_and(spikedata[trial]['chan'] == unit[0], spikedata[trial]['unit'] == unit[1])]['ts'])
            rates[i,j] = spcount/triallen
        
    # z score
    rmean = np.mean(rates,axis=0)
    rstd = np.std(rates,axis=0)
    rates = (rates - rmean)/rstd

    return rates, targets

def modulation_depths(plx,hdf, task, control=False):
    rates, targets = normed_movement_response(plx,hdf, task)
    if control:
        np.random.shuffle(targets)
    targangles = np.arctan2(targets[:,1],targets[:,0])
    moddepths = np.zeros([len(plx.units)])
    print "fitting tuning curves"
    for i in range(len(plx.units)):
        if any(np.isnan(rates[:,i])):
            moddepths[i]=np.nan
        else:
            unit = plx.units[i]
            fitfunc = lambda p, x: p[0]*np.cos(x+p[1]) + p[2]*x # Target function
            errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
            p0 = [1., 0., 0] # Initial guess for the parameters
            p1, success = scipy.optimize.leastsq(errfunc, p0[:], args=(targangles, rates[:,i]))
            xvals = np.linspace(targangles.min(), targangles.max(), 100)
            tuningcurve = fitfunc(p1,xvals)
            curvemax = np.max(tuningcurve)
            curvemean = np.mean(tuningcurve)
            moddepths[i] = np.abs(curvemax - curvemean)
    return moddepths

def categorize_units(plx,decoder):
    bmiunits = [(unit[0], unit[1]) for unit in decoder.units]
    bmichans = [unit[0] for unit in decoder.units]
    directinds = [i for i,unit in enumerate(plx.units) if unit in bmiunits]

    tempinds = [i for i,unit in enumerate(plx.units) if unit not in bmiunits]
    remainder = [plx.units[i] for i in tempinds]

    tempinds = [i for i,unit in enumerate(remainder) if unit[0] in bmichans]
    tempunits = [remainder[i] for i in tempinds]
    samechannelinds = [i for i,unit in enumerate(plx.units) if unit in tempunits]

    tempinds = [i for i,unit in enumerate(remainder) if unit[0] not in bmichans]
    remainder = [remainder[i] for i in tempinds]

    tempinds = [i for i,unit in enumerate(remainder) if unit[0] < 129]
    tempunits = [remainder[i] for i in tempinds]
    samehemiinds = [i for i,unit in enumerate(plx.units) if unit in tempunits]

    tempinds = [i for i,unit in enumerate(remainder) if unit[0] >= 129]
    remainder = [remainder[i] for i in tempinds]

    opphemiinds = [i for i,unit in enumerate(plx.units) if unit in remainder]

    return directinds, samechannelinds, samehemiinds, opphemiinds

def get_movement_periods(datafile, task):

    if type(datafile)==plexfile.Datafile:
        messages = parse.messages(datafile.events[:].data)
        key = 'state'
    else:
        messages = datafile.root.task_msgs[:]
        key = 'msg'

    reward_inds = np.array([i for i,m in enumerate(messages) if m[key]=='reward'])
    penalty_inds = np.array([i for i,m in enumerate(messages) if m[key]=='hold_penalty'])
    if task == 'bmi_control' or task == 'manual_control':
        origin_hold_inds = reward_inds - 3
        movement_inds = reward_inds - 2
        terminus_hold_inds = reward_inds - 1
    elif task == 'manual_control_2':
        origin_hold_inds = reward_inds - 5
        movement_inds = reward_inds - 4
        terminus_hold_inds = reward_inds - 3
    else:
        raise TypeError('Unknown task!')
    
    origin_hold_periods = np.array([(messages[i]['time'], messages[i+1]['time']) for i in origin_hold_inds])
    movement_periods = np.array([(messages[i]['time'], messages[i+1]['time']) for i in movement_inds])
    terminus_hold_periods = np.array([(messages[i]['time'], messages[i+1]['time']) for i in terminus_hold_inds])
    penalty_periods = np.array([(messages[i]['time'], messages[i+1]['time']) for i in penalty_inds])

    return origin_hold_periods, movement_periods, terminus_hold_periods, penalty_periods



def calc_movement_endpoints(hdf, origin_hold_periods, terminus_hold_periods):
    '''

    Returns endpoints of point to point movements of hand during reaching task.
    Start and end position are calculated as the mean position during the origin
    and terminus hold periods, respectively.
    
    Parameters
    ----------
    hdf : hdf5 file
        The file containing the data.

    origin_hold_periods : array
        N x 2 array of start and end frame of origin hold periods in hdf file.

    terminus_hold_periods : array
        N x 2 array of start and end frame of terminus hold periods in hdf file.

    Returns
    -------
    endpoints : array
        N x 3 x 2 array of x, y, and z coordinates of start and end points of
        movement in mm.

    '''

    vectors = np.zeros([len(origin_hold_periods), 3, 2])
    for i, (origin,terminus) in enumerate(zip(origin_hold_periods,
                                                terminus_hold_periods)):
        odata = basicanalysis.interp_missing(hdf.root.motiontracker\
                                    [origin[0]:origin[1],0,:][:,None,:])
        tdata = basicanalysis.interp_missing(hdf.root.motiontracker\
                                    [terminus[0]:terminus[1],0,:][:,None,:])
        vectors[i,:,0] = np.mean(odata.squeeze()[np.nonzero(~np.isnan(\
            odata.squeeze()[:,0]))[0]],0)
        vectors[i,:,1] = np.mean(tdata.squeeze()\
            [np.nonzero(~np.isnan(tdata.squeeze()[:,0]))[0]], 0)
    return vectors

def mt_to_plx(period_list, ts):
    if isinstance(period_list, tuple):
        return (ts[period_list[0]], ts[period_list[1]])
    plx_periods = []
    for per in period_list:
        plx_periods.append((ts[per[0]], ts[per[1]]))
    return plx_periods

def get_response(plx, chan, unit, period):
    spks = plx.spikes[period[0]:period[1]].data
    return spks[np.logical_and(spks['chan'] == chan, spks['unit'] == unit)]['ts']

def get_mean_response_all_trials(plx, chan, unit, periods):
    responses = np.zeros([len(periods)])
    for i, per in enumerate(periods):
        num_spikes = len(get_response(plx, chan, unit, per))
        time = per[1] - per[0]
        responses[i] = num_spikes/time
    return responses

def get_response_all_trials(plx, chan, unit, periods):
    responses = []
    for per in periods:
        responses.append(get_response(plx, chan, unit, per))
    return responses

def calc_vector_directions(vectors):
    normed = vectors/np.sqrt(np.sum(vectors**2,axis=1))[:,None]
    if vectors.shape[1]==2:
        theta = np.arctan2(normed[:,1],normed[:,0])
        return theta
    else:
        theta = np.arctan2(normed[:,2],normed[:,0])
        phi = np.arccos(normed[:,1])
        return np.append(theta[:,None], phi[:,None],1)

def get_mean_responses_all_units(plx, units, periods):
    responses = np.zeros([len(units), len(periods)])
    for i, unit in enumerate(units):
        print i, "/", len(units)
        responses[i,:] = get_mean_response_all_trials(plx, unit[0], unit[1], periods)
    return responses

def get_responses_all_units(plx, units, periods):
    responses = []
    for i, unit in enumerate(units):
        print i, "/", len(units)
        responses.append(get_response_all_trials(plx, unit[0], unit[1], periods))
    return responses

def calc_baseline_all_units(plx, units, periods):
    responses = get_mean_responses_all_units(plx, units, periods)
    means = np.mean(responses,1)
    stds = np.std(responses, 1)
    return means, stds

def zscore_data(data, means, stds):
    return (data - means[:,None])/stds[:,None]

def get_baseline_periods(penalties, ts):
    pp = np.array(mt_to_plx(penalties, ts))
    pp2 = pp[:,1]
    pp1 = pp2 - 1
    pp1 = list(pp1)
    pp2 = list(pp2)
    return zip(pp1, pp2)

def make_hist_direction_tuning(directions, responses, numbins):
    binsize = (2*np.pi)/numbins
    edges = np.linspace(-np.pi, np.pi, numbins+1)
    xvec = edges[0:-1] + (binsize/2)
    vals = np.zeros([numbins])
    sem = np.zeros([numbins])
    for bin in range(numbins):
        vals[bin] = np.mean(responses[np.nonzero(np.logical_and(directions>=edges[bin], directions<edges[bin+1]))[0]])
        sem[bin] = np.std(responses[np.nonzero(np.logical_and(directions>=edges[bin], directions<edges[bin+1]))[0]])/np.sqrt(len(responses[np.nonzero(np.logical_and(directions>=edges[bin], directions<edges[bin+1]))[0]]))
    return np.append(np.append(xvec[:,None], vals[:,None], 1),sem[:,None],1)

def make_hist_movement_activation_single_trial(responsetimes, period, numbins):
    shifted = responsetimes - period[0]
    edges = np.linspace(0,period[1]-period[0],numbins+1)
    vals = np.zeros([numbins])
    binsize = (period[1]-period[0])/numbins
    for bin in range(numbins):
        vals[bin] = sum(np.logical_and(shifted>=edges[bin], shifted<edges[bin+1]))
    return vals

def make_hist_movement_activation_all_trials(responses, periods, numbins):
    time = periods[0][1]-periods[0][0]
    binsize = time/numbins
    edges = np.linspace(0,time,numbins+1)
    xvec = edges[0:-1] + (binsize/2)
    vals = np.zeros([len(responses), numbins])
    for i, (trial, per) in enumerate(zip(responses, periods)):
        vals[i,:] = make_hist_movement_activation_single_trial(trial, per, numbins)
    vals = vals.sum(0)
    return np.append(xvec[:,None], vals[:,None],1)

def new_periods(periods):
    output = []
    for p in periods:
        output.append((p[0], p[0]+2))
    return output

def zscore_list(data, means, stds):
    output = []
    for i, unit in enumerate(data):
        vals = []
        for trial in unit:
            vals.append((trial - means[i])/stds[i])
        output.append(vals)
    return output

if __name__ == "__main__":
    #write lines of code in here
    #to run, inside ipython, type "run filename.py"
    plx, hdf, ts = rfanalysis.open_files('/home/helene/Downloads/cart20120731_01.plx', '/home/helene/Downloads/cart20120731_01.hdf')
    or_periods, move_periods, ter_periods, pen_periods = get_movement_periods(hdf)
    directions = calc_vector_directions(calc_movement_vectors(hdf, or_periods, ter_periods))
    units, udict = rfanalysis.list_all_units(plx)
    means, stds = calc_baseline_all_units(plx, units, get_baseline_periods(pen_periods, ts))
    responses = get_mean_responses_all_units(plx, units, mt_to_plx(move_periods, ts))
    normed_responses = zscore_data(responses, means, stds)
    newp = new_periods(mt_to_plx(or_periods,ts))
    time_responses = get_responses_all_units(plx, units, newp)