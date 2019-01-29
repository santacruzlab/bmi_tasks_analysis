import db.dbfunctions as dbf
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

from riglib.bmi import sskfdecoder
from db.tracker import models

def get_decoder_bias(decoder_name):
    if isinstance(decoder_name, int):
        decoder_name = dbf.TaskEntry(decoder_name, db_name=dbf.db_name).decoder_record.name
    decoder = models.Decoder.objects.using(dbf.db_name).get(name=decoder_name)
    decoder = decoder.load(db_name = dbf.db_name)
    if 'sskf' in decoder_name:
    #if isinstance(decoder, sskfdecoder.SSKFDecoder): #'sskf' in decoder_name:
        unbiased_name = decoder_name[:-6]
        unbiased_decoder = models.Decoder.objects.get(name=unbiased_name).load()
        bias_vec = decoder.filt.F[3:6,-1] - unbiased_decoder.filt.F[3:6,-1]
    else:
        bias_vec = np.asarray(decoder.filt.A)[3:6,-1]
    gain = np.linalg.norm(bias_vec)
    angle = np.arctan2(bias_vec[-1], bias_vec[0])
    if angle < 0: angle += 2*np.pi
    return dict(gain=gain, angle=angle*180/np.pi)

def trial_array(sess_list,save=None):
    ''' Works for center out 8 target task only. sess_list input must be a list of 1 or more session IDs. If more than 1 they will
    be concatenated as if they were a single continuous session.
    
    Each row of the output array represents a single initiated trial (trials count as initiated when a center hold is completed).
    The first 3 columns indicate whether that trial ended in a reward, timeout error, or a hold error, and the timestamp in
    sec of the beginning of the trial (counted as the time at the end of the center hold state). The 4th column indicates
    which target was presented (0-8) and the 4th and 5th columns list the reach time and movement error for that trial
    (these are nans for timeout trials since the cursor never made it to the target).'''
    
    var_dict = {'completed':0,'timed_out':1,'hold_error':2,'target_number':3,'reach_time':4,'movement_error':5}
    angle_tolerance = .01
    angle_list = np.arange(-3*np.pi/4,5*np.pi/4,np.pi/4.) # 8 targets from -135 to 180 degrees
    
    all_states = []
    all_targstate_inds = []
    all_target_numbers = []
    all_trajectories = []
    all_biases = []
    
    for j, sess_id in enumerate(sess_list):
        hdf = dbf.get_hdf(sess_id) 
        states = hdf.root.task_msgs[:] 
        
        # pull out all target states and corresponding target index values, then remove center targets
        target_states = [(i,s) for i,s in enumerate(states[:-2]) if (s[0]=='target')]
        target_inds = [hdf.root.task[s[1][1]]['target_index'][0] for s in target_states]
        target_states = [target_states[i] for i, ti in enumerate(target_inds) if ti==1]
        targstate_inds = [s[0] for s in target_states]
        
        # get location of target for each trial
        target_locs = [hdf.root.task[s[1][1]]['target'] for s in target_states]
        target_numbers = []        
        # Find the angle of the target for each trial and classify it with the correct target index key
        for t in target_locs:
            targ_angle = np.arctan2(t[2],t[0])
            target_num = None
            for i, ang in enumerate(angle_list):
                if targ_angle>=ang-angle_tolerance and targ_angle<=ang+angle_tolerance:
                    target_num = i
            assert (target_num is not None), "Unrecognized target angle! "
            target_numbers.append(target_num)
            
        # Get cursor trajectory for each trial
        trajectories = [hdf.root.task[states[s[0]][1]:states[s[0]+1][1]]['cursor'] for s in target_states]

        
        # Adjust timestamps and indices if concatenating multiple sessions
        if j>0:
            last_time = all_states[-1][1]
            last_ind = len(all_states)
            states = [(s[0], s[1]+last_time) for s in states]
            targstate_inds = [k+last_ind for k in targstate_inds]
        
        # Combine data from this file with previous ones
        all_states.extend(states)
        all_targstate_inds.extend(targstate_inds)
        all_target_numbers.extend(target_numbers)
        all_trajectories.extend(trajectories) 

        try:
            biases = [hdf.root.task[s[1][1]]['bias'] for s in target_states]
            all_biases.extend(biases)
        except:
            pass

    
    # initialize array to hold trial data
    output_array = np.zeros([len(all_targstate_inds),6])
    output_array[:,var_dict['reach_time']] = np.nan
    output_array[:,var_dict['movement_error']] = np.nan
    
    for i, state_ind in enumerate(all_targstate_inds):
        state_ts = all_states[state_ind][1]
        # if trial ended in timeout
        if all_states[state_ind+1][0]=='timeout_penalty':
            output_array[i,var_dict['timed_out']] = state_ts
            output_array[i,var_dict['target_number']] = all_target_numbers[i]
        # if trial ended in hold error
        if all_states[state_ind+2][0]=='hold_penalty':
            output_array[i,var_dict['hold_error']] = state_ts
            output_array[i,var_dict['reach_time']] = (all_states[state_ind+1][1] - state_ts)/60.
            output_array[i,var_dict['target_number']] = all_target_numbers[i]
            # Find the movement error (max perpendicular distance from straight line path to target)
            points = all_trajectories[i][:,[0,2]]
            dists = np.zeros(len(points))
            ang = angle_list[all_target_numbers[i]]
            m = np.tan(ang)
            if np.abs(m)>1000000:
                dists = points[:,0]
            else:
                dists = np.abs(points[:,1]-m*points[:,0])/np.sqrt(m**2 + 1)
            output_array[i,var_dict['movement_error']] = np.max(np.abs(dists))
        # if trial ended in reward
        if all_states[state_ind+3][0]=='reward':
            output_array[i,var_dict['completed']] = state_ts
            output_array[i,var_dict['reach_time']] = (all_states[state_ind+1][1] - state_ts)/60.
            output_array[i,var_dict['target_number']] = all_target_numbers[i]
            # Find the movement error (max perpendicular distance from straight line path to target)
            points = all_trajectories[i][:,[0,2]]
            dists = np.zeros(len(points))
            ang = angle_list[all_target_numbers[i]]
            m = np.tan(ang)
            if np.abs(m)>1000000:
                dists = points[:,0]
            else:
                dists = np.abs(points[:,1]-m*points[:,0])/np.sqrt(m**2 + 1)
            output_array[i,var_dict['movement_error']] = np.max(np.abs(dists))

    if save is not None:
        np.save(save, output_array)
                       
    return output_array, var_dict, all_biases

def extract_reach_times(tarray,n=None,pos=0,target=None,include_incompletes=False):
    
    # if no target specified, average across all targets
    if target is not None:
        tarray = tarray[tarray[:,3]==target,:]
        
    # if include_incompletes is True, includes trials ending in hold errors, otherwise
    # just use trials ending in reward
    if include_incompletes:
        tarray = tarray[np.logical_or(tarray[:,0]>0,tarray[:,2]>0),:]
    else:
        tarray = tarray[tarray[:,0]>0,:]
        
    # if no n is specified, average across all trials. if n is specified, pos
    # determines whether we use the first n, middle n, or last n trials. NOTE: n
    # specifies the number of trials for the target specified in target input, unless
    # target = None
    if n is not None:
        assert n<=len(tarray), "Not enough trials!"
        if pos==0:
            tarray = tarray[:n]
        elif pos==1:
            tarray = tarray[n-int(n/2):n-int(n/2)+n]
        elif pos==2:
            tarray = tarray[-n:]
        else:
            raise Exception('pos input must be 0, 1, or 2')
    
    # Returns reach SPEED, not time        
    # return 8.0/tarray[:,4]
    return tarray[:,4]

def extract_movement_errors(tarray,n=None,pos=0,target=None,include_incompletes=False):
    
    # if no target specified, average across all targets
    if target is not None:
        tarray = tarray[tarray[:,3]==target,:]
        
    # if include_incompletes is True, includes trials ending in hold errors, otherwise
    # just use trials ending in reward
    if include_incompletes:
        tarray = tarray[np.logical_or(tarray[:,0]>0,tarray[:,2]>0),:]
    else:
        tarray = tarray[tarray[:,0]>0,:]
        
    # if no n is specified, average across all trials. if n is specified, pos
    # determines whether we use the first n, middle n, or last n trials. NOTE: n
    # specifies the number of trials for the target specified in target input, unless
    # target = None
    if n is not None:
        assert n<=len(tarray), "Not enough trials!"
        if pos==0:
            tarray = tarray[:n]
        elif pos==1:
            tarray = tarray[n-int(n/2):n-int(n/2)+n]
        elif pos==2:
            tarray = tarray[-n:]
        else:
            raise Exception('pos input must be 0,1, or 2')
            
    return tarray[:,5]

def calc_hold_error_rate(tarray,n=None,pos=0,target=None):
    # if no target specified, average across all targets
    if target is not None:
        tarray = tarray[tarray[:,3]==target,:]
        
    # remove timeout trials
    tarray = tarray[np.logical_or(tarray[:,0]>0,tarray[:,2]>0),:]
        
    # if no n is specified, average across all trials. if n is specified, pos
    # determines whether we use the first n, middle n, or last n trials. NOTE: n
    # specifies the number of trials for the target specified in target input, unless
    # target = None
    if n is not None:
        assert n<=len(tarray), "Not enough trials!"
        if pos==0:
            tarray = tarray[:n]
        elif pos==1:
            tarray = tarray[n-int(n/2):n-int(n/2)+n]
        elif pos==2:
            tarray = tarray[-n:]
        else:
            raise Exception('pos input must be 0,1, or 2')
            
    return float(np.sum(tarray[:,2]>0))/(np.sum(tarray[:,0]>0)+np.sum(tarray[:,2]>0))

def calc_time_per_reward(tarray,n=None,pos=0,target=None):
    # This measures the mean time spent moving the cursor per reward achieved. It
    # excludes trials that ended in timeout since timeouts are almost always
    # due to the subject taking a break as opposed to control being too bad to get to the target.
    
    # if no target specified, average across all targets
    if target is not None:
        tarray = tarray[tarray[:,3]==target,:]
        
    # remove timeout trials
    tarray = tarray[np.logical_or(tarray[:,0]>0,tarray[:,2]>0),:]
        
    # if no n is specified, average across all trials. if n is specified, pos
    # determines whether we use the first n, middle n, or last n trials. NOTE: n
    # specifies the number of trials for the target specified in target input, unless
    # target = None
    if n is not None:
        assert n<=len(tarray), "Not enough trials!"
        if pos==0:
            tarray = tarray[:n]
        elif pos==1:
            tarray = tarray[n-int(n/2):n-int(n/2)+n]
        elif pos==2:
            tarray = tarray[-n:]
        else:
            raise Exception('pos input must be 0,1, or 2')
    
    # total sum of all reach times divided by number of rewarded triasl        
    return float(np.sum(tarray[:,4]))/np.sum(tarray[:,0]>0)

def target_means(tarray, emeasure='RT'):
    ''' Returns the mean and SE for each individual target'''

    err_means = np.zeros(8)
    err_se = np.zeros(8)

    for t in range(8):
        if emeasure=='RT':
            temp = extract_reach_times(tarray,target=t)
        elif emeasure=='merr':
            temp = extract_movement_errors(tarray,target=t)
        else:
            raise Exception('Unrecognized error measure!')
        err_means[t] = np.mean(temp)
        err_se[t] = np.std(temp)/np.sqrt(len(temp))

    return err_means, err_se

def plot_bias_polar(tarray_list, **kwargs):
    if kwargs.get('partial') is None:
        partial = [0]*len(tarray_list)
    else:
        partial = kwargs.get('partial')
    angs = list(np.arange(-3*np.pi/4,5*np.pi/4,np.pi/4.))
    angs = angs+[angs[0]]
    all_mns = []
    all_ses = []
    max_mean = 0
    for i, tarray in enumerate(tarray_list):
        if partial[i]==-1:
            tarray = tarray[np.nonzero(tarray[:,0]>0)[0][:160]]
        if partial[i]==1:
            tarray = tarray[np.nonzero(tarray[:,0]>0)[0][-160:]]

        # if inds is not None:
        #     if range>0:
        #         tarray = tarray[:inds]
        #     else:
        #         tarray = tarray[inds:]

        mns, ses = target_means(tarray,emeasure=kwargs.get('emeasure'))
        if np.max(mns)> max_mean: max_mean = np.max(mns)
        mns = list(mns)+[mns[0]]
        ses = list(ses) + [ses[0]]
        all_mns.append(mns)
        all_ses.append(ses)

    #cols = ['r','b','g','p','o','y']
    if not kwargs.get('hide_plots'):
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
        if kwargs.get('cols') is None:
            cols = ['k']*len(tarray_list)
        else:
            cols = kwargs.get('cols')
        for i in range(len(all_mns)):
        #     if partial[i] == -1:
        #         col = 'r'
        #     elif partial[i]==1:
        #         col = 'b'
        #     else:
        #         col = 'grey'
            ax.errorbar(angs, all_mns[i], yerr=all_ses[i],color=cols[i],linewidth=3, capsize=0,elinewidth=1)
            # if i==0:
            #     ax.errorbar(angs, all_mns[i], yerr=all_ses[i],color='grey',linewidth=3, capsize=0,elinewidth=1)
            # elif i==1:
            #     ax.errorbar(angs, all_mns[i], yerr=all_ses[i],color='r',linewidth=3, capsize=0,elinewidth=1)
            # else:
            #     ax.errorbar(angs, all_mns[i], yerr=all_ses[i],linewidth=3, capsize=0,elinewidth=1)
        ax.set_rmax(max_mean*1.3)
        ax.grid(True)
        if kwargs.get('labels') is not None:
            ax.legend(kwargs.get('labels'))

    return angs, all_mns, all_ses


def plot_session_bias(sess_list,**kwargs):
    all_data = [trial_array(s)[0] for s in sess_list]
    angs, all_mns, all_ses = plot_bias_polar(all_data,**kwargs)
    return angs, all_mns, all_ses

def plot_early_vs_late(bias_id, baseline_id=None,n=20,emeasure='RT'):
    labels = ['early', 'late']
    tarray = trial_array(bias_id)[0]
    early_inds = np.nonzero(tarray[:,0]>0)[0][:8*n]
    late_inds = np.nonzero(tarray[:,0]>0)[0][-8*n:]
    data = [tarray[early_inds,:],tarray[late_inds,:]]
    if baseline_id is not None:
        barray = trial_array(baseline_id)[0]
        data.append(barray)
        labels.append('baseline')
    plot_bias_polar(data, labels=labels)

def plot_bias(baseline_id,bias_id,n=20,savefile=None):
    
    # Extract relevant RT means, SD, and perform T tests
    
    baseline_data = trial_array(baseline_id)[0]
    baseline_means = np.zeros(8)
    baseline_se = np.zeros(8)
    pvals = np.zeros([8,2])
    bias_data = trial_array(bias_id)[0]
    bias_early_means = np.zeros(8)
    bias_late_means = np.zeros(8)
    bias_early_se = np.zeros(8)
    bias_late_se = np.zeros(8)
    
    for t in range(8):
        temp = extract_reach_times(baseline_data,target=t)
        baseline_means[t] = np.mean(temp)
        baseline_se[t] = np.std(temp)/np.sqrt(len(temp))
        
        temp_early = extract_reach_times(bias_data,n=n,pos=0,target=t)
        bias_early_means[t] = np.mean(temp_early)
        bias_early_se[t] = np.std(temp_early)/np.sqrt(len(temp_early))
        
        temp_late = extract_reach_times(bias_data,n=n,pos=2,target=t)
        bias_late_means[t] = np.mean(temp_late)
        bias_late_se[t] = np.std(temp_late)/np.sqrt(len(temp_late))
        
        pvals[t,0] = np.round(stats.ttest_ind(temp, temp_early, equal_var=False)[1],decimals=2)
        pvals[t,1] = np.round(stats.ttest_ind(temp_early, temp_late, equal_var=False)[1],decimals=2)
    #####################################
    

    # Plot the data
    
    max_data_val = np.max(np.concatenate((baseline_means,bias_early_means,bias_late_means)))
    ylims = (0,max_data_val*1.2)
    colors = [(183/255.,192/255.,232/255.), (159/255.,194/255.,143/255.), (196/255.,173/255.,120/255.)]
    
    # This maps the order that the targets are indexed to the subplot positions
    order = [(2,0),(2,1),(2,2),(1,2),(0,2),(0,1),(0,0),(1,0)]
    fig, axs = plt.subplots(3,3)
    
    for t in range(8):
        axhandle=axs[order[t][0],order[t][1]]
        data_mean = [baseline_means[t], bias_early_means[t], bias_late_means[t]]
        data_err = [baseline_se[t], bias_early_se[t], bias_late_se[t]]
        b1 = axhandle.bar([1],data_mean[0],width=1,yerr=data_err[0],color=colors[0],ecolor='k')
        b2 = axhandle.bar([2],data_mean[1],width=1,yerr=data_err[1],color=colors[1],ecolor='k')
        b3 = axhandle.bar([3],data_mean[2],width=1,yerr=data_err[2],color=colors[2],ecolor='k')
        #axhandle.get_xaxis().set_visible(False)
        axhandle.set_ylim(ylims)
        axhandle.set_yticks([0,np.round(ylims[1]/2.,decimals=1),np.round(ylims[1],decimals=1)])
        axhandle.set_xticks([2,3])
        axhandle.tick_params(axis='y', which='major', labelsize=6)
        
        if pvals[t,0]<.05:
            xticklabs = ['*']
        else:
            xticklabs = [' ']  
        if pvals[t,1]<.05:
            xticklabs.append('*')
        else:
            xticklabs.append(' ')
            
        axhandle.set_xticklabels(xticklabs)
        for t in axhandle.xaxis.get_major_ticks(): 
            t.tick1On = False 
            t.tick2On = False
        axhandle.tick_params(axis='x', which='major', labelsize=8)
            
        
    axs[1,1].axis('off')
    plt.figlegend((b1,b2,b3),('Baseline','Early bias', 'Late bias'),'center')
    plt.suptitle('Reach time (seconds)', fontsize=14)
    
    
    # Save the plot if desired
    if savefile is not None:
        try:
            savefig('/storage/plots/'+str(bias_id[0])+'_'+savefile, dpi=300)
        except:
            print "/storage/plots folder doesn't exist!"
