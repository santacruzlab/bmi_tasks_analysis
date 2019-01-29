'''
Calculate metrics on a whole data block
'''
import numpy as np
import datetime


def target_reaching_angular_error(te):
    from scipy.stats import nanmean
    err = te.angular_error
    reaching = te.get_state_inds('target') * te.label_trying(ds_factor=1)
    reaching_err = err[reaching]
    reaching_err = reaching_err[~np.isnan(reaching_err)]
    return reaching_err

def hold_err_counts(te):
    task_msgs = te.hdf.root.task_msgs[:]
    n_hold_errors = len(np.nonzero(task_msgs['msg'] == 'hold_penalty')[0])
    n_success = len(np.nonzero(task_msgs['msg'] == 'reward')[0])
    return n_hold_errors, n_success

def reward_counts(te):
    task_msgs = te.hdf.root.task_msgs[:]
    n_success = len(np.nonzero(task_msgs['msg'] == 'reward')[0])
    return n_success

def starting_postures(te, full=False):
    '''
    Get all the starting arm postures for the tentacle arm during a tentacle task. Can only be used
    for tasks with a 'premove' state, e.g., bmi_joint_perturb
    '''
    task_msgs = te.hdf.root.task_msgs[:]
    premove_times = task_msgs[task_msgs['msg'] == 'premove']['time']
    starting_configs = []
    for premove_time in premove_times:
        starting_config = te.hdf.root.task[premove_time]['decoder_state'].ravel()
        n_iter = 7
        k = 0
        while not np.all(starting_config[4:8] == 0) and k < n_iter:
            starting_config = te.hdf.root.task[premove_time]['decoder_state'].ravel()        
            premove_time += 1
            k += 1
            
        starting_config = tuple(starting_config[:4])

        # Make sure that all the angles are in the range [-pi, pi]
        starting_config = tuple([np.arctan2(np.sin(angle), np.cos(angle)) for angle in starting_config])
        starting_configs.append(starting_config)

    if full:
        return np.vstack(starting_configs)
    else:
        starting_configs = np.unique(starting_configs)
        return starting_configs

def clda_time(te, max_depth=np.inf):
    '''
    Determine how much time was spent on CLDA for a given decoder on a particular day
    '''
    decoder_record = te.decoder_record
    from db.tracker import models 
    depth = 0
    total_time = 0
    while depth < max_depth:
        rec = models.TaskEntry.objects.using(te.record._state.db).get(id=decoder_record.entry_id)
        t = datetime.datetime.strptime(rec.offline_report()['Runtime'], r'%H:%M:%S') - datetime.datetime.strptime('', '')
        total_time += t.seconds
        try:
            decoder_record = dbfn.TaskEntry(rec, dbname=te.record._state.db).decoder_record
        except:
            break
        depth += 1
    return total_time

