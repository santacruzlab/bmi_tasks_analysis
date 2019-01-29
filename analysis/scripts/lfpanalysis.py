from basicanalysis import *
from offlineprediction import *
from rfanalysis import *
from plexon import plexfile
from riglib.nidaq import parse
import numpy as np
import cPickle
import sys

plx_path = '/storage/bmi3d/plexon/'
hdf_path = '/storage/bmi3d/rawdata/hdf/'
seq_path = '/storage/bmi3d/sequences/'

#plx_path = '/media/A805-7C6F/split data/'
#hdf_path = '/media/A805-7C6F/split data/'
#seq_path = '/media/A805-7C6F/split data/'


# split vis-tact stims
# file_list = ['cart20130126_01',
# 			'cart20130128_01',
# 			'cart20130129_01',
# 			'cart20130130_01']
# order_list = [0, 1, 1, 0]

#combined vis-tact stims
file_list = ['cart20130123_02','cart20130124_01','cart20130125_01']
order_list = [2,2,2]


for count, session in enumerate(file_list[0]): ##### GET RID OF 0

	#load files for session
	print count, "loading file"
	sys.stdout.flush()
	plx = plexfile.openFile(plx_path + session + '.plx')
	hdf = tables.openFile(hdf_path + session + '.hdf')
	seq = cPickle.load(open(seq_path + session[:-3] + '.pkl'))

	# Find times of stimuli
	print count, "finding stim times"
	sys.stdout.flush()
	stimtimes = tap_times(plx, min_duration = 60)
	stimstart = stimtimes[0][0]

	# Get list of numdisplay state begin times
	#(exclude anything past 1 hour for damaged data files)
	# and match each to the corresponding stimulus occurrence
	states = parse.messages(plx.events[:3590].data)
	disptimes = [s[0] for s in states if s[1] == 'numdisplay']
	output, seq2 = match_taps(seq[:len(disptimes)],disptimes,stimtimes)

	# get rid of zone 22
	output = list(np.array(output)[np.array(seq2) != 22])
	seq2 = list(np.array(seq2)[np.array(seq2) != 22])

	# control- use random times
	#for i in range(len(output)):
	#	output[i] = (5+np.random.rand(1)[0]*(plx.length-10), 5+np.random.rand(1)[0]*(plx.length-10))


	# Generate PSTH for all PMv units 1 s before and after stims
	if count==0:
		unitlist = range(128,256)
	print count, "getting LFP data"
	sys.stdout.flush()
	lfp_data = lfp_epochs(plx, units, generate_epochs(np.array([val[0] for val in output]),1.0,1.0))

	# subtract the mean for each channel
	lfp_data = swapaxes(swapaxes(lfp_data,1,2)/np.mean(np.mean(lfp_data,axis=0),axis=1),1,2)
	# Separate visual from tactile trials and sort them by zone number.
	if order_list[count]==0:
		seq_t = seq2[:296]
		seq_v = seq2[296:]
		output_t = output[:296]
		output_v = output[296:]
	elif order_list[count]==1:
		seq_v = seq2[:296]
		seq_t = seq2[296:]
		output_v = output[:296]
		output_t = output[296:]
	else:
		seq_t = seq2[:]
		seq_v = seq2[:]
		output_t = output[:]
		output_v = output[:]

	grouped_t,order_t = sort_stimuli(seq_t)
	grouped_v,order_v = sort_stimuli(seq_v)

	# Separate PTSH for visual vs. tactile trials
	print count, "separating vis and tact stims"
	sys.stdout.flush()
	psth_t = np.zeros([lfp_data.shape[0],lfp_data.shape[1],len(order_t)*grouped_t.shape[1]])
	for i, zone in enumerate(order_t):
	    psth_t[:,:,i*grouped_t.shape[1]:(i+1)*grouped_t.shape[1]] = lfp_data[:,:,grouped_t[i,:].astype(int)]
	psth_v = np.zeros([lfp_data.shape[0],lfp_data.shape[1],len(order_v)*grouped_v.shape[1]])
	for i, zone in enumerate(order_t):
	    psth_v[:,:,i*grouped_v.shape[1]:(i+1)*grouped_v.shape[1]] = lfp_data[:,:,grouped_v[i,:].astype(int)]


	# Create master list of results for each zone that other days will be added to
	print count, "saving data"
	sys.stdout.flush()
	if count==0:
		master_t = [psth_t[:,:,i*grouped_t.shape[1]:(i+1)*grouped_t.shape[1]] for i in range(len(order_t))]
		master_v = [psth_v[:,:,i*grouped_v.shape[1]:(i+1)*grouped_v.shape[1]] for i in range(len(order_v))]
		ord_t = order_t[:]
		ord_v = order_v[:]
	else:
		master_t = [np.concatenate((master_t[i],fr_t[:,:,i*grouped_t.shape[1]:(i+1)*grouped_t.shape[1]]),axis=2) for i in range(len(ord_t))]
		master_v = [np.concatenate((master_v[i],fr_v[:,:,i*grouped_v.shape[1]:(i+1)*grouped_v.shape[1]]),axis=2) for i in range(len(ord_v))]





