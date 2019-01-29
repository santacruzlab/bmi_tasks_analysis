from basicanalysis import *
from offlineprediction import *
from rfanalysis import *
from plexon import plexfile
from riglib.nidaq import parse
import numpy as np
import cPickle
import sys

#plx_path = '/storage/bmi3d/plexon/'
#hdf_path = '/storage/bmi3d/rawdata/hdf/'
#seq_path = '/storage/bmi3d/sequences/'

plx_path = '/media/A805-7C6F/split data/'
hdf_path = '/media/A805-7C6F/split data/'
seq_path = '/media/A805-7C6F/split data/'


# split vis-tact stims
file_list = ['cart20130126_01',
			'cart20130128_01',
			'cart20130129_01',
			'cart20130130_01']
order_list = [0, 1, 1, 0]

#combined vis-tact stims
# file_list = ['cart20130123_02','cart20130124_01','cart20130125_01']
# order_list = [2,2,2]


for count, session in enumerate(file_list):

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
		udict = create_dict(plx)
		unitlist = plx.units[udict[(133,1)]:]
		#unitlist = plx.units[:udict[(129,1)]]
	print count, "getting spike counts"
	sys.stdout.flush()
	spike_counts = 10.0*psth_analysis(plx,unitlist, generate_epochs(np.array([val[0] for val in output]),1.0,1.0),20)

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


######################### FOR REGRESSION ANALYSIS ####################
# 	# Get time periods for visual and tactile parts of experiment
# 	if grouped_t.shape[1]>3:
# 		tact_train_start = output_t[int(np.min(grouped_t[:,0]))][0] - 1.0
# 		tact_train_end = output_t[int(np.max(grouped_t[:,int(np.floor(grouped_t.shape[1]/4)*-1 - 1)]))][1] + 1.0
# 		tact_val_start = tact_train_end.copy()
# 		tact_val_end = output_t[int(np.max(grouped_t[:,-1]))][1] + 1.0
# 	else:
# 		tact_train_start = output_t[int(np.min(grouped_t[:,0]))][0] - 1.0
# 		tact_train_end = output_t[int(np.max(grouped_t[:,-2]))][1] + 1.0
# 		tact_val_start = tact_train_end.copy()
# 		tact_val_end = output_t[int(np.max(grouped_t[:,-1]))][1] + 1.0
# 	if grouped_v.shape[1]>3:
# 		vis_train_start = output_v[int(np.min(grouped_v[:,0]))][0] - 1.0
# 		vis_train_end = output_v[int(np.max(grouped_v[:,int(np.floor(grouped_v.shape[1]/4)*-1 - 1)]))][1] + 1.0
# 		vis_val_start = vis_train_end.copy()
# 		vis_val_end = output_v[int(np.max(grouped_v[:,-1]))][1] + 1.0
# 	else:
# 		vis_train_start = output_v[int(np.min(grouped_v[:,0]))][0] - 1.0
# 		vis_train_end = output_v[int(np.max(grouped_v[:,-2]))][1] + 1.0
# 		vis_val_start = vis_train_end.copy()
# 		vis_val_end = output_v[int(np.max(grouped_v[:,-1]))][1] + 1.0

# 	nunits = len(unitlist)
# 	unitinds = np.array([])
# 	nstims_t = len(order_t)
# 	nstims_v = len(order_v)
# 	ndelays = 20
# 	binsize = .1
	
# 	#Bin spikes
# 	nbins = np.floor((tact_train_end-tact_train_start)/binsize)
# 	tact_train_bins = np.linspace(tact_train_start+binsize,tact_train_start + (binsize*nbins),nbins)
# 	binned_temp=np.array([bin for bin in plx.spikes.bin(tact_train_bins, binlen=binsize)])
# 	tact_train_spikes = binned_temp[:,np.array([udict[unit] for unit in unitlist])]

# 	################ Create design matrix
# 	Atemp_t = np.zeros([int((tact_train_end-tact_train_start)/binsize),nstims_t])

# 	for i in range(len(order_t)):
# 		epochs = np.array(output_t)[grouped_t[i,:].astype(np.int)]
# 		for epoch in epochs:
# 			if epoch[1]<tact_train_end:
# 				test = epoch[0]
# 				ind = np.floor((test - tact_train_start)/binsize)
# 				bstart = tact_train_start + ind*binsize
# 				bend = tact_train_start + (ind+1)*binsize
# 				if (test - bstart)>(bend-test):
# 					start_bin = ind+1
# 				else:
# 					start_bin = ind
# 				test = epoch[1]
# 				ind = np.floor((test - tact_train_start)/binsize)
# 				bstart = tact_train_start + ind*binsize
# 				bend = tact_train_start + (ind+1)*binsize
# 				if (test - bstart)>(bend-test):
# 					end_bin = ind
# 				else:
# 					end_bin = ind+1
# 				Atemp_t[start_bin:end_bin+1,i] = 1.0

# 	A_t = np.zeros([Atemp_t.shape[0],Atemp_t.shape[1]*(ndelays*2+1)])

# 	for i in range(-ndelays,ndelays+1):
# 	    if i<=0:
# 	        temp = Atemp_t[-i:,:].copy()
# 	        A_t[:temp.shape[0],(i+ndelays)*Atemp_t.shape[1]:(i+ndelays+1)*Atemp_t.shape[1]] = temp
# 	    else:
# 	        temp = Atemp_t[:-i,:].copy()
# 	        A_t[i:i+temp.shape[0],(i+ndelays)*Atemp_t.shape[1]:(i+ndelays+1)*Atemp_t.shape[1]] = temp
	
# 	nbins = np.floor((vis_train_end-vis_train_start)/binsize)
# 	vis_train_bins = np.linspace(vis_train_start+binsize,vis_train_start + (binsize*nbins),nbins)
# 	binned_temp=np.array([bin for bin in plx.spikes.bin(vis_train_bins, binlen=binsize)])
# 	vis_train_spikes = binned_temp[:,np.array([udict[unit] for unit in unitlist])]

# 	Atemp_v = np.zeros([int((vis_train_end-vis_train_start)/binsize),nstims_v])

# 	for i in range(len(order_v)):
# 		epochs = np.array(output_v)[grouped_v[i,:].astype(np.int)]
# 		for epoch in epochs:
# 			if epoch[1]<vis_train_end:
# 				test = epoch[0]
# 				ind = np.floor((test - vis_train_start)/binsize)
# 				bstart = vis_train_start + ind*binsize
# 				bend = vis_train_start + (ind+1)*binsize
# 				if (test - bstart)>(bend-test):
# 					start_bin = ind+1
# 				else:
# 					start_bin = ind
# 				test = epoch[1]
# 				ind = np.floor((test - vis_train_start)/binsize)
# 				bstart = vis_train_start + ind*binsize
# 				bend = vis_train_start + (ind+1)*binsize
# 				if (test - bstart)>(bend-test):
# 					end_bin = ind
# 				else:
# 					end_bin = ind+1
# 				Atemp_v[start_bin:end_bin+1,i] = 1.0

# 	A_v = np.zeros([Atemp_v.shape[0],Atemp_v.shape[1]*(ndelays*2+1)])

# 	for i in range(-ndelays,ndelays+1):
# 	    if i<=0:
# 	        temp = Atemp_v[-i:,:].copy()
# 	        A_v[:temp.shape[0],(i+ndelays)*Atemp_v.shape[1]:(i+ndelays+1)*Atemp_v.shape[1]] = temp
# 	    else:
# 	        temp = Atemp_v[:-i,:].copy()
# 	        A_v[i:i+temp.shape[0],(i+ndelays)*Atemp_v.shape[1]:(i+ndelays+1)*Atemp_v.shape[1]] = temp

# 	##############


# 	#######Validation data
# 	#Bin spikes
# 	nbins = np.floor((tact_val_end-tact_val_start)/binsize)
# 	tact_val_bins = np.linspace(tact_val_start+binsize,tact_val_start + (binsize*nbins),nbins)
# 	binned_temp=np.array([bin for bin in plx.spikes.bin(tact_val_bins, binlen=binsize)])
# 	tact_val_spikes = binned_temp[:,np.array([udict[unit] for unit in unitlist])]

# 	################ Create design matrix
# 	Atemp_t = np.zeros([int((tact_val_end-tact_val_start)/binsize),nstims_t])

# 	for i in range(len(order_t)):
# 		epochs = np.array(output_t)[grouped_t[i,:].astype(np.int)]
# 		for epoch in epochs:
# 			if epoch[1]<tact_val_end:
# 				test = epoch[0]
# 				ind = np.floor((test - tact_val_start)/binsize)
# 				bstart = tact_val_start + ind*binsize
# 				bend = tact_val_start + (ind+1)*binsize
# 				if (test - bstart)>(bend-test):
# 					start_bin = ind+1
# 				else:
# 					start_bin = ind
# 				test = epoch[1]
# 				ind = np.floor((test - tact_val_start)/binsize)
# 				bstart = tact_val_start + ind*binsize
# 				bend = tact_val_start + (ind+1)*binsize
# 				if (test - bstart)>(bend-test):
# 					end_bin = ind
# 				else:
# 					end_bin = ind+1
# 				Atemp_t[start_bin:end_bin+1,i] = 1.0

# 	A_t_val = np.zeros([Atemp_t.shape[0],Atemp_t.shape[1]*(ndelays*2+1)])

# 	for i in range(-ndelays,ndelays+1):
# 	    if i<=0:
# 	        temp = Atemp_t[-i:,:].copy()
# 	        A_t_val[:temp.shape[0],(i+ndelays)*Atemp_t.shape[1]:(i+ndelays+1)*Atemp_t.shape[1]] = temp
# 	    else:
# 	        temp = Atemp_t[:-i,:].copy()
# 	        A_t_val[i:i+temp.shape[0],(i+ndelays)*Atemp_t.shape[1]:(i+ndelays+1)*Atemp_t.shape[1]] = temp
	
# 	nbins = np.floor((vis_val_end-vis_val_start)/binsize)
# 	vis_val_bins = np.linspace(vis_val_start+binsize,vis_val_start + (binsize*nbins),nbins)
# 	binned_temp=np.array([bin for bin in plx.spikes.bin(vis_val_bins, binlen=binsize)])
# 	vis_val_spikes = binned_temp[:,np.array([udict[unit] for unit in unitlist])]

# 	Atemp_v = np.zeros([int((vis_val_end-vis_val_start)/binsize),nstims_v])

# 	for i in range(len(order_v)):
# 		epochs = np.array(output_v)[grouped_v[i,:].astype(np.int)]
# 		for epoch in epochs:
# 			if epoch[1]<vis_val_end:
# 				test = epoch[0]
# 				ind = np.floor((test - vis_val_start)/binsize)
# 				bstart = vis_val_start + ind*binsize
# 				bend = vis_val_start + (ind+1)*binsize
# 				if (test - bstart)>(bend-test):
# 					start_bin = ind+1
# 				else:
# 					start_bin = ind
# 				test = epoch[1]
# 				ind = np.floor((test - vis_val_start)/binsize)
# 				bstart = vis_val_start + ind*binsize
# 				bend = vis_val_start + (ind+1)*binsize
# 				if (test - bstart)>(bend-test):
# 					end_bin = ind
# 				else:
# 					end_bin = ind+1
# 				Atemp_v[start_bin:end_bin+1,i] = 1.0

# 	A_v_val = np.zeros([Atemp_v.shape[0],Atemp_v.shape[1]*(ndelays*2+1)])

# 	for i in range(-ndelays,ndelays+1):
# 	    if i<=0:
# 	        temp = Atemp_v[-i:,:].copy()
# 	        A_v_val[:temp.shape[0],(i+ndelays)*Atemp_v.shape[1]:(i+ndelays+1)*Atemp_v.shape[1]] = temp
# 	    else:
# 	        temp = Atemp_v[:-i,:].copy()
# 	        A_v_val[i:i+temp.shape[0],(i+ndelays)*Atemp_v.shape[1]:(i+ndelays+1)*Atemp_v.shape[1]] = temp


# ##########################################################################


	# Separate PTSH for visual vs. tactile trials
	print count, "separating vis and tact stims"
	sys.stdout.flush()
	psth_t = np.zeros([spike_counts.shape[0],spike_counts.shape[1],len(order_t)*grouped_t.shape[1]])
	for i, zone in enumerate(order_t):
	    psth_t[:,:,i*grouped_t.shape[1]:(i+1)*grouped_t.shape[1]] = spike_counts[:,:,grouped_t[i,:].astype(int)]
	psth_v = np.zeros([spike_counts.shape[0],spike_counts.shape[1],len(order_v)*grouped_v.shape[1]])
	for i, zone in enumerate(order_t):
	    psth_v[:,:,i*grouped_v.shape[1]:(i+1)*grouped_v.shape[1]] = spike_counts[:,:,grouped_v[i,:].astype(int)]

	# Calculate baseline FR for the day
	print count, "calculating baselines"
	sys.stdout.flush()
	baseline, bsd, bfr = baseline_fr(plx, unitlist, np.array([5.0,plx.length-5.0]),binsize=.1)

	# Z score firing rates
	fr_t = (psth_t - baseline[None,:,None])/bsd[None,:,None]
	fr_v = (psth_v - baseline[None,:,None])/bsd[None,:,None]
	bfr = (bfr - baseline)/bsd

	######### REGRESSION ANALYSIS
	#tact_train_spikes = (tact_train_spikes - baseline*binsize)/bsd*binsize
	#vis_train_spikes = (vis_train_spikes - baseline*binsize)/bsd*binsize
	#tact_val_spikes = (tact_val_spikes - baseline*binsize)/bsd*binsize
	#vis_val_spikes = (vis_val_spikes - baseline*binsize)/bsd*binsize
	######################

	# Create master list of results for each zone that other days will be added to
	print count, "saving data"
	sys.stdout.flush()
	if count==0:
		master_t = [fr_t[:,:,i*grouped_t.shape[1]:(i+1)*grouped_t.shape[1]] for i in range(len(order_t))]
		master_v = [fr_v[:,:,i*grouped_v.shape[1]:(i+1)*grouped_v.shape[1]] for i in range(len(order_v))]
		ord_t = order_t[:]
		ord_v = order_v[:]
		master_bl = bfr.copy()
		# A_t_master = A_t.copy()
		# A_v_master = A_v.copy()
		# A_t_val_master = A_t_val.copy()
		# A_v_val_master = A_v_val.copy()
		# tact_train_spikes_master = tact_train_spikes.copy()
		# vis_train_spikes_master = vis_train_spikes.copy()
		# tact_val_spikes_master = tact_val_spikes.copy()
		# vis_val_spikes_master = vis_val_spikes.copy()
	else:
		master_t = [np.concatenate((master_t[i],fr_t[:,:,i*grouped_t.shape[1]:(i+1)*grouped_t.shape[1]]),axis=2) for i in range(len(ord_t))]
		master_v = [np.concatenate((master_v[i],fr_v[:,:,i*grouped_v.shape[1]:(i+1)*grouped_v.shape[1]]),axis=2) for i in range(len(ord_v))]
		master_bl = np.concatenate((master_bl,bfr),axis=0)
		# A_t_master = np.concatenate((A_t_master, A_t))
		# A_v_master = np.concatenate((A_v_master, A_v))
		# A_t_val_master = np.concatenate((A_t_val_master, A_t_val))
		# A_v_val_master = np.concatenate((A_v_val_master, A_v_val))
		# tact_train_spikes_master = np.concatenate((tact_train_spikes_master,tact_train_spikes))
		# vis_train_spikes_master = np.concatenate((vis_train_spikes_master,vis_train_spikes))
		# tact_val_spikes_master = np.concatenate((tact_val_spikes_master,tact_val_spikes))
		# vis_val_spikes_master = np.concatenate((vis_val_spikes_master,vis_val_spikes))