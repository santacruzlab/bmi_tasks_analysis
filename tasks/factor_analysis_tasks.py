from bmimultitasks import BMIControlMulti, BMIResetting, BMIResettingObstacles
from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife
from cursor_clda_tasks import CLDARMLKFOFCIVC
from riglib.experiment import traits
from riglib.bmi.clda import OFCLearner
import sklearn.decomposition as skdecomp
import numpy as np
from riglib.bmi import extractor
import tables
import pickle
import matplotlib.pyplot as plt
from riglib.bmi import clda
from riglib.bmi.clda import FeedbackControllerLearner
from riglib.bmi import feedback_controllers

class FactorBMIBase(BMIResetting):

    #Choose task to use trials from (no assist)
    #TODO make offline function to quickly assess optimal number of factors
    sequence_generators = ['centerout_2D_discrete','generate_catch_trials', 'all_shar_trials'] 
    input_type_list = ['shared','private', 'shared_scaled', 'private_scaled', 'all', 'all_scaled_by_shar',
        'sc_shared+unsc_priv', 'sc_shared+sc_priv', 'main_shared', 'main_sc_shared','main_sc_private', 'main_sc_shar+unsc_priv',
        'main_sc_shar+sc_priv','pca', 'split']#, 'priv_shar_concat']
    input_type = traits.OptionsList(*input_type_list, bmi3d_input_options=input_type_list)


    def init(self):
        nU = self.decoder.n_units

        #Add datatypes for 1) trial type 2) input types: 
        self.decoder.filt.FA_input_dict = {}

        self.input_types = [i+'_input' for i in self.input_type_list] + ['task_input']


        for k in self.input_types:
            if k == 'split_input':
                nUn = self.decoder.trained_fa_dict['fa_main_shar_n_dim'] + nU
                #nUn = 2*nU
            elif k=='task_input' and self.input_type == 'split':
                nUn = self.decoder.trained_fa_dict['fa_main_shar_n_dim'] + nU
                #nUn = 2*nU
            
            else:
                nUn = nU
            self.decoder.filt.FA_input_dict[k] = np.zeros((nUn, 1)) 
            self.decoder.filt.FA_input_dict[k][:] = np.nan
            self.add_dtype(k, 'f8', (nUn, 1))

        #Add datatype for 'trial-type':
        self.add_dtype('fa_input', np.str_, 16)
        super(FactorBMIBase, self).init()

    def init_decoder_state(self):
        try: 
            fa_dict = self.decoder.trained_fa_dict
            self.decoder.filt.FA_kwargs = fa_dict
    
        except:
            raise Exception('Must run riglib.train.add_fa_dict_to_decoder and resave with dbq.save')      

        #Check if isinstance of FAKalmanFilter or KalmanFilter
        from riglib.bmi.kfdecoder import KalmanFilter, FAKalmanFilter

        if isinstance(self.decoder.filt, KalmanFilter):
            self.decoder.filt.__class__ = FAKalmanFilter

        #Add FA elements to dict:
        print 'adding ', self.input_type, ' as input_type to FA decoder'
        import time
        time.sleep(2.)
        self.decoder.filt.FA_input = self.input_type
        super(FactorBMIBase, self).init_decoder_state()

    def _cycle(self):
        for k in self.input_types:
            try:
                self.task_data[k] = self.decoder.filt.FA_input_dict[k]
            except:
                print k, self.decoder.filt.FA_input_dict[k].shape, self.task_data[k].shape
        self.task_data['fa_input'] = self.decoder.filt.FA_input

        super(FactorBMIBase, self)._cycle()

    def _parse_next_trial(self):
        if type(self.next_trial[1]) is str:
            self.targs = self.next_trial[0]
        else:
            self.targs = self.next_trial
        #print 'trial: ', self.decoder.filt.FA_input, self.targs

    ### FA param saving functions ###: 
    def cleanup_hdf(self):
        '''
        Re-open the HDF file and save any extra task data kept in RAM
        '''
        super(FactorBMIBase, self).cleanup_hdf()
        try:
            self.write_FA_data_to_hdf_table(self.h5file.name, self.decoder.filt.FA_kwargs)
            print 'writing FA params to HDF file'
        except:
            print 'error in writing FA params to hdf file'
            import traceback
            traceback.print_exc()

    @staticmethod
    def write_FA_data_to_hdf_table(hdf_fname, FA_dict, ignore_none=False):
        
        import tables
        compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)

        h5file = tables.openFile(hdf_fname, mode='a')
        fa_grp = h5file.createGroup(h5file.root, "fa_params", "Parameters for FA model used")

        for key in FA_dict:
            if isinstance(FA_dict[key], np.ndarray):
                h5file.createArray(fa_grp, key, FA_dict[key])
            else:
                try:
                    h5file.createArray(fa_grp, key, np.array([FA_dict[key]]))
                except:
                    print 'cannot save: ', key, 'from FA in hdf file'
        h5file.close()

    @classmethod
    def generate_FA_matrices(self, training_task_entry, plot=False, hdf=None, dec=None, bin_spk=None):
        
        import utils.fa_decomp as pa
        if bin_spk is None:
            if training_task_entry is not None:
                from db import dbfunctions as dbfn
                te = dbfn.TaskEntry(training_task_entry)
                hdf = te.hdf
                dec = te.decoder

            bin_spk, targ_pos, targ_ix, z, zz = self.extract_trials_all(hdf, dec)

        #Zscore is in time x neurons
        zscore_X, mu = self.zscore_spks(bin_spk)

        # #Find optimal number of factors: 
        LL, psv = pa.find_k_FA(zscore_X, iters=3, max_k = 10, plot=False)

        #Np.nanmean:
        nan_ix = np.isnan(LL)
        samp = np.sum(nan_ix==False, axis=0)
        ll = np.nansum(LL, axis=0)
        LL_new = np.divide(ll, samp)
        
        num_factors = 1+(np.argmax(LL_new))
        print 'optimal LL factors: ', num_factors

        FA = skdecomp.FactorAnalysis(n_components=num_factors)

        #Samples x features:
        FA.fit(zscore_X)

        #FA matrices: 
        U = np.mat(FA.components_).T
        i = np.diag_indices(U.shape[0])
        Psi = np.mat(np.zeros((U.shape[0], U.shape[0])))
        Psi[i] = FA.noise_variance_
        A = U*U.T
        B = np.linalg.inv(U*U.T + Psi)
        mu_vect = np.array([mu[0,:]]).T #Size = N x 1
        sharL = A*B

        #Calculate shared / priv scaling:
        bin_spk_tran = bin_spk.T
        mu_mat = np.tile(np.array([mu[0,:]]).T, (1, bin_spk_tran.shape[1]))
        demn = bin_spk_tran - mu_mat
        shared_bin_spk = (sharL*demn)
        priv_bin_spk = bin_spk_tran - mu_mat - shared_bin_spk

        #Scaling:
        eps = 1e-15
        x_var = np.var(np.mat(bin_spk_tran), axis=1) + eps
        pr_var = np.var(priv_bin_spk, axis=1) + eps
        sh_var = np.var(shared_bin_spk,axis=1) + eps

        priv_scalar = np.sqrt(np.divide(x_var, pr_var))
        shared_scalar = np.sqrt(np.divide(x_var, sh_var))

        if plot:
            tmp = np.diag(U.T*U)
            plt.plot(np.arange(1, num_factors+1), np.cumsum(tmp)/np.sum(tmp), '.-')
            plt.plot([0, num_factors+1], [.9, .9], '-')

        #Get main shared space: 
        u, s, v = np.linalg.svd(A)
        s_red = np.zeros_like(s)
        s_hd = np.zeros_like(s)

        ix = np.nonzero(np.cumsum(s**2)/float(np.sum(s**2))>.90)[0]
        if len(ix) > 0:
            n_dim_main_shared = ix[0]+1
        else:
            n_dim_main_shared = len(s)
        if n_dim_main_shared < 2:
            n_dim_main_shared = 2
        print "main shared: n_dim: ", n_dim_main_shared, np.cumsum(s)/float(np.sum(s))
        s_red[:n_dim_main_shared] = s[:n_dim_main_shared]
        s_hd[n_dim_main_shared:] = s[n_dim_main_shared:]

        main_shared_A = u*np.diag(s_red)*v
        hd_shared_A = u*np.diag(s_hd)*v
        main_shared_B = np.linalg.inv(main_shared_A + hd_shared_A + Psi)

        uut_psi_inv = main_shared_B.copy()
        u_svd = u[:, :n_dim_main_shared]

        main_sharL = main_shared_A*main_shared_B

        main_shar = main_sharL*demn
        main_shar_var = np.var(main_shar, axis=1) + eps
        main_shar_scal = np.sqrt(np.divide(x_var, main_shar_var))

        main_priv = demn - main_shar
        main_priv_var = np.var(main_priv, axis=1) + eps
        main_priv_scal = np.sqrt(np.divide(x_var, main_priv_var))

        # #Get PCA decomposition:
        #LL, ax = pa.FA_all_targ_ALLms(hdf, iters=2, max_k=20, PCA_instead=True)
        #num_PCs = 1+(np.argmax(np.mean(LL, axis=0)))

        # Main PCA space: 
        # Get cov matrix: 
        cov_pca = np.cov(zscore_X.T)
        eig_val, eig_vec = np.linalg.eig(cov_pca)

        tot_var = sum(eig_val)
        cum_var_exp = np.cumsum([i/tot_var for i in sorted(eig_val, reverse=True)])
        n_PCs = np.nonzero(cum_var_exp > 0.9)[0][0]+1

        proj_mat = eig_vec[:, :n_PCs]
        proj_trans = np.mat(proj_mat)*np.mat(proj_mat.T)

        #PC matrices:
        return dict(fa_sharL=sharL, fa_mu=mu_vect, fa_shar_var_sc=shared_scalar, fa_priv_var_sc=priv_scalar, 
            U=U, Psi=Psi, training_task_entry=training_task_entry, FA_iterated_power=FA.iterated_power, 
            FA_score=FA.score(zscore_X), FA_LL = np.array(FA.loglike_), fa_main_shared=main_sharL, 
            fa_main_shared_sc=main_shar_scal, fa_main_private_sc=main_priv_scal,
            fa_main_shar_n_dim = n_dim_main_shared, sing_vals=s, own_pc_trans=proj_trans, FA_model=FA, 
            uut_psi_inv=uut_psi_inv, u_svd=u_svd)


    @classmethod
    def zscore_spks(self, proc_spks):
        '''Assumes a time x units matrix'''
        mu = np.tile(np.mean(proc_spks, axis=0), (proc_spks.shape[0], 1))
        zscore_X = proc_spks - mu
        return zscore_X, mu

    @classmethod
    def extract_trials_all(self, hdf, dec, neural_bins = 100, time_cutoff=None, hdf_ix=False):
        '''
        Summary: method to extract all time points from trials
        Input param: hdf: task file input
        Input param: rew_ix: rows in the hdf file corresponding to reward times
        Input param: neural_bins: ms per bin
        Input param: time_cutoff: time in minutes, only extract trials before this time
        Input param: hdf_ix: bool, whether to return hdf row corresponding to time of decoder 
        update (and hence end of spike bin)

        Output param: bin_spk -- binned spikes in time x units
                      targ_i_all -- target location at each update
                      targ_ix -- target index 
                      trial_ix -- trial number
                      reach_time -- reach time for trial
                      hdf_ix -- end bin in units of hdf rows
        '''
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

        if time_cutoff is not None:
            it_cutoff = time_cutoff*60*60
        else:
            it_cutoff = len(hdf.root.task)
        #Get Go cue and 
        go_ix = np.array([hdf.root.task_msgs[it-3][1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0] == 'reward'])
        go_ix = go_ix[go_ix<it_cutoff]
        rew_ix = rew_ix[go_ix < it_cutoff]

        targ_i_all = np.array([[-1, -1]])
        trial_ix_all = np.array([-1])
        reach_tm_all = np.array([-1])
        hdf_ix_all = np.array([-1])

        bin_spk = np.zeros((1, hdf.root.task[0]['spike_counts'].shape[0]))-1
        drives_neurons = dec.drives_neurons
        drives_neurons_ix0 = np.nonzero(drives_neurons)[0][0]
        update_bmi_ix = np.nonzero(np.diff(np.squeeze(hdf.root.task[:]['internal_decoder_state'][:, drives_neurons_ix0, 0])))[0]+1

        for ig, (g, r) in enumerate(zip(go_ix, rew_ix)):
            spk_i = hdf.root.task[g:r]['spike_counts'][:,:,0]

            #Sum spikes in neural_bins:
            bin_spk_i, nbins, hdf_ix_i = self._bin_spks(spk_i, g, r, update_bmi_ix)
            bin_spk = np.vstack((bin_spk, bin_spk_i))
            targ_i_all = np.vstack(( targ_i_all, np.tile(hdf.root.task[g+1]['target'][[0,2]], (bin_spk_i.shape[0], 1)) ))
            trial_ix_all = np.hstack(( trial_ix_all, np.zeros(( bin_spk_i.shape[0] ))+ig ))
            reach_tm_all = np.hstack((reach_tm_all, np.zeros(( bin_spk_i.shape[0] ))+((r-g)*1000./60.) ))
            hdf_ix_all = np.hstack((hdf_ix_all, hdf_ix_i ))

        targ_ix = self._get_target_ix(targ_i_all[1:,:])

        if hdf_ix:
            return bin_spk[1:,:], targ_i_all[1:,:], targ_ix, trial_ix_all[1:], reach_tm_all[1:], hdf_ix_all[1:]
        else:
            return bin_spk[1:,:], targ_i_all[1:,:], targ_ix, trial_ix_all[1:], reach_tm_all[1:]

    @classmethod
    def _get_target_ix(self, targ_pos):
        #Target Index: 
        b = np.ascontiguousarray(targ_pos).view(np.dtype((np.void, targ_pos.dtype.itemsize * targ_pos.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_targ = targ_pos[idx,:]

        #Order by theta: 
        theta = np.arctan2(unique_targ[:,1],unique_targ[:,0])
        thet_i = np.argsort(theta)
        unique_targ = unique_targ[thet_i, :]
        
        targ_ix = np.zeros((targ_pos.shape[0]), )
        for ig, (x,y) in enumerate(targ_pos):
            targ_ix[ig] = np.nonzero(np.sum(targ_pos[ig,:]==unique_targ, axis=1)==2)[0]
        return targ_ix

    @classmethod
    def _bin_spks(self, spk_i, g_ix, r_ix, update_bmi_ix):

        #Need to use 'update_bmi_ix' from ReDecoder to get bin edges correctly:
        trial_inds = np.arange(g_ix, r_ix+1)
        end_bin = np.array([(j,i) for j, i in enumerate(trial_inds) if np.logical_and(i in update_bmi_ix, i>=(g_ix+5))])
        nbins = len(end_bin)
        bin_spk_i = np.zeros((nbins, spk_i.shape[1]))

        hdf_ix_i = []
        for ib, (i_ix, hdf_ix) in enumerate(end_bin):
            #Inclusive of EndBin
            bin_spk_i[ib,:] = np.sum(spk_i[i_ix-5:i_ix+1,:], axis=0)
            hdf_ix_i.append(hdf_ix)
        return bin_spk_i, nbins, np.array(hdf_ix_i)
    @staticmethod
    def generate_catch_trials(nblocks=5, ntargets=8, distance=10, perc_shar=10, perc_priv=10):
        '''
        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin and a second field indicating the extractor type (full, shared, priv)

        1 shared / 1 private for 

        nblocks: multiples of 80 
        perc_shar, perc_priv: multiples of 10, please

        '''
        assert (not np.mod(perc_shar, 10)) and (not np.mod(perc_priv, 10))

        #Make blocks of 80 trials: 
        theta = []
        for i in range(10):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)

        #Each target has correct % of private and correct % of shared targets
        trial_type = np.empty(len(theta), dtype='S10')

        for i in temp:
            targ_ix = np.nonzero(theta==i)[0]
            trial_ix = np.arange(len(targ_ix))
            tmp_trial = np.array(['all']*len(targ_ix), dtype='S10')

            n_trial_shar = np.floor(perc_shar/100.*float(len(targ_ix)))
            n_trial_priv = np.floor(perc_priv/100.*float(len(targ_ix)))

            tmp_trial[:int(n_trial_shar)] = ['shared']
            tmp_trial[int(n_trial_shar):int(n_trial_shar+n_trial_priv)] = ['private']
            np.random.shuffle(tmp_trial)
            trial_type[targ_ix] = tmp_trial

        #Make Target set: 
        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T

        Pairs = np.tile(pairs, [nblocks, 1, 1])
        Trial_type = np.tile(trial_type, [nblocks])


        #Will yield a tuple where target location is in next_trial[0], trial_type is in next_trial[1]
        return zip(Pairs, Trial_type)

    @staticmethod
    def all_shar_trials(nblocks=5, ntargets=8, distance=10):
        '''
        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin and a second field indicating the extractor type (always shared)
        '''
        #Make blocks of 80 trials: 
        theta = []
        for i in range(10):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)

        #Each target has correct % of private and correct % of shared targets
        trial_type = np.empty(len(theta), dtype='S10')
        trial_type[:] = 'shared'

        #Make Target set: 
        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T

        Pairs = np.tile(pairs, [nblocks, 1, 1])
        Trial_type = np.tile(trial_type, [nblocks])

        #Will yield a tuple where target location is in next_trial[0], trial_type is in next_trial[1]
        return zip(Pairs, Trial_type)

class FactorBMI_catch_trials(FactorBMIBase):
    sequence_generators = ['generate_catch_trials', 'all_shar_trials'] 

    def _parse_next_trial(self):
        self.targs = self.next_trial[0]
        #Define extractor kwargs type used in BMILoop to get correct neural data
        if self.next_trial[1] == 'all':
            tr_type = self.next_trial[1]
        else:
            tr_type = self.next_trial[1]+'_scaled'
        self.decoder.filt.FA_input = tr_type
        print 'trial: ', self.decoder.filt.FA_input

class FactorBMI_w_obstacles(FactorBMIBase, BMIResettingObstacles):
    sequence_generators = ['centerout_2D_discrete_w_obstacle', 'generate_catch_trials', 'all_shar_trials','centerout_2D_discrete'] 
    
    # def __init__(self, *args, **kwargs):
    #     super(FactorBMI_w_obstacles, self).__init__(self, *args, **kwargs)
    def _parse_next_trial(self):
        self.targs = self.next_trial[0]
        #Width and height of obstacle
        self.obstacle_size = self.next_trial[1]
        self.obstacle = self.obstacle_list[self.obstacle_dict[self.obstacle_size]]
        self.obstacle_location = self.next_trial[2]

class FactorBMI_catch_w_obstacles(BMIResettingObstacles, FactorBMI_catch_trials):
    sequence_generators = ['generate_catch_trials', 'all_shar_trials'] 

    def __init__(self, *args, **kwargs):
        super(FactorBMI_catch_w_obstacles, self).__init__(self, *args, **kwargs)

class CLDA_FactorBMIBase(FactorBMIBase, LinearlyDecreasingHalfLife):
    #Choose task to use trials from (no assist)
    
    sequence_generators = ['centerout_2D_discrete'] 
    memory_decay_rate = traits.Float(0.5, desc="")
    batch_time = traits.Float(0.1, desc='The length of the batch in seconds')
    decoder_sequence = traits.String('test', desc='signifier to group together sequences of decoders')
    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'batch_time', 'half_life', 'half_life_time']

    def __init__(self, *args, **kwargs):
        super(CLDA_FactorBMIBase, self).__init__(*args,**kwargs)
        self.learn_flag = True

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(CLDA_FactorBMIBase, self).call_decoder(*args, **kwargs)

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])
        self.updater.default_gain = self.memory_decay_rate

    def create_learner(self):
        # self.batch_size = int(self.batch_time/self.decoder.binlen)
        # self.learner = ObstacleLearner(self.batch_size)
        # self.learn_flag = True

        self.batch_size = int(self.batch_time/self.decoder.binlen)
        A, B, _ = self.decoder.ssm.get_ssm_matrices()
        Q = np.mat(np.diag([10, 10, 10, 5, 5, 5, 0]))
        R = 10**6*np.mat(np.eye(B.shape[1]))
        from tentaclebmitasks import OFCLearnerTentacle
        self.learner = OFCLearnerTentacle(self.batch_size, A, B, Q, R)
        self.learn_flag = True

class ObstacleLearner(FeedbackControllerLearner):
    def __init__(self, batch_size, decoding_rate=60., *args, **kwargs):
        F_dict = pickle.load(open('/storage/assist_params/assist_20levels_ppf.pkl'))
        F = F_dict[-1]
        B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))
        fb_ctrl = feedback_controllers.LinearFeedbackController(A=B, B=B, F=F)
        super(ObstacleLearner, self).__init__(batch_size, fb_ctrl, style='additive')

class CLDA_BMIResettingObstacles(BMIResettingObstacles, LinearlyDecreasingHalfLife):
    sequence_generators = ['centerout_2D_discrete', 'centerout_2D_discrete_w_obstacle'] 
    batch_time = traits.Float(0.1, desc='The length of the batch in seconds')
    decoder_sequence = traits.String('test', desc='signifier to group together sequences of decoders')
    memory_decay_rate = traits.Float(0.5, desc="")
    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'batch_time', 'half_life', 'half_life_time']

    def __init__(self, *args, **kwargs):
        super(CLDA_BMIResettingObstacles, self).__init__(*args, **kwargs)
        self.learn_flag = True

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(CLDA_BMIResettingObstacles, self).call_decoder(*args, **kwargs)

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])
        self.updater.default_gain = self.memory_decay_rate

    def create_learner(self):
        # self.batch_size = int(self.batch_time/self.decoder.binlen)
        # self.learner = ObstacleLearner(self.batch_size)
        # self.learn_flag = True

        self.batch_size = int(self.batch_time/self.decoder.binlen)
        A, B, _ = self.decoder.ssm.get_ssm_matrices()
        Q = np.mat(np.diag([10, 10, 10, 5, 5, 5, 0]))
        R = 10**6*np.mat(np.eye(B.shape[1]))
        from tentaclebmitasks import OFCLearnerTentacle
        self.learner = OFCLearnerTentacle(self.batch_size, A, B, Q, R)
        self.learn_flag = True

class CLDA_FactorBMIResettingObstacles(FactorBMI_w_obstacles, CLDA_BMIResettingObstacles):
    sequence_generators = ['centerout_2D_discrete', 'centerout_2D_discrete_w_obstacle'] 
    def __init__(self, *args, **kwargs):
        super(CLDA_FactorBMIResettingObstacles, self).__init__(*args, **kwargs)

