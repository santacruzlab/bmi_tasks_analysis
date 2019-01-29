"""
THIS FILE IS DEPRECATED. See bmimultitsks.py for virtual BMI tasks. 
This file remains in the repository only for legacy/recordkeeping purposes
"""

import os
from manualcontrol import *
from riglib.bmi import clda
import riglib.bmi
import pdb
import multiprocessing as mp
import time
import pickle

class VisualFeedback(ManualControl):

    background = (.5,.5,.5,1) # Set the screen background color to grey
    noise_level = traits.Float(0.0,desc="Percent noise to add to straight line movements.")

    def __init__(self, *args, **kwargs):
        self.prev_cursor = np.array([0,0,0])
        super(VisualFeedback, self).__init__(*args, **kwargs)

    def update_cursor(self):
        self.update_target_location()
        # create random noise x and z velocities
        velnoise = np.array([np.random.randn(1)[0], 0, np.random.randn(1)[0]])*.1
        # calculate straight line x and z velocities
        targetvec = self.target_xz/10. - self.prev_cursor[[0,2]] #divide target units by 10 to convert from mm to cm
        vecmag = np.sqrt(targetvec[0]**2+ targetvec[1]**2)
        if vecmag<.05:
        	velideal = np.array([0,0,0])
        else:
        	direction = targetvec/vecmag
        	velideal = np.array([direction[0],0,direction[1]])*.1 # constant velocity for now, maybe change to bell shaped curve later??
        # combine ideal velocity with noise
        vel = velideal*(1-self.noise_level) + velnoise*self.noise_level
        # calculate new cursor position
        pt = self.prev_cursor + vel
        self.cursor.translate(*pt,reset=True)
        self.prev_cursor = pt.copy()
        self.task_data['cursor'] = self.cursor.xfm.move.copy()
        
        #write to screen
        self.draw_world()

class ManualWithPredictions(ManualControl):

    '''
    Manual control task with a small cursor showing predicted positions at the
    same time.
    '''

    def __init__(self, *args, **kwargs):
        super(ManualWithPredictions, self).__init__(*args, **kwargs)
        # Create prediction cursor
        self.predicted = Sphere(radius=.2, color=(1,1,1,.5))
        self.add_model(self.predicted)

    def update_cursor(self):
        # Get data from 1st marker on motion tracker,
        # take average of all data points since last poll
        pt = self.motiondata.get()
        if len(pt) > 0:
            pt = pt[:, 0, :]
            conds = pt[:, 3]
            inds = np.nonzero((conds>=0) & (conds!=4))
            if len(inds[0]) > 0:
                pt = pt[inds[0],:3]
                #pt -= [430, -130, 385]
                pt = pt.mean(0) * 0.35
                #ignore y direction
                pt[1] = 0
                #move cursor to marker location
                self._update(pt)
            else:
                self.no_data_count += 1
        else:
            self.no_data_count +=1
        pt = self.neurondata.get(all=True).copy()
        pt[1] = 0
        self.predicted.translate(*pt[:3]*.35,reset=True)
        #write to screen
        self.draw_world()

class BMIControl(ManualControl):
    '''
    Target capture task with cursor position controlled by BMI output.
    '''

    background = (.5,.5,.5,1) # Set the screen background color to grey
    assist_level = traits.Float(0.0,desc="Level of assist to apply to BMI output")
    max_attempts = traits.Int(3, desc='The number of attempts at a target before\
        skipping to the next one')
    update_rate = 1.0/60 #rate that the decoder is called

    def __init__(self, *args, **kwargs):
        super(BMIControl, self).__init__(*args, **kwargs)
        self.cursor.color = (1.0,1.0,1.0,1) # Set cursor color to white

    def init(self):

        # Number of display loop iterations equal to one decoder bin. Best if 1/decoder.binlen is a multiple of 1/60, otherwise
        # spike counts won't be exactly correct.
        self.bmicount = 0
        self.bminum = int(self.decoder.binlen/self.update_rate)
        self.dtype.append(('spike_counts','u4',(len(self.decoder.units), ))) 
        self.dtype.append(('bin_edges','f8',2))
        try:
            self.n_subbins = self.decoder.n_subbins
        except:
            self.n_subbins = 1
        print "#$ of sub bins: %d" % self.n_subbins

        super(BMIControl, self).init()        
        
    def update_target_location(self):
        # Determine the task target for assist/decoder adaptation purposes (convert
        # units from cm to mm for decoder)
        if self.state=='origin' or self.state=='origin_hold':
            self.location = self.origin_target.xfm.move
        elif self.state=='terminus' or self.state=='terminus_hold':
            self.location = self.terminus_target.xfm.move
        self.task_data['target'] = self.location[:3]

    def call_decoder(self, spike_counts):
        # Get the decoder output & convert from mm to cm
        return self.decoder(spike_counts, task_data=self.task_data,
            target=self.target_xz, assist_level=self.assist_level)

    def get_spike_ts(self):
        '''
        Get the array of spike timestamps
        '''
        return self.neurondata.get()

    ### def get_spike_counts(self):
    ###     '''
    ###     Get the binned spike counts & record the smallest and largest 
    ###     timestamps used in the bin
    ###     '''
    ###     ts = self.get_spike_ts()
    ###     if len(ts) == 0:
    ###         counts = np.zeros(len(self.decoder.units))
    ###         self.task_data['bin_edges'] = np.array([np.nan, np.nan])
    ###     else:
    ###         min_ind = np.argmin(ts['ts'])
    ###         max_ind = np.argmax(ts['ts'])
    ###         self.task_data['bin_edges'] = np.array([ts[min_ind][0], ts[max_ind][0]])
    ###         counts = self.decoder.bin_spikes(ts)

    ###     if hasattr(self, 'task_data'):
    ###         self.task_data['spike_counts'] = counts
    ###     
    ###     return counts


    def get_spike_counts(self):
        ts = self.get_spike_ts()
        start_time = self.get_time() #time.time()
        if len(ts)==0:
            counts = np.zeros([len(self.decoder.units), self.n_subbins])
            self.task_data['bin_edges'] = np.array([np.nan, np.nan])
        else:
            min_ind = np.argmin(ts['ts'])
            max_ind = np.argmax(ts['ts'])
            self.task_data['bin_edges'] = np.array([ts[min_ind]['ts'], ts[max_ind]['ts']])
            if self.n_subbins > 1:
                subbin_edges = np.linspace(self.last_get_spike_counts_time, start_time, self.n_subbins+1)
                subbin_inds = np.digitize(ts['arrival_ts'], subbin_edges)
                counts = np.vstack([self.decoder.bin_spikes(ts[subbin_inds == k]) for k in range(1, self.n_subbins+1)]).T
            else:
                counts = self.decoder.bin_spikes(ts).reshape(-1, 1)
        self.task_data['spike_counts'] = counts
        self.task_data['loop_time'] = start_time - self.last_get_spike_counts_time
        self.last_get_spike_counts_time = start_time
        return counts

    def _test_tried_enough(self, ts):
        return self.tries == self.max_attempts

    def _test_not_tried_enough(self, ts):
        return self.tries != self.max_attempts

    def update_cursor(self):
        # Decode the position of the cursor and update display. Function
        # is called every loop iteration
        self.update_target_location()
        self._update(self.call_decoder(self.get_spike_counts())[:3])
        ## if self.bmicount==self.bminum-1:
        ##     self._update(self.call_decoder(self.get_spike_counts())[:3])
        ##     self.bmicount=0
        ## else:
        ##     self.bmicount+=1
        # Save the cursor location to the file
        self.task_data['cursor'] = self.cursor.xfm.move.copy()
        # Write to screen
        self.draw_world()


    @staticmethod
    def sim_target_seq_generator(n_targs, n_trials):
        '''
        Simulated generator for simulations of the BMIControl and CLDAControl tasks
        '''
        center = np.zeros(2)
        pi = np.pi
        targets = 8*np.vstack([[np.cos(pi/4*k), np.sin(pi/4*k)] for k in range(8)])

        target_inds = np.random.randint(0, n_targs, n_trials)
        target_inds[0:n_targs] = np.arange(min(n_targs, n_trials))
        for k in range(n_trials):
            targ = targets[target_inds[k], :]
            yield np.array([[center[0], 0, center[1]],
                            [targ[0], 0, targ[1]]]).T


class CLDAControl(BMIControl):
    '''
    BMI task that periodically refits the decoder parameters based on intended
    movements toward the targets. Inherits directly from BMIControl.
    '''

    batch_time = traits.Float(80.0, desc='The length of the batch in seconds')
    half_life = traits.Float(120.0, desc='Half life of the adaptation in seconds')
    decoder_sequence = traits.String('test', desc='signifier to group together sequences of decoders')

    def create_updater(self):
        clda_input_queue = mp.Queue()
        clda_output_queue = mp.Queue()
        self.updater = clda.KFSmoothbatch(clda_input_queue, clda_output_queue,
            self.batch_time, self.half_life)

    def create_learner(self):
        self.learner = clda.CursorGoalLearner(self.batch_size)
        self.learn_flag = False

    def init(self):
        super(CLDAControl, self).init()
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        self.create_learner()
        self.create_updater()
        # Create the BMI system which combines the decoder, learner, and updater
        self.bmi_system = riglib.bmi.BMISystem(self.decoder, self.learner,
            self.updater)
            
        if isinstance(self.updater, clda.KFRML):
            self.updater.init_suff_stats(self.decoder)

    def update_learn_flag(self):
        # Tell the adaptive BMI when to learn (skip parts of the task where we
        # assume the subject is not trying to move toward the target)
        statelist = ['origin', 'origin_hold', 'terminus', 'terminus_hold']
        if self.state in statelist:
            self.learn_flag = True
        else:
            self.learn_flag = False

    def _rescale_bmi_state(self, decoded_state):
        return 0.1*decoded_state
        
    def call_decoder(self, spike_counts):
        # Get the decoder output & convert from mm to cm
        ds, uf =  self.bmi_system(spike_counts.reshape(-1,1), self.location, #self.target_xz,
            self.state, task_data=self.task_data, assist_level=self.assist_level,
            speed = self.bminum, target_radius=self.terminus_size, learn_flag=self.learn_flag)
        return ds, uf

    def update_cursor(self):
        # Runs every loop
        self.update_target_location()
        self.update_learn_flag()
        decoded_state, update_flag = self.call_decoder(self.get_spike_counts())
        self._update(self.decoder['hand_px', 'hand_py', 'hand_pz'])
        # The update flag is true if the decoder parameters were updated on this
        # iteration. If so, save an update message to the file.
        if update_flag:
            #send msg to hdf file
            self.hdf.sendMsg("update_bmi")
            print "updated params"
        # if self.bmicount==self.bminum-1:
        #     # Get the decoder output
        #     decoded_state, update_flag = self.call_decoder(self.get_spike_counts())
        #     # Remember that decoder is only decoding in 2D, y value is set to 0
        #     #self._update(np.array([decoded_state[0], 0, decoded_state[1]]))
        #     self._update(0.1*self.decoder['hand_px', 'hand_py', 'hand_pz'])
        #     # The update flag is true if the decoder parameters were updated on this
        #     # iteration. If so, save an update message to the file.
        #     if update_flag:
        #         #send msg to hdf file
        #         self.hdf.sendMsg("update_bmi")
        #     self.bmicount=0
        # else:
        #     self.bmicount+=1
        
        
        # Save the cursor location to the file
        self.task_data['cursor'] = self.cursor.xfm.move.copy()
        # Write to screen
        self.draw_world()

    def cleanup(self, database, saveid, **kwargs):
        super(CLDAControl, self).cleanup(database, saveid, **kwargs)
        import tempfile, cPickle, traceback, datetime

        # Open a log file in case of error b/c errors not visible to console
        # at this point
        f = open('/home/helene/Desktop/log', 'w')
        f.write('Opening log file\n')
        
        # save out the parameter history and new decoder unless task was stopped
        # before 1st update
        try:
            f.write('# of paramter updates: %d\n' % len(self.bmi_system.param_hist))
            if len(self.bmi_system.param_hist) > 0:
                f.write('Starting to save parameter hist\n')
                tf = tempfile.NamedTemporaryFile()
                # Get the update history for C and Q matrices and save them
                #C, Q, m, sd, intended_kin, spike_counts = zip(*self.bmi_system.param_hist)
                #np.savez(tf, C=C, Q=Q, mean=m, std=sd, intended_kin=intended_kin, spike_counts=spike_counts)
                pickle.dump(self.bmi_system.param_hist, tf)
                tf.flush()
                # Add the parameter history file to the database entry for this
                # session
                database.save_data(tf.name, "bmi_params", saveid)
                f.write('Finished saving parameter hist\n')

                # Save the final state of the decoder as a new decoder
                tf2 = tempfile.NamedTemporaryFile(delete=False)	
                cPickle.dump(self.decoder, tf2)
                tf2.flush()
                # create suffix for new decoder that has the sequence and the current day
                # and time. This suffix will be appended to the name of the
                # decoder that we started with and saved as a new decoder.
                now = datetime.datetime.now()
                month = str(now.month)
                if len(month)==1:
                    month = '0' + month
                day = str(now.day)
                if len(day)==1:
                    day = '0' + day
                hour = str(now.hour)
                if len(hour)==1:
                    hour = '0' + hour
                minute = str(now.minute)
                if len(minute)==1:
                    minute = '0' + minute
                decoder_name = self.decoder_sequence + month+ day + hour + minute
                database.save_bmi(decoder_name, saveid, tf2.name)
        except:
            traceback.print_exc(file=f)
        f.close()


class CLDAAutoAssist(CLDAControl):

    assist_start = traits.Float(.4, desc="Assist level to start out at")
    assist_end = traits.Float(0, desc="Assist level to end task at")
    assist_time = traits.Float(600, desc="Number of seconds to go from maximum to minimum assist level")

    def __init__(self, *args, **kwargs):
        super(CLDAAutoAssist, self).__init__(*args, **kwargs)
        self.assist_level=self.assist_start
        self.taskstart=time.time()
        self.assist_flag = True
        self.count=0

    def update_cursor(self):
        elapsed_time = time.time()-self.taskstart
        temp = self.assist_start - elapsed_time*(self.assist_start-self.assist_end)/self.assist_time
        if temp < self.assist_end:
            self.assist_level = self.assist_end
            if self.assist_flag:
                print "Assist at minimum"
                self.assist_flag=False
        else:
            self.assist_level = temp
        if self.count%3600==0:
            print "Assist level: ", self.assist_level
        self.count+=1

        super(CLDAAutoAssist, self).update_cursor()

class CLDAConstrainedSSKF(CLDAAutoAssist):
    def create_updater(self):
        clda_input_queue = mp.Queue()
        clda_output_queue = mp.Queue()
        self.updater = clda.KFOrthogonalPlantSmoothbatch(clda_input_queue, clda_output_queue,
            self.batch_time, self.half_life)

class CLDARMLKF(CLDAAutoAssist):
    def create_updater(self):
        self.updater = clda.KFRML(None, None, self.batch_time, self.half_life)

    def init(self):
        self.batch_time = self.decoder.binlen
        super(CLDARMLKF, self).init()


class SimBMIControl(BMIControl):
    def _init_fb_controller(self):
        # Initialize simulation controller
        from riglib.bmi.feedback_controllers import CenterOutCursorGoal
        self.input_device = CenterOutCursorGoal(angular_noise_var=0.13)

    def init(self):
        self._init_neural_encoder()
        self._init_fb_controller()
        self.wait_time = 0
        self.pause = False
        
    def _test_penalty_end(self, ts):
        # No penalty for simulated neurons
        return True

    def _init_neural_encoder(self):
        ## Simulation neural encoder
        from riglib.bmi.sim_neurons import CLDASimCosEnc
        sim_encoder_fname = os.path.join(os.getenv('HOME'), 'code/bmi3d/tests/sim_clda', 'test_ensemble.mat')
        self.encoder = CLDASimCosEnc(fname=sim_encoder_fname, return_ts=True)

    def get_spike_ts(self):
        cursor_pos = self.cursor.xfm.move
        target_pos = self.location
        ctrl    = self.input_device.get(target_pos[[0,2]], cursor_pos[[0,2]])
        ts_data = self.encoder(ctrl)
        return ts_data
        
    ##def get_spike_ts(self):
    ##    if self.decoder.bmicount == 0: #self.decoder.bminum-2:  
    ##        cursor_pos = self.cursor.xfm.move
    ##        target_pos = self.location
    ##        ctrl    = self.input_device.get(target_pos[[0,2]], cursor_pos[[0,2]])
    ##        #ctrl    = self.input_device.get(cursor_pos[[0,2]], target_pos[[0,2]])
    ##        ## print cursor_pos[[0, 2]]
    ##        ## print target_pos[[0, 2]]
    ##        ## print ctrl
    ##        ## print

    ##        ts_data = self.encoder(ctrl)
    ##        return ts_data
    ##    else:
    ##        return np.array([])

class SimBMIControlPPF(SimBMIControl):
    def _init_neural_encoder(self):
        from riglib.bmi import sim_neurons
        sim_encoder_fname = os.path.join(os.getenv('HOME'), 'code/bmi3d/tests/ppf', 'sample_spikes_and_kinematics_10000.mat')
        self.encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning_clda_sim(sim_encoder_fname, dt=0.005) #CosEnc(fname=sim_encoder_fname, return_ts=True)


class SimCLDAControl(SimBMIControl, CLDAControl):
    def init(self):
        '''
        Instantiate simulation decoder
        '''
        SimBMIControl.init(self)

        # Instantiate random seed decoder
        horiz_min, horiz_max = -14., 14.
        vert_min, vert_max = -14., 14.
        
        bounding_box = np.array([horiz_min, vert_min]), np.array([horiz_max, vert_max])
        states_to_bound = ['hand_px', 'hand_pz']

        neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
        stochastic_states = ['hand_vx', 'hand_vz']

        sim_units = self.encoder.get_units()
        self.decoder = riglib.bmi.train._train_KFDecoder_2D_sim(
            stochastic_states, neuron_driving_states, sim_units,
            bounding_box, 
            states_to_bound, include_y=True)
        mm_to_m = 0.001
        m_to_mm = 1000.
        cm_to_m = 0.01
        m_to_cm = 100.
        self.decoder.kf.C *= cm_to_m
        self.decoder.kf.W *= m_to_cm**2

        ## Instantiate BMISystem
        CLDAControl.init(self)
