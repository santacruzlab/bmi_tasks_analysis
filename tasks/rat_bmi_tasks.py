import bmimultitasks 
from riglib.experiment import LogExperiment
from riglib.bmi.bmi import BMILoop
from riglib.experiment import traits, Sequence
import numpy as np
from plantlist import plantlist

class RatBaseline(Sequence):
    
    session_length = 15.*60.
    is_bmi_seed = True
    sequence_generators = ['rand_rew_sched']
    next_rew_time = 5.
    reward_time = 0.3
    status= dict(
        wait = dict(start_trial='collect', stop=None),
        collect= dict(rand_rew='reward', stop=None),
        reward = dict(reward_end='wait')
        )

    def _parse_next_trial(self):
        self.next_rew_time = self.next_trial
        print self.next_rew_time


    def _test_rand_rew(self, ts):
        return ts > self.next_rew_time

    def init(self):
        self.add_dtype('cursor', 'f8', (1,))
        self.add_dtype('freq', 'f8', (1,))
        super(RatBaseline, self).init()

    def _start_reward(self):
        pass

    @staticmethod
    def rand_rew_sched(mean_time=10, std_time=3):
        return np.random.normal(mean_time, std_time, (1000,))

class RatBMI(BMILoop, LogExperiment):
    status = dict(
        wait = dict(start_trial='feedback_on', stop=None),
        feedback_on = dict(baseline_hit='periph_targets', stop=None),
        periph_targets = dict(target_hit='check_reward', timeout='noise_burst', stop=None),
        check_reward = dict(rewarded_target='reward', unrewarded_target='feedback_pause'),
        feedback_pause = dict(end_feedback_pause='wait'),
        reward = dict(reward_end='wait'),
        noise_burst = dict(noise_burst_end='noise_burst_timeout'),
        noise_burst_timeout = dict(noise_burst_timeout_end='wait')
        )

    #Flag for feedback on or not
    feedback = False
    prev_targ_hit = 't1'
    timeout_time = traits.Float(30.)
    noise_burst_time = traits.Float(3.)
    noise_burst_timeout_time = traits.Float(1.)
    reward_time = traits.Float(1., desc='reward time')
    #Frequency range: 
    aud_freq_range = traits.Tuple((1000., 20000.))
    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())

    #Time to average over: 
    nsteps = traits.Float(10.)
    feedback_pause = traits.Float(3.)

    def __init__(self, *args, **kwargs):
        super(RatBMI, self).__init__(*args, **kwargs)

        if hasattr(self, 'decoder'):
            print self.decoder
        else:
            self.decoder = kwargs['decoder']
        dec_params = dict(nsteps=self.nsteps, freq_lim=self.aud_freq_range)
        for k, (key, val) in enumerate(dec_params.items()):
            print key, val, self.decoder.filt.dec_params[key]
            assert self.decoder.filt.dec_params[key] == val
        self.decoder.filt.init_from_task(self.decoder.n_units, **dec_params)
        self.plant = plantlist[self.plant_type]


    def init(self, *args, **kwargs):
        self.add_dtype('cursor', 'f8', (2,))
        self.add_dtype('freq', 'f8', (2,))        
        super(RatBMI, self).init()
        self.decoder.count_max = self.feature_accumulator.count_max


    def _cycle(self):
        self.rat_cursor = self.decoder.filt.get_mean()
        self.task_data['cursor'] = self.rat_cursor
        self.task_data['freq'] = self.decoder.filt.F
        self.decoder.cnt = self.feature_accumulator.count
        self.decoder.feedback = self.feedback
        super(RatBMI, self)._cycle()

    # def move_plant(self):
    #     if self.feature_accumulator.count == self.feature_accumulator.count_max:
    #         print 'self.plant.drive from task.py'
    #         self.plant.drive(self.decoder)
    def _start_wait(self):
        return True

    def _test_start_trial(self, ts):
        return True

    def _test_rewarded_target(self, ts):
        if self.prev_targ_hit == 't1':
            return False
        elif self.prev_targ_hit == 't2':
            return True

    def _test_unrewarded_target(self, ts):
        if self.prev_targ_hit == 't1':
            return True
        elif self.prev_targ_hit == 't2':
            return False

    def _start_feedback_pause(self):
        self.feedback = False

    def _test_end_feedback_pause(self, ts):
        return ts > self.feedback_pause

    def _start_reward(self):
        print 'reward!'

    def _start_feedback_on(self):
        self.feedback = True

    def _test_baseline_hit(self, ts):
        if self.prev_targ_hit == 't1':
            #Must go below baseline:
            return self.rat_cursor <= self.decoder.filt.mid
        elif self.prev_targ_hit == 't2':
            #Must rise above baseline:
            return self.rat_cursor >= self.decoder.filt.mid
        else:
            return False

    def _test_target_hit(self, ts):
        if self.rat_cursor >= self.decoder.filt.t1:
            self.prev_targ_hit = 't1'
            self.feedback = False
            return True
        elif self.rat_cursor <= self.decoder.filt.t2:
            self.prev_targ_hit = 't2'
            self.feedback = False
            return True
        else:
            return False

    def _test_timeout(self, ts):
        return ts > self.timeout_time

    def _test_noise_burst_end(self, ts):
        return ts > self.noise_burst_time

    def _test_noise_burst_timeout_end(self, ts):
        return ts > self.noise_burst_timeout_time

    def _start_noise_burst(self):
        self.feedback = False
        self.plant.play_white_noise()

    def move_plant(self):
        super(RatBMI, self).move_plant()

    def get_current_assist_level(self):
        return 0.
