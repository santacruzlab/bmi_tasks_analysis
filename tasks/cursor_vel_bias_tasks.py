'''
Cursor tasks where a velocity bias may be added in a particular direction
'''
from bmimultitasks import BMIControlMulti
from riglib.experiment import traits, experiment

class BMICursorBias(BMIControlMulti):
    status = dict(
        wait = dict(start_trial="premove", stop=None),
        premove=dict(premove_complete="target"),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", trial_restart="premove"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )

    premove_time = traits.Float(0.5, desc='length of frozen state')

    def _while_premove(self):
        reset_pos = self.init_decoder_mean # np.mat(np.hstack([np.zeros(3), np.zeros(3), 1]).reshape(-1,1))
        # reset_pos = np.mat(np.hstack([np.zeros(3), np.zeros(3), 1]).reshape(-1,1))
        self.decoder.filt.state.mean = reset_pos

    def _end_timeout_penalty(self):
        pass    

    def _test_premove_complete(self, ts):
        return ts>=self.premove_time

    def _test_hold_complete(self,ts):
        if self.target_index==0:
            return True
        else:
            return ts>=self.hold_time

    def _test_trial_incomplete(self, ts):
        return (self.target_index<self.chain_length-1) and (self.target_index != -1) and (self.tries<self.max_attempts)

    def _test_trial_restart(self, ts):
        return (self.target_index==-1) and (self.tries<self.max_attempts)

class BMICursorBiasCatch(BMICursorBias):

    catch_interval = traits.Int(10, desc='Number of trials between bias catch trials')
    bias_magnitude = traits.Float(1,desc='Strength of velocity bias in cm/sec')
    catch_flag = False
    catch_count = 0
    bias = 0.0

    def init(self):
        self.add_dtype('bias', 'f8', 1)
        super(BMICursorBiasCatch, self).init()
        

    def _start_wait(self):
        if self.catch_count==self.catch_interval:
            self.catch_flag = True
            self.catch_count = 0
            if np.random.rand(1)>.5:
                self.bias = -1*self.bias_magnitude
            else:
                self.bias = self.bias_magnitude
            print "bias", self.bias
        else:
            self.catch_flag = False
            self.catch_count+=1
        super(BMICursorBias, self)._start_wait()

    def _cycle(self):
        if self.catch_flag and self.state == 'target':
            self.decoder.filt.state.mean[3,0] += self.bias/6

        if self.catch_flag:
            self.task_data['bias'] = self.bias
        else:
            self.task_data['bias'] = 0.0
        super(BMICursorBiasCatch, self)._cycle()
