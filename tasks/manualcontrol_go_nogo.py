import manualcontrolmulti_COtasks
import numpy as np
from riglib.experiment import traits, Sequence
import tasks

class manualcontrol_go_nogo(manualcontrolmulti_COtasks.ManualControlMulti_plusvar):
    reach_time_max = traits.Float(1, desc="Length of time before timeout error duing Go trials")
    periph_hold_time = traits.Float(.2, desc="PHT")
    nogo_hold_time = traits.Float(1., desc="Hold time for No Go trial")

    status = dict(
        wait = dict(start_trial="orig", stop=None),
        orig = dict(enter_orig="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="periph_targ"),
        periph_targ = dict(catch_trial="catch_hold_center", go_trial="go_trial_periph",timeout="timeout_penalty"),
        catch_hold_center = dict(nogo_hold_complete="reward",leave_early="hold_penalty"),
        go_trial_periph = dict(enter_periph="go_trial_hold_periph", go_trial_timeout="timeout_penalty"),
        go_trial_hold_periph = dict(per_hold_complete="reward", leave_early = "hold_penalty"),
        timeout_penalty = dict(timeout_penalty_end="wait"),
        hold_penalty = dict(hold_penalty_end="wait"),
        reward = dict(reward_end="wait")
    )

    sequence_generators = ['go_no_go']
    
    def _parse_next_trial(self):
        t = self.next_trial
        self.targs = np.vstack((t['orig'], t['periph']))
        self.catch = t['catch']
        self.mc_targ_orig = t['orig']
        self.mc_targ_periph = t['periph']
        self.cnt = 0
        self.t_inf = t

    def _start_orig(self):
        target = self.targets[0]
        self.target_location = self.mc_targ_orig
        target.move_to_position(self.target_location)
        target.cue_trial_start()

    def _start_hold(self):
        super(manualcontrol_go_nogo, self)._start_hold()
        target = self.targets[1]
        target.move_to_position(self.mc_targ_periph)
        target.cue_trial_start()

    def _start_periph_targ(self):
        # target = self.targets[1]
        # self.target_location = self.mc_targ_periph
        # target.move_to_position(self.target_location)
        # target.cue_trial_start()   
        self.target_location = self.mc_targ_periph
        target = self.targets[0]
        target.cue_trial_end_success() 
        target2 = self.targets[1]
        target2.cue_trial_start()

    def _start_catch_hold_center(self):
        target = self.targets[1]
        #Set target_location: 
        self.target_location = self.mc_targ_orig
        target.turn_yellow()
        self.targ_rew_ind = 0

    def _start_go_trial_hold_periph(self):
        self.targ_rew_ind = 1

    def _start_reward(self):
        target = self.targets[self.targ_rew_ind]
        target.cue_trial_end_success()
        super(manualcontrol_go_nogo, self)._start_reward()


    def _cycle(self):
        super(manualcontrol_go_nogo, self)._cycle()
        self.cnt += 1

    #### Test Functions ###
    def _test_enter_orig(self,ts):
        cursor_pos = self.plant.get_endpoint_pos()

        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius - self.cursor_radius)
        
    def _test_catch_trial(self, ts):
        #catch=False
        #if self.catch>0 and ts > self.catch_time:
        #    catch = True
        #return catch
        return self.catch > 0

    def _test_per_hold_complete(self, ts):
        return ts > self.periph_hold_time

    def _test_nogo_hold_complete(self, ts):
        return ts > self.nogo_hold_time

    def _test_go_trial(self, ts):
        return self.catch == 0

    def _test_enter_periph(self,ts):
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius - self.cursor_radius)

    def _test_go_trial_timeout(self, ts):
        return ts>self.reach_time_max
        
    @staticmethod
    def go_no_go(nblocks=100, ntargets=4, boundaries=(-18,18,-12,12),
        distance=10,perc_no_go=20):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin.

        Parameters
        ----------
        nblocks: int
            number of blocks (total num of targets = nblock * ntargets)
        ntargets : int
            The number of target pairs per  block
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between the targets in a pair.
        perc_catch: float
            percentage of 'no_go' trials
        no_go_time_stats: 2 element Tuple
            mean and var of time to cue 'no_go' trials


        Returns
        -------
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        num_catch = len(theta)*perc_no_go/100.
        ind_catch_trials = np.arange(len(theta))
        np.random.shuffle(ind_catch_trials)
        ind_catch_trials = ind_catch_trials[:num_catch]

        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T

        catch = np.zeros((len(theta), ))
        catch[ind_catch_trials] = 1
        #catch[ind_catch_trials, 1] = no_go_time_stats[0] + np.random.uniform(low=-1,high=1)*no_go_time_stats[1]


        it = iter([dict(orig=pairs[i,0,:], periph=pairs[i,1,:], catch=catch[i]) for i in range(theta.shape[0])])

        return it

class manualcontrol_go_nogo_plus_gocatch(manualcontrol_go_nogo):
    status = dict(
        wait = dict(start_trial="orig", stop=None),
        orig = dict(enter_orig="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="periph_targ"),
        periph_targ = dict(catch_trial="catch_hold_center", go_trial="go_trial_periph",timeout="timeout_penalty"),
        catch_hold_center = dict(nogo_hold_complete="reward",leave_early="hold_penalty"),
        go_trial_periph = dict(enter_periph="go_trial_hold_periph", go_catch = "catch_hold_center", go_trial_timeout="timeout_penalty"),
        go_trial_hold_periph = dict(per_hold_complete="reward", leave_early = "hold_penalty"),
        timeout_penalty = dict(timeout_penalty_end="orig"),
        hold_penalty = dict(hold_penalty_end="wait"),
        reward = dict(reward_end="wait")
    )
    trig_go_catch_mean = traits.Float(1., desc="Mean of time before go --> nogo")
    trig_go_catch_std = traits.Float(1., desc= "Std. Dev. of time before go --> nogo")
    gocatch_targ_rad = traits.Float(4., desc= "Virtual Target radius for go catch trials")
    sequence_generators = ['go_nogo_plus_gocatch']

    def _parse_next_trial(self):
        super(manualcontrol_go_nogo_plus_gocatch, self)._parse_next_trial()
        self.gocatch = self.t_inf['gocatch']
        if self.gocatch:
            self.trig_go_catch = self.trig_go_catch_mean + np.random.uniform(low=-1,high=1)*self.trig_go_catch_std
            self.trig_go_catch = np.min([self.trig_go_catch, self.reach_time_max])

    def _test_go_catch(self, ts):
        self.go_catch=False
        if ((self.gocatch > 0) and (ts > self.trig_go_catch)):
            self.go_catch=True
        return self.go_catch

    def _test_leave_early(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        if self.gocatch:
            rad = self.gocatch_targ_rad - self.cursor_radius
        else:
            rad = self.target_radius - self.cursor_radius
        return d > rad

    @staticmethod
    def go_nogo_plus_gocatch(nblocks=100, ntargets=4, boundaries=(-18,18,-12,12),
        distance=10,perc_no_go=20, perc_go_catch=20):

        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        num_catch = len(theta)*perc_no_go/100.
        ind_catch_trials = np.arange(len(theta))
        np.random.shuffle(ind_catch_trials)
        ind_catch_trials = ind_catch_trials[:num_catch]

        num_gocatch = len(theta)*(100-perc_no_go)/100.*perc_go_catch/100.
        ind = np.array([True]*len(theta))
        ind[ind_catch_trials] = False
        ind2 = np.nonzero(ind)[0]
        np.random.shuffle(ind2)
        ind_gocatch_trials = ind2[:num_gocatch]

        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T

        catch = np.zeros((len(theta), ))
        catch[ind_catch_trials] = 1

        gocatch = np.zeros((len(theta), ))
        gocatch[ind_gocatch_trials] = 1
        #catch[ind_catch_trials, 1] = no_go_time_stats[0] + np.random.uniform(low=-1,high=1)*no_go_time_stats[1]

        it = iter([dict(orig=pairs[i,0,:], periph=pairs[i,1,:], catch=catch[i], gocatch=gocatch[i]) for i in range(theta.shape[0])])

        return it
