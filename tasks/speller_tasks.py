'''

'''
from bmimultitasks import BMIControlMulti
from passivetasks import EndPostureFeedbackController, MachineOnlyFilter
from manualcontrolmultitasks import VirtualRectangularTarget, VirtualCircularTarget
from riglib.bmi.state_space_models import LinearVelocityStateSpace, State
from riglib.plants import CursorPlant
from riglib.stereo_opengl.window import Window, WindowDispl2D
from riglib.experiment import traits
import numpy as np

from riglib.bmi.bmi import BMILoop, Decoder, GaussianState
from riglib.experiment import traits, Sequence
from riglib.bmi.state_space_models import offset_state
from riglib.bmi import goal_calculators
from riglib.bmi.extractor import DummyExtractor
from riglib.bmi.assist import SSMLFCAssister
from riglib.bmi import extractor
from riglib.bmi import clda, feedback_controllers
from riglib.bmi.assist import FeedbackControllerAssist

from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife
from riglib import plants
import pygame

from plantlist import plantlist



#########################
#### BMI sub-classes ####
#########################
class PointClickSSM(LinearVelocityStateSpace):
    def __init__(self, *args, **kwargs):
        states = [
            State('hand_px', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
            State('hand_pz', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            State('click_p', stochastic=False, drives_obs=False, min_val=0., max_val=1., order=0),
            State('hand_vx', stochastic=True,  drives_obs=True, order=1),
            State('hand_vz', stochastic=True,  drives_obs=True, order=1),
            State('click_v', stochastic=True,  drives_obs=True, order=1),
            offset_state
        ]
        super(PointClickSSM, self).__init__(states, *args, **kwargs)

class MouseFBController(feedback_controllers.LQRController):
    def __init__(self, **kwargs):
        ssm = PointClickSSM()
        A, B, _ = ssm.get_ssm_matrices()
        pos_inds, = np.nonzero(ssm.state_order == 0)
        Q = np.mat(np.diag(np.hstack([1, 1, 500., np.zeros(len(pos_inds)), 0])))
        R = 10000*np.mat(np.eye(B.shape[1]))
        super(MouseFBController, self).__init__(A, B, Q, R, **kwargs)        

mouse_motion_model = MouseFBController()

_red = np.array([1., 0., 0, 1])
_green = np.array([0, 1., 0, 1])
_blue = np.array([0, 0., 1., 1])

class Mouse(CursorPlant):
    click_ = 0
    click_threshold = 0.8
    click_raw = 0
    hdf_attrs = [('cursor', 'f8', (3,)), ('mouse_state', 'f8', (3,))]
    def __init__(self, *args, **kwargs):
        super(Mouse, self).__init__(*args, **kwargs)
        self.click_posedge = False
        self.click_negedge = False

    def draw(self):
        # cursor should be blue when not clicked and red when fully clicked
        # print self.click_
        self.cursor.color = _red*self.click_ + _blue*(1-self.click_)
        self.cursor.translate(*self.position, reset=True)

    def set_intrinsic_coordinates(self, pt):
        self.pt = pt
        if pt[-1] < 0:
            pt[-1] = 0
        elif pt[-1] > 1:
            pt[-1] = 1

        self.position = np.array([pt[0], 0., pt[1]])
        # bound the click state between 0 and 1
        self.click_raw = max(min(pt[2], 1), 0)
        new_click_ = float(self.click_raw > self.click_threshold)
        self.click_posedge = not self.click_ and new_click_
        self.click_negedge = self.click_ and not new_click_
        self.click_ = new_click_

        self.draw()

    def get_intrinsic_coordinates(self):
        return np.hstack([self.position[[0,2]], self.click_raw])       

    def get_data_to_save(self):
        mouse_state = np.hstack([self.position[[0,2]], self.click_raw])
        return dict(cursor=self.position, mouse_state=mouse_state)

mouse = Mouse(endpt_bounds=(-20, 20, -14, 14, 0, 1))
speller_plantlist = dict(mouse=mouse)

class MouseFakeKF(MachineOnlyFilter):
    '''
    KF-like infterface for the computer's actual mouse
    '''
    win_res = (1000, 560)
    def _forward_infer(self, *args, **kwargs):
        pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)
        pos[0] = 50*(pos[0]/self.win_res[0] - 0.5)
        pos[1] = 28*-(pos[1]/self.win_res[1] - 0.5)
        click_state = pygame.mouse.get_pressed()[0] # left click
        pos_state = np.hstack([pos, click_state])
        xt = np.hstack([pos_state, np.zeros_like(pos_state), 1])
        cov = np.zeros([len(xt), len(xt)])
        mean = xt.reshape(-1,1)
        return GaussianState(mean, cov)

class Key(object):
    '''
    Container for information for a single key on a keyboard
    '''
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

keyboard_spec = dict(
    q=Key(default_key='q', pos=np.array([-15.0, 0.0, 4.5])),
    w=Key(default_key='w', pos=np.array([-12.0, 0.0, 4.5])),
    e=Key(default_key='e', pos=np.array([-9.0, 0.0, 4.5])),
    r=Key(default_key='r', pos=np.array([-6.0, 0.0, 4.5])),
    t=Key(default_key='t', pos=np.array([-3.0, 0.0, 4.5])),
    y=Key(default_key='y', pos=np.array([0.0, 0.0, 4.5])),
    u=Key(default_key='u', pos=np.array([3.0, 0.0, 4.5])),
    i=Key(default_key='i', pos=np.array([6.0, 0.0, 4.5])),
    o=Key(default_key='o', pos=np.array([9.0, 0.0, 4.5])),
    p=Key(default_key='p', pos=np.array([12.0, 0.0, 4.5])),

    a=Key(default_key='a', pos=np.array([-12.0, 0.0, 1.5])),
    s=Key(default_key='s', pos=np.array([-9.0, 0.0, 1.5])),
    d=Key(default_key='d', pos=np.array([-6.0, 0.0, 1.5])),
    f=Key(default_key='f', pos=np.array([-3.0, 0.0, 1.5])),
    g=Key(default_key='g', pos=np.array([0.0, 0.0, 1.5])),
    h=Key(default_key='h', pos=np.array([3.0, 0.0, 1.5])),
    j=Key(default_key='j', pos=np.array([6.0, 0.0, 1.5])),
    k=Key(default_key='k', pos=np.array([9.0, 0.0, 1.5])),
    l=Key(default_key='l', pos=np.array([12.0, 0.0, 1.5])),

    z=Key(default_key='z', pos=np.array([-9.0, 0.0, -1.5])),
    x=Key(default_key='x', pos=np.array([-6.0, 0.0, -1.5])),
    c=Key(default_key='c', pos=np.array([-3.0, 0.0, -1.5])),
    v=Key(default_key='v', pos=np.array([0.0, 0.0, -1.5])),
    b=Key(default_key='b', pos=np.array([3.0, 0.0, -1.5])),
    n=Key(default_key='n', pos=np.array([6.0, 0.0, -1.5])),
    m=Key(default_key='m', pos=np.array([9.0, 0.0, -1.5])),

    backspace=Key(default_key=' ', pos=np.array([15, 0.0, -1.5])),    
    shift=Key(default_key=' ', pos=np.array([-15, 0.0, -1.5])),

    space=Key(default_key=' ', pos=np.array([0.0, 0.0, -4.5])),
    enter=Key(default_key=' ', pos=np.array([15, 0.0, -4.5])),
)

class KeyTarget(VirtualCircularTarget):
    def click(self):
        self.position[1] = 5
        self.sphere.translate(*self.position, reset=True)
        # self.draw()

    def unclick(self):
        self.position[1] = 2
        self.sphere.translate(*self.position, reset=True)
        # print "unclick"
        # self.draw()

###################
###### Task classes
###################
class MouseSpeller(LinearlyDecreasingAssist, BMILoop, Sequence, Window):
    ## Traits: runtime-configurable parameters
    reward_time = traits.Float(.5, desc="Length of juice reward")
    target_radius = traits.Float(1.3, desc="Radius of targets in cm")
    window_size = traits.Tuple((1920*2, 1080), descr='window size')
    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')
    max_error_penalty = traits.Float(3, desc='Max number of penalties (unrewarded backspaces) for false positive clicks')

    background = (0.5, 0.5, 0.5, 1)

    state = 'wait'

    sequence_generators = ['rand_key_seq_gen', 'mackenzie_soukoreff_corpus']

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None, false_click="target"),
        hold = dict(hold_complete="targ_transition", timeout="timeout_penalty", leave_early="target"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    trial_end_states = ['reward', 'timeout_penalty', 'hold_penalty']

    def __init__(self, *args, **kwargs):
        kwargs['instantiate_targets'] = False
        self.plant = mouse #plants.CursorPlant() #MouseFakeKF() #plantlist[self.plant_type]
        super(MouseSpeller, self).__init__(*args, **kwargs)

        ## instantiate the keyboard targets
        n_target_per_row = [5, 10, 9, 9, 6]
        self.targets = dict()
        for key in keyboard_spec:
            targ = KeyTarget(target_radius=self.target_radius, target_color=np.array([0.25, 0.25, 0.25, 1]))
            # targ = VirtualRectangularTarget(target_width=self.target_radius*2, target_color=np.array([0.25, 0.25, 0.25, 1]))
            self.targets[key] = targ
            
            p = keyboard_spec[key].pos
            p[1] += 2
            targ.move_to_position(p)

            for model in targ.graphics_models:
                self.add_model(model)

        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        self.current_target = None

        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

        self.text_output = ''

    def init(self):
        self.add_dtype('target', 'f8', (3,))
        self.add_dtype('target_index', 'i', (1,))        
        super(MouseSpeller, self).init()

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        
        for key, targ in self.targets.items():
            in_targ = targ.pt_inside(self.plant.get_endpoint_pos())
            if in_targ and self.plant.click_:
                targ.click()
            else:
                targ.unclick()

        if not (self.current_target is None):
            self.task_data['target'] = self.current_target.get_position()
        else:
            self.task_data['target'] = np.ones(3)*np.nan

        self.task_data['target_index'] = self.target_index

        self.move_plant()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(MouseSpeller, self)._cycle()

    def move_plant(self):
        super(MouseSpeller, self).move_plant()

    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        ent_targ = self.current_target.pt_inside(self.plant.get_endpoint_pos())
        clicked = self.plant.click_
        return ent_targ and not clicked
        
    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        outside = not self.current_target.pt_inside(self.plant.get_endpoint_pos())
        if outside:
            self.target_index -= 1
        return outside

    def _record_char(self, new_char):
        new_char = self.targs[self.target_index]
        if new_char == 'space': new_char = ' '
        if new_char == 'enter': new_char = '\n'
        if new_char == 'shift': new_char = ''
        if new_char == 'backspace' and len(self.text_output) > 0:
            self.text_output = self.text_output[:-1]
        elif new_char == 'backspace':
            pass
        else:
            self.text_output += new_char


    def _test_hold_complete(self, ts):
        hold_complete = self.plant.click_ and self.current_target.pt_inside(self.plant.get_endpoint_pos())
        if hold_complete:
            self._record_char(self.targs[self.target_index])
        return hold_complete

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super(MouseSpeller, self).update_report_stats()
        self.reportstats['Text output'] = self.text_output

    def _test_timeout(self, ts):
        return ts>self.timeout_time

    def _test_timeout_penalty_end(self, ts):
        return ts>self.timeout_penalty_time

    def _test_hold_penalty_end(self, ts):
        return ts>self.hold_penalty_time

    def _test_trial_complete(self, ts):
        return self.target_index == len(self.targs) - 1

    def _test_trial_incomplete(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts)

    def _test_trial_abort(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries==self.max_attempts)

    def _test_reward_end(self, ts):
        return ts>self.reward_time

    def _test_false_click(self, ts):
        if self.plant.click_posedge:
            for key, targ in self.targets.items():
                if targ == self.current_target:
                    return False
                in_targ = targ.pt_inside(self.plant.get_endpoint_pos())
                if in_targ:
                    # add backspace as a target
                    self.n_false_clicks_this_trial  += 1
                    if self.n_false_clicks_this_trial < self.max_error_penalty:
                        self.targs.insert(self.target_index, 'backspace')
                    
                    self.target_index -= 1
                    self.current_target.reset()
                    self._record_char(key)
                    return True
            # if the click was outside any of the targets
            return False
        else:
            return False


    #### STATE FUNCTIONS ####
    def _parse_next_trial(self):
        self.targs = list(self.next_trial)

    def _start_wait(self):
        super(MouseSpeller, self)._start_wait()
        self.n_false_clicks_this_trial = 0
        self.tries = 0
        self.target_index = -1
        #hide targets
        for key in self.targets:
            self.targets[key].reset()

        #get target locations for this trial
        self._parse_next_trial()
        # self.chain_length = len(self.targs)

    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.targs[self.target_index]]
        self.current_target = target
        # self.target_location = target.pos
        target.cue_trial_start()

    def _end_hold(self):
        # change current target color to green
        self.current_target.reset()

    def _start_hold_penalty(self):
        #hide targets
        for key in self.targets:
            self.targets[key].reset()

        self.tries += 1
        self.target_index = -1
    
    def _start_timeout_penalty(self):
        #hide targets
        for key in self.targets:
            self.targets[key].reset()

        self.tries += 1
        self.target_index = -1

    def _start_targ_transition(self):
        self.current_target.reset()

    def _start_reward(self):
        super(MouseSpeller, self)._start_reward()
        self.current_target.cue_trial_end_success()

    @staticmethod
    def rand_key_seq_gen(length=1000, seq_len=2):
        key_sequences = []
        keys = keyboard_spec.keys()
        n_keys = len(keys)

        for k in range(length):
            inds = np.random.randint(0, n_keys, seq_len)
            key_sequences.append([keys[m] for m in inds])

        return key_sequences

    @staticmethod
    def mackenzie_soukoreff_corpus(length=1000):
        trials = []
        fh = open('/storage/task_data/MouseSpeller/phrases2.txt')

        missing_chars = []
        for line in fh:
            for char in line:
                if char == ' ':

                    char = 'space'
                elif char in ['\n', '\r']:
                    char = 'enter'

                if char in keyboard_spec:
                    trials.append([char])
                elif chr(ord(char) + (ord('a') - ord('A'))) in keyboard_spec:
                    char = chr(ord(char) + (ord('a') - ord('A')))
                    trials.append(['shift', char])
                elif char not in keyboard_spec:
                    missing_chars.append(char)
        print "missing_chars", missing_chars
        return trials

    def create_assister(self):
        self.assister = FeedbackControllerAssist(mouse_motion_model, style='mixing')

    def create_goal_calculator(self):
        self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        if self.current_target is None:
            target_pos = np.zeros(2)
        else:
            target_pos = self.current_target.get_position()[[0,2]]
        opt_click_state = float(self.state == 'hold')
        target_pos_state = np.hstack([target_pos, opt_click_state])
        data, solution_updated = self.goal_calculator(target_pos_state)
        target_state, error = data
        return np.tile(np.array(target_state).reshape(-1,1), [1, self.decoder.n_subbins])

class MouseSpellerMouseInput(MouseSpeller):
    '''
    Testing version of the MouseSpeller where the "decoder" is really just the masked mouse input
    '''
    def load_decoder(self):
        from db.namelist import bmi_state_space_models
        from config import config
        self.ssm = PointClickSSM()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MouseFakeKF(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()

class VisualFeedbackMouseSpeller(MouseSpeller):
    is_bmi_seed = True
    assist_level = (0.5, 0.5)

    def load_decoder(self):
        from db.namelist import bmi_state_space_models
        from config import config
        self.ssm = PointClickSSM()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()

class CLDAMouseSpeller(MouseSpeller, LinearlyDecreasingHalfLife):
    '''
    Adapt mouse decoders
    '''
    batch_time = traits.Float(0.1, desc='The length of the batch in seconds')
    decoder_sequence = 'mouse' # prefix to add onto the name of any decoders trained in this block

    def create_learner(self):
        self.batch_size = int(self.batch_time/self.decoder.binlen)
        self.learner = clda.FeedbackControllerLearner(self.batch_size, mouse_motion_model)
        self.learn_flag = True

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def call_decoder(self, *args, **kwargs):
        kwargs['half_life'] = self.current_half_life
        return super(CLDAMouseSpeller, self).call_decoder(*args, **kwargs)

class CLDAMouseSpellerNLStats(CLDAMouseSpeller):
    pass

#################################
#### Simulation task classes ####
#################################
class SimMouseSpeller(MouseSpeller):
    assist_level = (0, 0)
    def _init_fb_controller(self):
        # Initialize simulation controller
        self.input_device = mouse_motion_model

    def _init_neural_encoder(self):
        ## Simulation neural encoder
        from riglib.bmi.sim_neurons import KalmanEncoder
        n_features = 20
        self.encoder = KalmanEncoder(PointClickSSM(), n_features)

    def load_decoder(self):
        '''
        Instantiate the neural encoder and "train" the decoder
        '''
        from riglib.bmi import train

        encoder = self.encoder
        n_samples = 20000
        units = self.encoder.get_units()
        n_units = len(units)

        # draw samples from the W distribution
        ssm = PointClickSSM()
        A, _, W = ssm.get_ssm_matrices()
        mean = np.zeros(A.shape[0])
        mean[-1] = 1
        state_samples = np.random.multivariate_normal(mean, 100*W, n_samples)

        spike_counts = np.zeros([n_units, n_samples])
        self.encoder.call_ds_rate = 1
        for k in range(n_samples):
            spike_counts[:,k] = np.array(self.encoder(state_samples[k])).ravel()

        kin = state_samples.T

        self.decoder = train.train_KFDecoder_abstract(ssm, kin, spike_counts, units, 0.1)
        self.encoder.call_ds_rate = 6

    def init(self):
        self._init_neural_encoder()
        self._init_fb_controller()
        self.wait_time = 0
        self.pause = False
        super(SimMouseSpeller, self).init()
        
    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.SimDirectObsExtractor(self.input_device, self.encoder, 
            n_subbins=self.decoder.n_subbins, units=self.decoder.units, task=self)
        self._add_feature_extractor_dtype()

    def create_feature_accumulator(self):
        '''
        Instantiate the feature accumulator used to implement rate matching between the Decoder and the task,
        e.g. using a 10 Hz KFDecoder in a 60 Hz task
        '''
        from riglib.bmi import accumulator
        feature_shape = [self.decoder.n_features, 1]
        feature_dtype = np.float64
        acc_len = int(self.decoder.binlen / self.update_rate)
        acc_len = max(1, acc_len)

        self.feature_accumulator = accumulator.NullAccumulator(acc_len)

class SimCLDAMouseSpeller(CLDAMouseSpeller, SimMouseSpeller):
    def load_decoder(self):
        '''
        Instantiate the neural encoder and "train" the decoder
        '''
        from riglib.bmi import train

        encoder = self.encoder
        n_samples = 20000
        units = self.encoder.get_units()
        n_units = len(units)

        # draw samples from the W distribution
        ssm = PointClickSSM()
        A, _, W = ssm.get_ssm_matrices()
        mean = np.zeros(A.shape[0])
        mean[-1] = 1
        state_samples = np.random.multivariate_normal(mean, 100*W, n_samples)

        spike_counts = np.zeros([n_units, n_samples])
        self.encoder.call_ds_rate = 1
        for k in range(n_samples):
            spike_counts[:,k] = np.array(self.encoder(state_samples[k])).ravel()

        unit_inds = np.arange(n_units)
        np.random.shuffle(unit_inds)
        spike_counts = spike_counts[unit_inds, :]

        kin = state_samples.T

        self.decoder = train.train_KFDecoder_abstract(ssm, kin, spike_counts, units, 0.1)
        self.encoder.call_ds_rate = 6

    def init(self):
        self._init_neural_encoder()
        self._init_fb_controller()
        self.wait_time = 0
        self.pause = False
        encoder_C = self.encoder.C.copy()
        super(SimCLDAMouseSpeller, self).init()

        self.encoder.C = encoder_C