from tasks import manualcontrolmultitasks
import numpy as np
from riglib.experiment import traits

class KinarmControlMultitask(manualcontrolmultitasks.ManualControlMulti):

    sequence_generators = manualcontrolmultitasks.ManualControlMulti.sequence_generators
    last_pt = np.array([0., 0., 0.])

    def move_effector(self):
        self = kinarm_move_effector(self)

class KinarmControlMultitask_obstacles(KinarmControlMultitask, manualcontrolmultitasks.JoystickMultiObstacles):
    sequence_generators = manualcontrolmultitasks.JoystickMultiObstacles.sequence_generators

class KinarmControlMultitask_catch_obstacles(KinarmControlMultitask_obstacles):
    ''' Task to make obstacles appear on a certain percent of trials '''

    percent_trials_w_obs = traits.Float(.5, desc="Percent of trials where obstacle appears")
    display_obstacle = 0

    def init(self):
        self.add_dtype('display_obstacle', 'f8', (1,))
        super(KinarmControlMultitask_catch_obstacles, self).init()

    def _cycle(self):
        self.task_data['display_obstacle'] = self.display_obstacle
        super(KinarmControlMultitask_catch_obstacles, self)._cycle()

    def _parse_next_trial(self):
        self.targs = self.next_trial[0]
        #Width and height of obstacle

        self.trial_obstacle_list = []
        self.trial_obstacle_loc = []

        obs = self.next_trial[1]
        if len(obs.shape) == 1:
            obs = np.array([obs])

        for io, o in enumerate(obs):
            self.trial_obstacle_list.append(self.obstacle_list[io])

            #Decide whether to display obstacle in the correct location
            tmp = np.random.rand()
            if tmp <= self.percent_trials_w_obs:
                self.display_obstacle = 1
                self.trial_obstacle_loc.append(o)
            else:
                self.display_obstacle = 0
                self.trial_obstacle_loc.append(np.array([-100., 0., -100.]))

        for j in np.arange(io+1, 5):
            o = self.obstacle_list[j]
            o.move_to_position(np.array([-100., 0., -100.]))
            #print 'hiding: obs number and loc: ', 

class KinarmFreeChoice(manualcontrolmultitasks.JoystickSpeedFreeChoice):
    sequence_generators = manualcontrolmultitasks.JoystickSpeedFreeChoice.sequence_generators
    pre_choise_pause_time = traits.Float(3., desc='time before allowed to make a choice')
    status = dict(
        wait = dict(start_trial="targ_transition", stop=None),
        pre_choice_orig = dict(enter_orig='choice_target', timeout='timeout_penalty', stop=None),
        choice_target = dict(enter_choice_target='targ_transition', timeout='timeout_penalty', stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", make_choice='pre_choice_orig'),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )

    def move_effector(self):
        self = kinarm_move_effector(self)

        self.current_pt = self.current_pt + (np.array([np.random.rand()-0.5, 0., np.random.rand()-0.5])*self.joystick_speed)
        self.plant.set_endpoint_pos(self.current_pt)
        self.last_pt = self.current_pt.copy()

    def _test_enter_choice_target(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        enter_targ = 0
        for ic, c in enumerate(self.choice_locs):
            d = np.linalg.norm(cursor_pos - c)
            if d <= self.choice_target_rad: #NOTE, gets in if CENTER of cursor is in target (not entire cursor)
                enter_targ+=1

                #Set chosen as new input: 
                self.chosen_input_ix = ic
                self.joystick_speed = self.input_type_dict[ic]
                print 'trial: ', self.choice_instructed, self.joystick_speed

                #Declare that choice has been made:
                self.choice_made = 1

                #Change color of cursor: 
                sph = self.plant.graphics_models[0]
                sph.color = self.input_type_dict[ic, 'color']
        if ts > self.pre_choise_pause_time:
            return enter_targ > 0
        else:
            return False
            
def kinarm_move_effector(self):
    x = self.kinarmdata.get()
    #Take last point yielded: 
    if np.logical_and(len(x.shape) == 3, x.shape[0] >= 1):
        xgain = -100. # no idea why these values are 100.  Doesn't actually make sense.  Based on calibration, should be around 149.5.  But, this is convenient since it
                        # it keeps everything in the metric system (as does Dexterit-E)
        ygain = 100.
        xoffset = 0. # this used to be -0.05.  Changed the center offset and the window length parameters,and -0.03 became better.  Not sure if consistent.  May need 
                            # to trial and error a couple more times to make sure.
        yoffset = -0.15 # Approximately correct.  Could probably brought down a bit more
        pt = np.array([xgain*(x[-1, 0, 22]+xoffset), 0., ygain*(x[-1, 0, 23]+yoffset)])

        self.current_pt = pt
        self.plant.set_endpoint_pos(self.current_pt)
        self.last_pt = self.current_pt.copy()
    else:
        self.plant.set_endpoint_pos(self.last_pt)
    return self