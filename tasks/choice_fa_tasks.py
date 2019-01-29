from factor_analysis_tasks import FactorBMIBase
import target_graphics
#from manualcontrolfreechoice import target_colors
from riglib.bmi.goal_calculators import GoalCalculator
from riglib.experiment import traits
import numpy as np


target_colors = {
"yellow": (1,1,0,0.75),
"magenta": (1,0,1,0.75),
"purple":(0.608,0.188,1,0.75),
"dodgerblue": (0.118,0.565,1,0.75),
"teal":(0,0.502,0.502,0.75),
"olive":(0.420,0.557,0.137,.75),
"juicyorange": (1,0.502,0.,0.75),
"hotpink":(1,0.0,0.606,.75),
"lightwood": (0.627,0.322,0.176,0.75),
"elephant":(0.409,0.409,0.409,0.5),
"green":(0., 1., 0., 0.5)}


class Choice_Goal_Calc(GoalCalculator):
    def __init__(self, ssm):
        self.ssm = ssm
        self.last_pos = np.zeros((3, ))

    def __call__(self, target_pos, choice_pos, choice_ix, target_ix, state):
        if state == 'target':
            pos = target_pos[target_ix,:]
            self.last_pos = pos
        
        elif state == 'choice_target':
            pos = choice_pos[choice_ix, :]
            self.last_pos = pos
        elif state == 'pre_choice_orig':
            pos = np.array([0., 0., 0.])
        else:
            pos = self.last_pos

        vel = np.zeros_like(pos)
        return np.hstack([pos, vel, 1]).reshape(-1, 1)


class FreeChoiceFA(FactorBMIBase):
    '''
    Task where the virtual plant starts in configuration sampled from a discrete set and resets every trial
    '''

    sequence_generators = ['centerout_2D_discrete_w_free_choice', 'centerout_2D_discrete_w_free_choice_v2',
        'centerout_2D_discrete_w_free_choices_evenly_spaced'] 
    #sequence_generators = ['centerout_2D_discrete'] 
    
    input_type_list = ['shared','private', 'shared_scaled', 'private_scaled', 'all', 'all_scaled_by_shar',
        'sc_shared+unsc_priv', 'sc_shared+sc_priv', 'main_shared', 'main_sc_shared', 'main_sc_private', 'main_sc_shar+unsc_priv',
        'main_sc_shar+sc_priv','pca', 'split']

    input_type_0 = traits.OptionsList(*input_type_list, bmi3d_input_options=input_type_list)
    color_0 = traits.OptionsList(*target_colors.keys(), bmi3d_input_options=target_colors.keys())

    input_type_1 = traits.OptionsList(*input_type_list, bmi3d_input_options=input_type_list)
    color_1 = traits.OptionsList(*target_colors.keys(), bmi3d_input_options=target_colors.keys())
    
    input_type_2 = traits.OptionsList(*input_type_list, bmi3d_input_options=input_type_list)
    color_2 = traits.OptionsList(*target_colors.keys(), bmi3d_input_options=target_colors.keys())
    
    input_type_3 = traits.OptionsList(*input_type_list, bmi3d_input_options=input_type_list)
    color_3 = traits.OptionsList(*target_colors.keys(), bmi3d_input_options=target_colors.keys())
    
    choice_assist = traits.Float(0.)
    target_assist = traits.Float(0.)
    choice_target_rad = traits.Float(2.)

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
    hidden_traits = ['arm_hide_rate', 'arm_visible', 'hold_penalty_time', 'rand_start', 'reset', 'window_size', 'assist_level', 
        'assist_level_time', 'plant_hide_rate', 'plant_visible', 'show_environment', 'trials_per_reward']


    def __init__(self, *args, **kwargs):
        super(FreeChoiceFA, self).__init__(*args, **kwargs)

        seq_params = eval(kwargs.pop('seq_params', '{}'))
        print 'SEQ PARAMS: ', seq_params, type(seq_params)
        self.choice_per_n_blocks = seq_params.pop('blocks_per_free_choice', 1)
        self.n_free_choices = seq_params.pop('n_free_choices', 2)
        self.n_targets = seq_params.pop('ntargets', 8)

        self.input_type_dict = dict()
        self.input_type_dict[0]=self.input_type_0
        self.input_type_dict[0, 'color']=target_colors[self.color_0]

        self.input_type_dict[1]=self.input_type_1
        self.input_type_dict[1, 'color']=target_colors[self.color_1]

        self.input_type_dict[2]=self.input_type_2
        self.input_type_dict[2, 'color']=target_colors[self.color_2]

        self.input_type_dict[3]=self.input_type_3
        self.input_type_dict[3, 'color']=target_colors[self.color_3]

        # Instantiate the choice targets
        self.choices_targ_list = []
        for c in range(self.n_free_choices):
            self.choices_targ_list.append(target_graphics.VirtualCircularTarget(target_radius=self.choice_target_rad, 
                target_color=self.input_type_dict[c, 'color']))

        for c in self.choices_targ_list:
            for model in c.graphics_models:
                self.add_model(model)

        self.subblock_cnt = 0
        self.subblock_end = self.choice_per_n_blocks*self.n_targets
        self.choice_made = 0
        self.choice_ts = 0
        self.chosen_input_ix = -1
        self.choice_locs = np.zeros((self.n_free_choices, 3))

    def init(self):
        self.add_dtype('trial_type', np.str_, 16)
        self.add_dtype('choice_ix', 'f8', (1, ))
        self.add_dtype('choice_targ_loc', 'f8', (self.n_free_choices, 3))

        super(FreeChoiceFA, self).init()

    def _start_pre_choice_orig(self):
        target = self.targets[0]        
        target.move_to_position(np.array([0., 0., 0.]))
        target.cue_trial_start()
        self.chosen_input_ix = -1

    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        for target in self.choices_targ_list:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _test_enter_orig(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos)
        return d <= self.target_radius


    def update_level(self):
        pass

    def create_goal_calculator(self):
        self.goal_calculator = Choice_Goal_Calc(self.decoder.ssm)

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        target_state = self.goal_calculator(self.targs, self.choice_locs, self.choice_asst_ix, self.target_index, self.state)
        return np.array(target_state).reshape(-1,1)

    def _parse_next_trial(self):
        print 'parse next: ', self.next_trial[2][0]
        pairs = self.next_trial[0]
        self.targs = pairs[:, :, 1]
        self.choice_locs = pairs[:, :, 0]
        self.choice_asst_ix = self.next_trial[1][0]
        self.choice_instructed = self.next_trial[2][0]

        if self.subblock_cnt >= self.subblock_end:
            self.choice_made = 0
            self.subblock_cnt = 0

    def _test_make_choice(self, ts):
        return not self.choice_made

    def _cycle(self):
        self.task_data['trial_type'] = self.choice_instructed
        self.task_data['choice_ix'] = self.chosen_input_ix
        self.task_data['choice_targ_loc'] = self.choice_locs
        super(FreeChoiceFA, self)._cycle()

    def _start_choice_target(self):
        self.choice_ts = 0
        if self.choice_instructed == 'Free':
            for ic, c in enumerate(self.choices_targ_list):
                #move a target to current location (target1 and target2 alternate moving) and set location attribute
                c.move_to_position(self.choice_locs[ic, :])
                c.sphere.color = self.input_type_dict[ic, 'color']
                c.show()
        elif self.choice_instructed == 'Instructed':
            ic = self.choice_asst_ix
            c = self.choices_targ_list[ic]
            c.move_to_position(self.choice_locs[ic,:])
            c.sphere.color = self.input_type_dict[ic, 'color']
            c.show()

        target = self.targets[0]
        target.hide()
        self.choice_ts = 0

    def _start_target(self):
        super(FreeChoiceFA, self)._start_target()
        self.current_assist_level = self.target_assist
        for ic, c in enumerate(self.choices_targ_list):
            c.hide()

    def _test_enter_choice_target(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        enter_targ = 0
        for ic, c in enumerate(self.choice_locs):
            d = np.linalg.norm(cursor_pos - c)
            if d <= self.choice_target_rad: #NOTE, gets in if CENTER of cursor is in target (not entire cursor)
                enter_targ+=1

                #Set chosen as new input: 
                self.chosen_input_ix = ic
                self.decoder.filt.FA_input = self.input_type_dict[ic]
                print 'trial: ', self.decoder.filt.FA_input, self.choice_instructed

                #Declare that choice has been made:
                self.choice_made = 1

                #Change color of cursor: 
                sph = self.plant.graphics_models[0]
                sph.color = self.input_type_dict[ic, 'color']

        return enter_targ > 0

    def _test_trial_incomplete(self, ts):
        if self.choice_made == 0:
            return False
        else:
            return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts)
    def _start_reward(self):
        self.subblock_cnt+=1
        super(FreeChoiceFA, self)._start_reward()

    @staticmethod
    def centerout_2D_discrete_w_free_choice(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10, n_free_choices=2, blocks_per_free_choice = 1, percent_instructed=50.):
        return True

    @staticmethod
    def centerout_2D_discrete_w_free_choice_v2(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10, n_free_choices=2, blocks_per_free_choice = 1, percent_instructed=50.):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin and a sequence of 2D (x and z) target locations for nblocks 
        of free choices where the location of each choice changes. 

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between the targets in a pair.

        n_free_choices: number of choices. 

        Returns
        -------
        ([nblocks x ntargets x 2 x 3], [nblocks x n_free_choices x 3]) array of 1) pairs of target locations
        and 2) set of free choices 


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        theta_choice = []
        ix_choice_assist = []
        ix_choice_instructed = []
        for i in range(nblocks):
            temp_ = []
            for j in range(blocks_per_free_choice):
                temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
                np.random.shuffle(temp)
                temp_ = temp_+list(temp)

            theta.append(temp_)
            temp2 = np.arange(0, np.pi/2., np.pi/2./n_free_choices)+(np.pi/4.)+(np.pi/2.)*np.random.randint(0, 2)
            temp3 = np.random.randint(0, n_free_choices)
            temp4 = np.random.rand()
            if temp4 < percent_instructed/100.:
                ix_choice_instructed.append('Instructed')
            else:
                ix_choice_instructed.append('Free')

            np.random.shuffle(temp2)
            theta_choice.append(temp2)
            ix_choice_assist.append(temp3)

        theta = np.vstack(theta)
        theta_choice = np.vstack(theta_choice) #nblocks x n_free_choices
        ix_choice_assist = np.array(ix_choice_assist)
        ix_choice_instructed = np.array(ix_choice_instructed)

        #### calculate targets: 
        x = distance*np.cos(theta)
        y = np.zeros((nblocks, ntargets*blocks_per_free_choice))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([nblocks, ntargets*blocks_per_free_choice, 2, 3])
        pairs[:,:, 1,:] = np.dstack([x, y, z])

        #### calculate free choices: 
        x = distance*np.cos(theta_choice)
        y = np.zeros((nblocks, n_free_choices))
        z = distance*np.sin(theta_choice)

        choice = np.zeros((nblocks, n_free_choices, 3))
        choice = np.dstack(( x, y, z))

        g = []
        for i in range(nblocks):
            chz = choice[i, :, :]
            chz_assist = ix_choice_assist[i]
            type_chz = ix_choice_instructed[i]
            for j in range(ntargets*blocks_per_free_choice):
                tg = pairs[i, j, :, :]
                g.append((np.dstack((chz, tg)), [chz_assist], [type_chz]))
        return g

    @staticmethod
    def centerout_2D_discrete_w_free_choices_evenly_spaced(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10, n_free_choices=2, blocks_per_free_choice = 1, percent_instructed=50., choice_targ_ang=30.):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin and a sequence of 2D (x and z) target locations for nblocks 
        of free choices where the location of each choice changes -- specifically the location
        of the free choices are opposite the previous target, and spaced at 30 degree angle offsets

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between the targets in a pair.

        n_free_choices: number of choices. 

        Returns
        -------
        ([nblocks x ntargets x 2 x 3], [nblocks x n_free_choices x 3]) array of 1) pairs of target locations
        and 2) set of free choices 


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        theta_choice = []
        ix_choice_assist = []
        ix_choice_instructed = []
        last_targ_ang_ = 0.
        for i in range(nblocks):
            temp_ = []
            for j in range(blocks_per_free_choice):
                temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
                np.random.shuffle(temp)
                temp_ = temp_+list(temp)

            theta.append(temp_)

            ang = np.array([-choice_targ_ang*(np.pi/180), choice_targ_ang*(np.pi/180.)])
            temp2 = ang + np.pi + last_targ_ang_
            last_targ_ang_ = temp_[-1]
            
            temp3 = np.random.randint(0, n_free_choices)
            temp4 = np.random.rand()
            if temp4 < percent_instructed/100.:
                ix_choice_instructed.append('Instructed')
            else:
                ix_choice_instructed.append('Free')

            np.random.shuffle(temp2)
            theta_choice.append(temp2)
            ix_choice_assist.append(temp3)

        theta = np.vstack(theta)
        theta_choice = np.vstack(theta_choice) #nblocks x n_free_choices
        ix_choice_assist = np.array(ix_choice_assist)
        ix_choice_instructed = np.array(ix_choice_instructed)

        #### calculate targets: 
        x = distance*np.cos(theta)
        y = np.zeros((nblocks, ntargets*blocks_per_free_choice))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([nblocks, ntargets*blocks_per_free_choice, 2, 3])
        pairs[:,:, 1,:] = np.dstack([x, y, z])

        #### calculate free choices: 
        x = distance*np.cos(theta_choice)
        y = np.zeros((nblocks, n_free_choices))
        z = distance*np.sin(theta_choice)

        choice = np.zeros((nblocks, n_free_choices, 3))
        choice = np.dstack(( x, y, z))

        g = []
        for i in range(nblocks):
            chz = choice[i, :, :]
            chz_assist = ix_choice_assist[i]
            type_chz = ix_choice_instructed[i]
            for j in range(ntargets*blocks_per_free_choice):
                tg = pairs[i, j, :, :]
                g.append((np.dstack((chz, tg)), [chz_assist], [type_chz]))
        return g
