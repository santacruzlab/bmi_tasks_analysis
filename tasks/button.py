'''Docstring for button.py'''

from riglib.experiment import LogExperiment, traits

class ButtonTask(LogExperiment):
    side = traits.String("left", desc='Use "left" for one side, "right" for the other')
    reward_time = traits.Float(5, desc='Amount of reward (in seconds)')
    penalty_time = traits.Float(5, desc='Amount of penalty (in seconds)')
    
    status = dict(
        left=dict(left_correct="reward", left_incorrect="penalty", stop=None),
        right=dict(right_correct="reward", right_incorrect="penalty", stop=None),
        reward=dict(post_reward="picktrial"),
        penalty=dict(post_penalty="picktrial"),
    )
    
    state = "picktrial"
    
    def __init__(self, **kwargs):
        from riglib import button
        super(ButtonTask, self).__init__(**kwargs)
        self.button = button.Button()
    
    def _start_picktrial(self):
        self.set_state(self.side)
    
    def _get_event(self):
        if self.button is not None:
            return self.button.pressed()
        return None
    
    def _while_left(self):
        self.event = self._get_event()
        
    def _while_right(self):
        self.event = self._get_event()
    
    def _test_left_correct(self, ts):
        return self.event is not None and self.event in [1, 2]
    def _test_left_incorrect(self, ts):
        return self.event is not None and self.event in [8, 4]
    def _test_right_correct(self, ts):
        return self.event is not None and self.event in [8, 4]
    def _test_right_incorrect(self, ts):
        return self.event is not None and self.event in [1, 2]
    
    def _test_post_reward(self, ts):
        return ts > self.reward_time
    
    def _test_post_penalty(self, ts):
        return ts > self.penalty_time
    
    def _test_both_correct(self, ts):
        return self.event is not None
    
    def _start_None(self):
        pass
