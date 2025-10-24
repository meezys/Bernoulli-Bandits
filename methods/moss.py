from .ucb import UCB_Methods
import numpy as np

class MOSS(UCB_Methods):
    def index(self, arm, round):
        return np.sqrt((4 / self.number_of_trials[arm]) * self.log_plus(self.horizon / (self.number_of_arms * self.number_of_trials[arm])))