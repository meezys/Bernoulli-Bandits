import numpy as np
from methods.ucb_methods import UCB_Methods

class UCB(UCB_Methods):
    def index(self, arm, round): 
        return np.sqrt((2 * np.log(round + 1)) / self.number_of_trials[arm])
