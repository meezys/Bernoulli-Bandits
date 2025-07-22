import numpy as np
from methods.ucb_methods import UCB_Methods

class UCB_2(UCB_Methods):
    """ UCB algorithm: Upper Confidence Bound method. Lattimore and Tardos (2009), The Upper Confidence Bound
    Algorithm: Asymptotic Optimality, Algorithm 6: Asymptotically Optimal UCB"""
    def index(self, arm, round):
        return np.sqrt((2 * np.log(self.f(round + 1))) / self.number_of_trials[arm])