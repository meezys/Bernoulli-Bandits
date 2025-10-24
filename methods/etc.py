from .method import Method
from math import ceil, log
import numpy as np
class ETC(Method):
    def run(self):
        '''Equation (6.5) Lattimore and Tardos (2009) Stochastic Bandits page 93.'''
        m = max(1, ceil(4 / self.delta**2 * log(self.horizon * self.delta**2 / 4)))
        for round in range(self.horizon):
            if round <= self.number_of_arms * m:
                arm = round % self.number_of_arms
            else:
                arm = np.argmax(self.sample_means)
            self.update_arm_average(arm, self.bernoulli_reward(self.arms[arm]))
            self.next_regret(self.optimality_gaps[arm])