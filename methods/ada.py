from methods.ucb import UCB_Methods
import numpy as np

class Ada_UCB(UCB_Methods):
    def summy(self, i):
        sum = 0
        for j in range(self.number_of_arms):
            sum += min(self.number_of_trials[i], np.sqrt(self.number_of_trials[j] * self.number_of_trials[i]))
        return sum
    def index(self, arm, round):
        return np.sqrt(2/self.number_of_trials[arm] * self.log_plus(self.horizon/self.summy(arm)))