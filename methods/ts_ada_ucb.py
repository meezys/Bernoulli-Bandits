from .ts_ucb import TS_UCB
import numpy as np

class TS_Ada_UCB(TS_UCB):
    def __init__(self, horizon, arms):
        super().__init__(horizon, arms)

    
    def index(self, arm, round):
        return np.sqrt(2/self.number_of_trials[arm] * self.log_plus(self.round/self.summy(arm)))
    
    def summy(self, i):
        sum = 0
        for j in range(self.number_of_arms):
            sum += min(self.total(i), np.sqrt(self.total(i) * self.total(i)))
        return sum
    
