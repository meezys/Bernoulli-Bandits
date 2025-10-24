from .ucb import UCB_Methods
import numpy as np

# $\hat{\mu}(t-1)+ \sqrt{\frac{2}{T_i(t-1)}\log\biggl(\frac{t}{\sum_{j=1}^{k}\min\{T_i(t-1), \sqrt{T_i(t-1)T_j(t-1)}\}} \biggr)}$

class Ada_UCB(UCB_Methods):
    def summy(self, i):
        sum = 0
        for j in range(self.number_of_arms):
            sum += min(self.number_of_trials[i], np.sqrt(self.number_of_trials[j] * self.number_of_trials[i]))
        return sum
    def index(self, arm, a):
        return np.sqrt(2/self.number_of_trials[arm] * self.log_plus( a/self.summy(arm)))
    