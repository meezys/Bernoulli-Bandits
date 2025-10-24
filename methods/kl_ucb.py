from .method import Method
import numpy as np
eps = 2e-16
class KL_UCB(Method):
    def run(self):
        """Run the KL-UCB algorithm."""
        for round in range(self.horizon):
            chosen = np.argmax(self.index(round))
            result = self.bernoulli_reward(self.arms[chosen])
            self.update_arm_average(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])

    def index(self, round):
        """Calculate the KL-UCB index for each arm."""
        indices = []
        for arm in range(self.number_of_arms):
            if self.number_of_trials[arm] == 0:
                indices.append(float('inf'))  # Infinite index for untried arms
            else:
                indices.append(min(1, self.kl_ucb(self.sample_means[arm], np.divide(np.log(round), self.number_of_trials[arm]) ,precision=1e-10)))
        return indices
    
    def kl_ucb(self, mean, log_term, precision=1e-10):
        upperbound = min(1, (mean + np.sqrt(2 * log_term)))
        return self.klucb(mean, log_term, lowerbound=0, upperbound=upperbound, precision=precision)

    def klucb(self, mean, log_term, lowerbound, upperbound, precision=1e-10):
        """Calculate the KL-UCB index for a given mean and log term."""
        l = max(mean, lowerbound)
        u = upperbound
        while (u-l)>precision:
            m = (l+u)/2
            # print(div(x, m))
            if self.kl_bern(mean, m)>log_term:
                u = m
            else:
                l = m
        return (l+u)/2

    def kl_bern(self,x,y):
        x = min(max(x, eps), 1-eps)
        y = min(max(y, eps), 1-eps)
        return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))
    
