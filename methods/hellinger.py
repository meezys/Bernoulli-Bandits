from methods.method import Method
import numpy as np
eps = 2e-16

class Hellinger(Method):
    def run(self):
        """Run the Hellinger UCB algorithm."""
        for round in range(self.horizon):
            chosen = np.argmax(self.index(round))
            result = self.bernoulli_reward(self.arms[chosen])
            self.update_arm_average(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])

    def index(self, round):
        """Calculate the Hellinger UCB index for each arm."""
        indices = []
        for arm in range(self.number_of_arms):
            if self.number_of_trials[arm] == 0:
                indices.append(float('inf'))  # Infinite index for untried arms
            else:
                indices.append(min(1, self.hellinger_ucb(self.sample_means[arm], np.divide(np.log(round), self.number_of_trials[arm]), precision=1e-10)))
        return indices
    
    def hellinger_ucb(self, mean, log_term, precision=1e-10):
        upperbound = 1
        return self.hellinger(mean, log_term, lowerbound=-1, upperbound=upperbound, precision=precision)

    def hellinger(self, mean, log_term, lowerbound, upperbound, precision=1e-10):
        """Calculate the Hellinger UCB index for a given mean and log term."""
        l = max(mean, lowerbound)
        u = upperbound
        while (u-l) > precision:
            m = (l+u)/2
            if self.hellinger_bern(mean, m) > log_term:
                u = m
            else:
                l = m
        return (l+u)/2

    def hellinger_bern(self, x, y):
        x = min(max(x, eps), 1-eps)
        y = min(max(y, eps), 1-eps)
        return 1. - np.sqrt(x*y) - np.sqrt((1-x)*(1-y))