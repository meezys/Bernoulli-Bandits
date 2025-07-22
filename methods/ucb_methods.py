from methods.method import Method
import numpy as np

class UCB_Methods(Method):
    def __init__(self, horizon, arms):
        super().__init__(horizon, arms)
        self.ucb_values = [0] * self.number_of_arms
    def run(self):
        for i in range(self.number_of_arms):
            self.sample_means[i] = self.bernoulli_reward(self.arms[i])
            self.number_of_trials[i] = 1
            self.next_regret(self.optimality_gaps[i])
        for round in range(self.number_of_arms, self.horizon):
            # All the UCB methods are similar, only different in the index calculation.
            self.ucb_values = [self.sample_means[i] + self.index(i, round) 
                               for i in range(self.number_of_arms)]
            chosen = np.argmax(self.ucb_values)
            result = self.bernoulli_reward(self.arms[chosen])
            self.update_arm_average(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])
    def index(self, arm, round):
        """Calculate the index for the UCB value. This method should be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    def f(self, t):
        return 1 + t * (np.log(t)**2)
    def log_plus(self, x):
        """ Calculate the logarithm of x plus one."""
        return np.log(max(x, 1))