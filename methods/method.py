import numpy as np
import matplotlib.pyplot as plt
class Method:

    def __init__(self, horizon, arms):

        """Initialize the method with a horizon and a list of arms."""
        self.horizon = horizon
        self.arms = arms
        self.number_of_arms = len(arms)
        
        '''Common metrics used across methods.'''
        self.sample_means = [0] * self.number_of_arms
        self.regret_history = []
        self.number_of_trials = [0] * self.number_of_arms

        '''Initialize the optimal arm and calculate the optimality gaps. These are used to calculate regret.'''
        self.optimal_arm = max(arms)
        self.optimality_gaps = [self.optimal_arm - arm for arm in arms]
        self.delta = np.mean([self.optimal_arm - arm for arm in arms if arm != self.optimal_arm])

    """Run the method given an environment and horizon regret."""
    def run(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    """Each solution is able to plot its regret history ."""
    def plot(self):
        x = plt.plot(range(len(self.regret_history)), self.regret_history, label=self.__class__.__name__)
        return x

    """Generating a random result that follows a Bernoulli Distribution."""
    def bernoulli_reward(self, arm):
        """Simulate a Bernoulli reward for a given arm."""
        return 1 if np.random.rand() < arm else 0
    
    """Update the regret history with the current regret."""
    def next_regret(self, current_regret):
        ''"Update the regret history with the current regret."""
        length = len(self.regret_history)
        if length > 0:
            new = (self.regret_history[-1] * ((length - 1) / length)) \
            + (current_regret / length)
        else:
            new = current_regret    
        self.regret_history.append(new)
    
    """Return the regret history."""
    def return_regret(self):
        return self.regret_history
    
    def update_arm_average(self, arm, result):
        """Update the average for the given arm without recalculating the mean of the entire list, using the previous
        mean and the number of entries."""
        if self.number_of_trials[arm] > 0:
            self.sample_means[arm] = (self.sample_means[arm] * ((self.number_of_trials[arm] - 1) / self.number_of_trials[arm])) \
            + (result / self.number_of_trials[arm])
        else:
            self.sample_means[arm] = result
        self.number_of_trials[arm] += 1