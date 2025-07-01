import math
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil, log  
# horizon is required for certain approaches, such as the ETC algorithm.
# It represents the total number of rounds or time steps in the simulation.

class Method:
    """Abstract class for bandit methods."""
    def __init__(self, horizon, arms):
        self.horizon = horizon
        self.arms = arms
        self.number_of_arms = len(arms)
        self.sample_means = [0] * self.number_of_arms
        self.number_of_trials = [0] * self.number_of_arms
        self.regret_history = []

        self.optimal_arm = max(arms)
        self.optimality_gaps = [self.optimal_arm - arm for arm in arms]
        self.delta = np.mean([self.optimal_arm - arm for arm in arms if arm != self.optimal_arm])
        
    def run(self):
        """Run the method and return the regret."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def plot(self):
        """Plot the regret history."""
        x = plt.plot(range(len(self.regret_history)), self.regret_history, label=self.__class__.__name__)
        return x

    def bernoulli_reward(self, arm):
        """Simulate a Bernoulli reward for a given arm."""
        return 1 if np.random.rand() < arm else 0
    
    def next_regret(self, current_regret):
        length = len(self.regret_history)
        if length > 0:
            new = (self.regret_history[-1] * ((length - 1) / length)) \
            + (current_regret / length)
        else:
            new = current_regret    
        self.regret_history.append(new)
    
    def return_regret(self):
        """Return the regret history."""
        return self.regret_history
    
    def update_arm_average(self, arm, result):
        """Update the moving average for the given arm."""
        if self.number_of_trials[arm] > 0:
            self.sample_means[arm] = (self.sample_means[arm] * ((self.number_of_trials[arm] - 1) / self.number_of_trials[arm])) \
            + (result / self.number_of_trials[arm])
        else:
            self.sample_means[arm] = result
        self.number_of_trials[arm] += 1

class AlphaBeta(Method):
    """Abstract class for methods which make use of the alpha-beta distribution."""
    def __init__(self, horizon, arms):
        super().__init__(horizon, arms)
        self.alpha_beta = [[1, 1]] * self.number_of_arms
    def alpha_beta_update(self, arm, result):
        """Update the moving average for the given arm using the alpha-beta method."""
        self.alpha_beta[arm] = [self.alpha_beta[arm][0] + result, self.alpha_beta[arm][1] + (1 - result)]

class eTC(Method):
    """Epsilon-Greedy with Thompson Sampling."""
    def run(self):
        m = max(1, ceil(4 / self.delta**2 * log(self.horizon * self.delta**2 / 4)))
        for round in range(self.horizon):
            if round <= self.number_of_arms * m:
                arm = round % self.number_of_arms
            else:
                arm = np.argmax(self.sample_means)
            self.update_arm_average(arm, self.bernoulli_reward(self.arms[arm]))
            self.next_regret(self.optimality_gaps[arm])


"""Simulate a Bernoulli reward for a given arm."""
def bernoulli_reward(arm):
    return 1 if np.random.rand() < arm else 0

class greedy(AlphaBeta):
    """Greedy algorithm with moving average."""
    def run(self):
        for round in range(self.horizon):
            if round < self.number_of_arms:
                chosen = round
            else:
                chosen = np.argmax(self.sample_means)
            result = bernoulli_reward(self.arms[chosen])
            self.update_arm_average(chosen, result)
            self.alpha_beta_update(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])

    def greedy_update(self, arm, result):
        """Update the moving average for the given arm using the alpha-beta method."""
        self.alpha_beta_update(self.alpha_beta, arm, result)
        self.arms_array[arm] = self.alpha_beta[arm][0] / (self.alpha_beta[arm][0] + self.alpha_beta[arm][1])

class ThompsonSampling(AlphaBeta):
    """Thompson Sampling algorithm."""
    def run(self):
        self.alpha_beta = [[1, 1]] * self.number_of_arms
        for _ in range(self.horizon):
            for arm in range(self.number_of_arms):
                # Sample from the Beta distribution for each arm
                self.sample_means[arm] = np.random.beta(self.alpha_beta[arm][0], self.alpha_beta[arm][1])
            chosen = np.argmax(self.sample_means)
            result = bernoulli_reward(self.arms[chosen])
            self.alpha_beta_update(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])
class UCB_Methods(Method):
    """Abstract class for UCB methods."""
    def __init__(self, horizon, arms):
        super().__init__(horizon, arms)
        self.ucb_values = [0] * self.number_of_arms

class UCB(UCB_Methods):
    """Upper Confidence Bound algorithm."""
    def run(self):
        for i in range(self.number_of_arms):
            self.sample_means[i] = bernoulli_reward(self.arms[i])
            self.number_of_trials[i] = 1
            self.next_regret(self.optimality_gaps[i])
        for round in range(self.number_of_arms, self.horizon):
            self.ucb_values = [self.sample_means[i] + math.sqrt(2 * log(round + 1) / self.number_of_trials[i]) 
                          for i in range(self.number_of_arms)]
            chosen = np.argmax(self.ucb_values)
            result = bernoulli_reward(self.arms[chosen])
            self.update_arm_average(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])
        return self.regret_history
class MOSS(UCB_Methods):
    def run(self):
        for i in range(self.number_of_arms):
            self.sample_means[i] = bernoulli_reward(self.arms[i])
            self.number_of_trials[i] = 1
            self.next_regret(self.optimality_gaps[i])
        for _ in range(self.number_of_arms, self.horizon):
            self.ucb_values = [self.sample_means[i] + 
                               math.sqrt((4 / self.number_of_trials[i]) * self.log_plus(self.horizon) / self.number_of_arms * self.number_of_trials[i]) 
                               for i in range(self.number_of_arms)]
            chosen = np.argmax(self.ucb_values)
            result = bernoulli_reward(self.arms[chosen])
            self.update_arm_average(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])
    def log_plus(x):
        """ Calculate the logarithm of x plus one."""
        return math.log(max(x, 1))


def plot(etc, thompson, greedy_regret, ucb_regret, moss_regret):
# Plotting the results
    plt.plot(range(len(etc)), etc, label='ETC Regret', color='blue')
    plt.plot(range(len(thompson)), thompson, label='Thompson Sampling Regret', color='orange')
    plt.plot(range(len(greedy_regret)), greedy_regret, label='Greedy Regret', color='green')
    plt.plot(range(len(ucb_regret)), ucb_regret, label='UCB Regret', color='red')
    plt.plot(range(len(moss_regret)), moss_regret, label='MOSS Regret', color='purple')
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Regret over Rounds')
    plt.legend()
    plt.show()

def main(horizon=10000, arms=[0.7, 0.8, 0.1, 0.9]):
    """Main function to run the simulation."""
    solution = UCB(horizon, arms)  # Create an instance of the ETC method
    # Run the ETC method
    # Plot the results
    solution.run()  # Run the method
    solution.plot()  # Plot the regret history
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Regret over Rounds')
    plt.legend()
    plt.show()  # Show the plot without blocking the execution
main()  # Execute the simulation
# This will run the simulations and plot the results.