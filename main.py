import math
import random
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil, log  
# horizon is required for certain approaches, such as the ETC algorithm.
# It represents the total number of rounds or time steps in the simulation.

"""Abstract class for bandit methods."""
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

"""Abstract class for methods which make use of the alpha-beta distribution."""
class AlphaBeta(Method):
    '''The Greedy and Thompson Sampling Methods make use of the pair (alpha,beta) which is the number of successes 
    and failure for each arm, respectively. .'''
    def __init__(self, horizon, arms):
        super().__init__(horizon, arms)
        self.alpha_beta = [[1, 1]] * self.number_of_arms
    
    """Update the alpha-beta values for an arm given a generated result."""
    def alpha_beta_update(self, arm, result):
        self.alpha_beta[arm] = [self.alpha_beta[arm][0] + result, self.alpha_beta[arm][1] + (1 - result)]


""" Explore and then Commit algorithm: . Lattimore and Tardos (2009) Stochastic Bandits: Learning and Regret Analysis. 
6.1 Algorithm and Regret Analysis. Algorithm 1: Explore-then-commit"""
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

"""Greedy algorithm with moving average. A Tutorial on Thompson Sampling; Algorithm 3.1(BernGreedy), p15:"""
class Greedy(AlphaBeta):
    def run(self):
        for round in range(self.horizon):
            if round < self.number_of_arms:
                chosen = round
            else:
                chosen = np.argmax(self.sample_means)
            result = self.bernoulli_reward(self.arms[chosen])
            self.update_arm_average(chosen, result)
            self.alpha_beta_update(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])

"""Thompson Sampling algorithm. A Tutorial on Thompson Sampling; Algorithm 3.2(BernTS), p15"""
class ThompsonSampling(AlphaBeta):
    def run(self):
        self.alpha_beta = [[1, 1]] * self.number_of_arms
        for _ in range(self.horizon):
            for arm in range(self.number_of_arms):
                # Sample from the Beta distribution for each arm
                self.sample_means[arm] = np.random.beta(self.alpha_beta[arm][0], self.alpha_beta[arm][1])
            chosen = np.argmax(self.sample_means)
            result = self.bernoulli_reward(self.arms[chosen])
            self.alpha_beta_update(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])
"""Abstract class for UCB methods."""
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
        return 1 + t * (log(t)**2)
    def log_plus(self, x):
        """ Calculate the logarithm of x plus one."""
        return math.log(max(x, 1))


""" UCB algorithm: Upper Confidence Bound method. Lattimore and Tardos (2009), The Upper Confidence Bound
Algorithm: Asymptotic Optimality, Algorithm 6: Asymptotically Optimal UCB"""
class UCB(UCB_Methods):
    def index(self, arm, round): 
        return math.sqrt((2 * math.log(round + 1)) / self.number_of_trials[arm])

class UCB_2(UCB_Methods):
    """ UCB algorithm: Upper Confidence Bound method. Lattimore and Tardos (2009), The Upper Confidence Bound
    Algorithm: Asymptotic Optimality, Algorithm 6: Asymptotically Optimal UCB"""
    def index(self, arm, round):
        return math.sqrt((2 * math.log(self.f(round + 1))) / self.number_of_trials[arm])
    
""" MOSS Algorithm. Lattimore and Tardos (2009), The Upper Confidence Bound Algorithm: Minimax Optimality, Algorithm 7: MOSS"""
class MOSS(UCB_Methods):
    def index(self, arm, round):
        return math.sqrt((4 / self.number_of_trials[arm]) * self.log_plus(self.horizon / (self.number_of_arms * self.number_of_trials[arm])))
    
""" Ada_UCB Algorithm. Lattimore and Tardos (2009), The Upper Confidence Bound Algorithm: Minimax Optimality, 9.4 Bibliographical Remarks; Note 3 """
class Ada_UCB(UCB_Methods):
    def summy(self, i):
        sum = 0
        for j in range(self.number_of_arms):
            sum += min(self.number_of_trials[i], math.sqrt(self.number_of_trials[j] * self.number_of_trials[i]))
        return sum
    def index(self, arm, round):
        return math.sqrt(2/self.number_of_trials[arm] * self.log_plus(self.horizon/self.summy(arm)))

class KL_UCB(UCB_Methods):
    """ KL-UCB Algorithm. Lattimore and Tardos (2009), The Upper Confidence Bound Algorithm: Bernoulli Noise, Algorithm 8: KL-UCB"""
    def index(self, arm, round):
        p = self.sample_means[arm]
        rhs = self.f(round + 1) / self.number_of_trials[arm]
        low, high = p, 1.0
        while high - low > 1e-6:
            q = (low + high) / 2
            if self.kl_divergence(p, q) > rhs:
                high = q
            else:
                low = q
        return low - p  # The index is added to sample_mean outside.
    def kl_divergence(self, p, q):
        """Calculate the KL divergence between two Bernoulli distributions."""
        if p == 0:
            return 0 if q == 0 else float('inf')
        if p == 1:
            return 0 if q == 1 else float('inf')
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


'''Main function to run the simulation and plot results.
This function allows for customization of arms, horizon, and methods used in the simulation.
If no arms are provided, it randomly generates a set of arms with probabilities between 0.1 and 0.9.
The horizon is set to 10000 by default, but can be adjusted.
The methods parameter allows for selection of different bandit algorithms to be tested, 
with a default set including Greedy, ThompsonSampling, UCB, MOSS,'''
def main(trials = 1, arms=None,horizon=10000, methods = [Greedy, ThompsonSampling, UCB, MOSS, ETC, Ada_UCB], display_arms = False):
    # Validate input parameters
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Horizon must be a positive integer.")
    if not isinstance(trials, int) or trials <= 0:
        raise ValueError("Trials must be a positive integer.")
    if methods is None or len(methods) == 0:
        raise ValueError("At least one method must be provided.")
    if arms is None or len(arms) < 2:
        number_of_arms = random.randint(2, 10)  # Randomly choose the number of arms
        arms = [random.uniform(0.1, 0.9) for _ in range(number_of_arms)]  # Randomly generate arm probabilities
    if not isinstance(arms, list):
        raise ValueError("Arms must be a list of probabilities.")
    if not all(0 <= arm <= 1 for arm in arms):
        raise ValueError("All arm probabilities must be between 0 and 1.") 
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer.")
    if not isinstance(trials, int) or trials <= 0:
        raise ValueError("Trials must be a positive integer.")
    if not all(callable(method) for method in methods):
        raise ValueError("All methods must be callable (i.e., classes or functions).")
    if not all(issubclass(method, Method) for method in methods):
        raise ValueError("All methods must be subclasses of the Method class.")
    if not all(hasattr(method, 'run') for method in methods):
        raise ValueError("All methods must have a 'run' method defined.")

    # Plot regret histories for each method
    print("Simulation Parameters:")
    print(f"  Methods: {[method.__name__ for method in methods]}")
    print(f"  Horizon: {horizon}")
    print(f"  Arms (rounded): {[round(arm, 2) for arm in arms]}")
    print(f"  Number of trials: {trials}")
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10.colors  # or any colormap
    for i, method in enumerate(methods):
        regrets = []
        for _ in range(trials):
            solution = method(horizon, arms)
            solution.run()
            regrets.append(solution.return_regret())    
        regrets = np.array(regrets)
        if trials > 1:
            mean_regret = np.mean(regrets, axis = 0)
            std_regret = np.std(regrets, axis = 0)
            print(f"Method: {method.__name__}, Mean Regret: {mean_regret[-1]:.4f}, Std Dev: {std_regret[-1]:.4f}")
            for regret in regrets:
                plt.plot(regret, color=colors[i], alpha = 0.1)  # Plot each trial with some transparency
            plt.plot(mean_regret, color=colors[i], label=method.__name__, linewidth=2)
        else:
            plt.plot(regrets[0], color=colors[i], label=method.__name__, linewidth=2)
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Regret over Rounds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if not display_arms:
        plt.show()
    else: 
        plt.show(block=False)  # Show non-blocking
        # Plot distribution of arm mean rewards for the last method run, useful for when the arm averages are randomly generated    
        solution= Greedy(horizon, arms)  # Create an instance of the method with the given horizon and arms
        optimality_gaps = sorted(solution.optimality_gaps, reverse=True)[:-1]  # Get the first n-1 elements
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(optimality_gaps)), optimality_gaps, tick_label=[f'Arm {i+1}' for i in range(len(optimality_gaps))], color=colors[-1])
        plt.xlabel('Arms')
        plt.ylabel('Suboptimality Gap')
        plt.title('Distribution of Arm Suboptimality Gaps')
        plt.tight_layout()
        plt.show()
        


'''To run a specific experiment, you can call the main function with desired parameters. You can specify the number of trials,
the arms (if you want to use specific probabilities), the horizon, and the methods you want to test. 
For example:
main(arms = [0.1, 0.2, 0.5], methods = [ThompsonSampling, Ada_UCB], trials = 50)
main(trials = 10) runs the simulation with 10 trials and the default methods and horizon, and random arms.
main(trials = 10, methods = [Greedy, ThompsonSampling, ETC]) runs it with 10 trials and the specified methods etc.

'''

# main(trials = 10, methods = [Greedy, ThompsonSampling, ETC])
# main(trials = 10)
# main(arms = [0.1, 0.2, 0.5], methods = [ThompsonSampling, Ada_UCB], trials = 50)


main(methods = [UCB, Ada_UCB], trials = 50)

