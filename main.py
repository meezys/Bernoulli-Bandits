import numpy as np
import matplotlib.pyplot as plt 
from math import ceil, log  
# horizon is required for certain approaches, such as the ETC algorithm.
# It represents the total number of rounds or time steps in the simulation.
horizon = 10000
arms = [0.7, 0.8, 0.9]
number_of_arms = len(arms)
optimal_arm = max(arms)
optimality_gaps = [optimal_arm - arm for arm in arms]
#Calculate the delta value based on the optimal arm and the other arms
# In the case of two arms, delta is the difference between the optimal arm and the other arm.
delta = np.mean([optimal_arm - arm for arm in arms if arm != optimal_arm])


"""Simulate a Bernoulli reward for a given arm."""
def bernoulli_reward(arm):
    return 1 if np.random.rand() < arm else 0


def ETC(delta):
    round = 0
    regret = []
    arms_array = [0] * number_of_arms
    number_of_trials = [0] * number_of_arms
    m = max(1, ceil(4/ delta**2 * log(horizon * delta**2/4)))
    for round in range(horizon):
        if round <= m * number_of_arms:
            arm = round % number_of_arms
            result = bernoulli_reward(arms[arm])
            update_arm_average(arms_array, number_of_trials, arm, result)
            regret.append(next_regret(regret[-1] if regret else 0, optimality_gaps[arm], round + 1))
        else:
            arm = np.argmax(arms_array)
            result = bernoulli_reward(arms[arm])
            update_arm_average(arms_array, number_of_trials, arm, result)
            regret.append(next_regret(regret[-1] if regret else 0, optimality_gaps[arm], round + 1))
    return regret

def ETC(round, delta):
    regret = []
    arms_array = [0] * number_of_arms
    number_of_trials = [0] * number_of_arms
    m = max(1, ceil(4 / delta**2 * log(horizon * delta**2 / 4)))
    

"""Moving average calculation: A_k = k-1/k * A_k-1 + 1/k * v_k."""
def update_arm_average(arms_array, number_of_trials, arm, result):
    """Update the moving average for the given arm."""
    if number_of_trials[arm] > 0:
        arms_array[arm] = (arms_array[arm] * ((number_of_trials[arm] - 1) / number_of_trials[arm])) \
        + (result / number_of_trials[arm])
    else:
        arms_array[arm] = result
    number_of_trials[arm] += 1

def next_regret(prev_regret, current_regret, length):
    if length > 0:
        new = (prev_regret * ((length - 1) / length)) \
        + (current_regret / length)
    else:
        return current_regret
    return new

def run_simulation():
    for round in range(horizon):
        etc = ETC(round, delta)  # Start the simulation with the first round
        


etc1 = ETC(delta)  # Start the simulation with the first round
plt.plot(range(len(etc)), etc, label='ETC Regret')
plt.xlabel('Rounds')
plt.ylabel('Regret')
plt.title('Regret over Rounds')
plt.legend()
plt.show()