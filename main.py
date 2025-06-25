import math
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil, log  
# horizon is required for certain approaches, such as the ETC algorithm.
# It represents the total number of rounds or time steps in the simulation.
 

"""Simulate a Bernoulli reward for a given arm."""
def bernoulli_reward(arm):
    return 1 if np.random.rand() < arm else 0


def ETC():
    round = 0
    regret = []
    arms_array = [0] * number_of_arms
    number_of_trials = [0] * number_of_arms
    m = max(1, ceil(4/ delta**2 * log(horizon * delta**2/4)))
    for round in range(horizon):
        if round <= m * number_of_arms:
            arm = round % number_of_arms
        else:
            arm = np.argmax(arms_array)
        result = bernoulli_reward(arms[arm])
        update_arm_average(arms_array, number_of_trials, arm, result)
        regret.append(next_regret(regret[-1] if regret else 0, optimality_gaps[arm], round + 1))
    return regret

def greedy():
    regret = []
    arms_array = [0] * number_of_arms
    alpha_beta = [[1, 1]] * number_of_arms
    for round in range(horizon):
        if round < number_of_arms:
            chosen = round
        else:
            chosen = np.argmax(arms_array)
        result = bernoulli_reward(arms[chosen])
        greedy_update(arms_array, alpha_beta, chosen, result)
        regret.append(next_regret(regret[-1] if regret else 0, optimality_gaps[chosen], round + 1))
    return regret

def greedy_update(arms_array, alpha_beta, arm, result):
    """Update the moving average for the given arm using the alpha-beta method."""
    alpha_beta_update(alpha_beta, arm, result)
    arms_array[arm] = alpha_beta[arm][0] / (alpha_beta[arm][0] + alpha_beta[arm][1])

def thompson_sampling():
    regret = []
    arms_array = [0] * number_of_arms
    alpha_beta = [[1, 1]] * number_of_arms
    for round in range(horizon):
        for arm in range(number_of_arms):
            # Sample from the Beta distribution for each arm
            arms_array[arm] = np.random.beta(alpha_beta[arm][0], alpha_beta[arm][1])
        
        chosen = np.argmax(arms_array)
        result = bernoulli_reward(arms[chosen])
        # Update the regret based on the chosen arm's optimality gap
        # and the current round.
        alpha_beta_update(alpha_beta, chosen, result)
        regret.append(next_regret(regret[-1] if regret else 0, optimality_gaps[chosen], round + 1))
    return regret

def alpha_beta_update(alpha_beta, arm, result):
    """Update the moving average for the given arm using the alpha-beta method."""
    alpha_beta[arm] = [alpha_beta[arm][0] + result, alpha_beta[arm][1] + (1 - result)]

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
"""This function calculates the next regret value based on the previous regret,
the current regret, and the length.
The chosen methods are ETC, Greedy, and Thompson Sampling."""
def run_simulations():
    global horizon
    horizon = 10000
    global arms
    arms = [0.7, 0.8, 0.1, 0.9]  # Example arm probabilities
    global number_of_arms
    number_of_arms = len(arms)
    
    # the optimal arm is the one with the highest probability of success
    global optimal_arm
    optimal_arm = max(arms)
    # Calculate the optimality gaps for each arm
    global optimality_gaps
    optimality_gaps = [optimal_arm - arm for arm in arms]
    # The delta 
    global delta
    delta = np.mean([optimal_arm - arm for arm in arms if arm != optimal_arm])

    # Calculate the delta value based on the optimal arm and the other arms
    thompson = thompson_sampling()  # Run Thompson Sampling
    etc = ETC()  # Start the simulation with the first round
    greedy_regret = greedy()  # Run Greedy algorithm
    
    # Plot the results
    plot(etc, thompson, greedy_regret)

def plot(etc, thompson, greedy_regret):
# Plotting the results
    plt.plot(range(len(etc)), etc, label='ETC Regret', color='blue')
    plt.plot(range(len(thompson)), thompson, label='Thompson Sampling Regret', color='orange')
    plt.plot(range(len(greedy_regret)), greedy_regret, label='Greedy Regret', color='green')
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Regret over Rounds')
    plt.legend()
    plt.show()

run_simulations()  # Execute the simulation
# This will run the simulations and plot the results.