from methods.greedy import Greedy
from methods.thompson import ThompsonSampling
from methods.ucb import UCB
from methods.moss import MOSS
from methods.etc import ETC
from methods.ada import Ada_UCB
from methods.method import Method
from methods.kl_ucb import KL_UCB
from methods.hellinger import Hellinger

import random
import numpy as np
import matplotlib.pyplot as plt

methods = [Greedy, ThompsonSampling, UCB, MOSS, ETC, Ada_UCB, KL_UCB, Hellinger]

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
    # plt.mode(dark=True)  # Set dark mode for the plot
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
            plt.plot(mean_regret, color=colors[i], label=method.__name__, linewidth=2)
        else:
            plt.plot(regrets[0], color=colors[i], label=method.__name__, linewidth=2)
    # next_color_index = len(methods) % len(colors) + 1
    # hl_and_kl = copied.plot(trials = trials, arms = arms)
    # plt.plot(hl_and_kl[0], color=colors[next_color_index], label='KL', linewidth=2, linestyle='--')
    # next_color_index += 1
    # plt.plot(hl_and_kl[1], color=colors[next_color_index], label='HL', linewidth=2, linestyle='--')
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Regret over Rounds')
    plt.legend()
    plt.grid(True)
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


main(methods = [Hellinger, KL_UCB, Ada_UCB, ThompsonSampling], trials = 500, horizon=1000)

