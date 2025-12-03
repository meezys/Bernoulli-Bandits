from methods.ts_ucb_2 import TS_UCB_2
from methods.ts_ucb_b import TS_UCB_B
from methods.ts_pmo import TS_UCB_A

CSV_FILE = 'results.csv'

import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

methods = [  TS_UCB_2, TS_UCB_B, TS_UCB_A]

'''Main function to run the simulation and plot results.
This function allows for customization of arms, horizon, and methods used in the simulation.
If no arms are provided, it randomly generates a set of arms with probabilities between 0.1 and 0.9.
The horizon is set to 10000 by default, but can be adjusted.
The methods parameter allows for selection of different bandit algorithms to be tested, 
with a default set including Greedy, ThompsonSampling, UCB, MOSS,'''
def main(trials = 1, arms=None,horizon=10000):
    # Validate input parameters
    m = [1, 20, 50, 100, 150]
    if arms is None or len(arms) < 2:
        number_of_arms = random.randint(2, 10)  # Randomly choose the number of arms
        arms = [random.uniform(0.1, 0.9) for _ in range(number_of_arms)]  # Randomly generate arm probabilities

    # Collect mean total regret per m to export to CSV
    results = []
    for ms in m:
        total_regret  = 0
        regrets = []
        for _ in range(trials):
            solution = TS_UCB_A(horizon, arms, ms)
            solution.run()
            regrets.append(solution.return_regret())
            total_regret += solution.total_regret()
        regrets = np.array(regrets)
        mean_total_regret = total_regret / trials
        print(f" TS-UCB({ms}) finished with an average regret of : {mean_total_regret:.4f}")
        results.append(f"{mean_total_regret:.4f}")

    # Write results to CSV with requested headings. Append if file exists.
    header = ['arms'] + [f"TS-UCB({ms})" for ms in m]
    row = [';'.join(map(str, arms))] + results
    csv_file = CSV_FILE
    try:
        file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
        mode = 'a' if file_exists else 'w'
        with open(csv_file, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        print(f"Wrote results to {csv_file} (appended={file_exists})")
    except Exception as e:
        print(f"Failed to write CSV: {e}")
        


'''To run a specific experiment, you can call the main function with desired parameters. You can specify the number of trials,
the arms (if you want to use specific probabilities), the horizon, and the methods you want to test. 
For example:
main(arms = [0.1, 0.2, 0.5], methods = [ThompsonSampling, Ada_UCB], trials = 50)
main(trials = 10) runs the simulation with 10 trials and the default methods and horizon, and random arms.
main(trials = 10, methods = [Greedy, ThompsonSampling, ETC]) runs it with 10 trials and the specified methods etc.
'''

import numpy as np
import pandas as pd

# Generate 10 random pairs
n_pairs = 10
x = np.random.uniform(0, 1, n_pairs)
y = np.random.uniform(0, 1, n_pairs)


for i in range(n_pairs):
    main(trials = 1000, horizon=1000, arms = [round(x[i],3),round(y[i], 3)])

