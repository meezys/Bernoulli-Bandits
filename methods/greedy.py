from methods.alphabeta import AlphaBeta
import numpy as np
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