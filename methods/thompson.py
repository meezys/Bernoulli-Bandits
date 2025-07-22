from methods.alphabeta import AlphaBeta
import numpy as np
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