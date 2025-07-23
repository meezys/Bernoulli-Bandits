from .alphabeta import AlphaBeta
import numpy as np

class TS_UCB(AlphaBeta):
    def __init__(self, horizon, arms):
        super().__init__(horizon, arms)

    def run(self):
        sampled_values = np.zeros(self.number_of_arms)
        for _ in range(self.horizon):
            for arm in range(self.number_of_arms):
                # Sample from the Beta distribution for each arm
                sampled_values[arm] = np.random.beta(self.alpha_beta[arm][0], self.alpha_beta[arm][1])
            f_t = np.max(sampled_values)
            psi = [np.divide(f_t - self.alpha_beta_mean(i), self.radius(i, round)) for i in range(self.number_of_arms)]
            chosen = np.argmin(psi)
            self.alpha_beta_update(chosen,self.bernoulli_reward(self.arms[chosen]))
            self.next_regret(self.optimality_gaps[chosen])
    
    def radius(self, arm, round):
        """Calculate the radius for the UCB value based on the number of trials."""
        return np.sqrt((3 * np.log(self.horizon) )/ max(1, self.total(arm) - 2))