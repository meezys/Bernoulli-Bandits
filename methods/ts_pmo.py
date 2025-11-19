from .alphabeta import AlphaBeta
import numpy as np
import random

class TS_UCB_A(AlphaBeta):
    def __init__(self, horizon, arms):
        random.seed(42)
        super().__init__(horizon, arms)

    def run(self):
        m = 150
        sampled_values = np.zeros(self.number_of_arms)
        for _ in range(self.horizon):
                # Sample from the Beta distribution for each arm
            sampled_values = [[np.random.beta(self.alpha_beta[arm][0], self.alpha_beta[arm][1]) for arm in range(self.number_of_arms)] for _ in range(m)]
            f_i = [np.max(sampled_values[i]) for i in range(m)]
            f_t = sum(f_i)/m 
            psi = [np.sqrt(self.ab_total(i)) *(f_t - self.alpha_beta_mean(i)) for i in range(self.number_of_arms)]
            chosen = np.argmin(psi)
            self.alpha_beta_update(chosen,self.bernoulli_reward(self.arms[chosen]))
            self.next_regret(self.optimality_gaps[chosen])
    
    # Var is $$\frac{p(1-p)}{n}$$ so that Informaton is $$\frac{n}{p(1-p)}$$
    

    def information(self, arm):
        p = self.alpha_beta_mean(arm)
        n = self.ab_total(arm)
        if p * (1 - p) == 0:
            return float('inf')  # Infinite information if variance is zero
        return n / (p * (1 - p))

    
    