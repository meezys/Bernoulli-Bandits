from .alphabeta import AlphaBeta
import numpy as np

class TS_UCB_2(AlphaBeta):
    def __init__(self, horizon, arms):
        super().__init__(horizon, arms)

    def run(self):
        sampled_values = np.zeros(self.number_of_arms)
        for _ in range(self.horizon):
            for arm in range(self.number_of_arms):
                # Sample from the Beta distribution for each arm
                sampled_values[arm] = np.random.beta(self.alpha_beta[arm][0], self.alpha_beta[arm][1])
            f_t = np.max(sampled_values)
            psi = [np.sqrt(self.ab_total(i)) *(f_t - self.alpha_beta_mean(i)) for i in range(self.number_of_arms)]
            chosen = np.argmin(psi)
            self.alpha_beta_update(chosen,self.bernoulli_reward(self.arms[chosen]))
            self.next_regret(self.optimality_gaps[chosen])
    
    # Var is $$\frac{p(1-p)}{n}$$ so that Informaton is $$\frac{n}{p(1-p)}$$


    
    