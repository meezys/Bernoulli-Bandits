from .alphabeta import AlphaBeta
import numpy as np
import random

class TS_UCB_B(AlphaBeta):
    def __init__(self, horizon, arms):
        random.seed(42)
        super().__init__(horizon, arms)

    def run(self):
        m = 150
        for _ in range(self.horizon):
            # Sample from the Beta distribution for each arm (m times total)
            f_i = [max([np.random.beta(self.alpha_beta[arm][0], self.alpha_beta[arm][1]) for arm in range(self.number_of_arms)]) for _ in range(m)]
            arms_subset = [self.subset_array(f_i, int(ms[i])) for i in range(self.number_of_arms)]
            
            # compute per-arm averages (guard against zero ms)
            f_t = [(arms_subset[i].sum() / ms[i]) if ms[i] != 0 else 0.0 for i in range(self.number_of_arms) ]
            psi = [np.sqrt(self.ab_total(i)) * (f_t[i] - self.alpha_beta_mean(i)) for i in range(self.number_of_arms)]
            chosen = np.argmin(psi)
            self.alpha_beta_update(chosen, self.bernoulli_reward(self.arms[chosen]))
            self.next_regret(self.optimality_gaps[chosen])
    
    # Var is $$\frac{p(1-p)}{n}$$ so that Informaton is $$\frac{n}{p(1-p)}$$
    
    def subset_array(self, arr, m):
        """
        Return a subset of size m (m <= len(arr)) sampled without replacement.
        Accepts lists or numpy arrays and returns a numpy array.
        """
        a = np.asarray(arr)
        n = a.shape[0]
        if m > n:
            raise ValueError("m must be <= len(arr)")
        if m == n:
            return a.copy()
        indices = np.random.choice(n, size=m, replace=False)
        return a[indices]
    def information(self, arm):
        p = self.alpha_beta_mean(arm)
        n = self.ab_total(arm)
        if p * (1 - p) == 0:
            return float('inf')  # Infinite information if variance is zero
        return n / (p * (1 - p))


