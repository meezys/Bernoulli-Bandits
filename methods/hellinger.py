from .method import Method
import numpy as np
eps = 2e-16

# $U_i = \arg \max_i \sup \{\dot\psi(\theta): H^2(P_{\hat{\theta_{i,t--1}}}, P_\theta) \leq 1- \exp(-c\frac{\log{(t)}}{N_i(t)})\} $
class Hellinger(Method):
    def run(self):
        """Run the Hellinger UCB algorithm."""
        for round in range(self.horizon):
            chosen = np.argmax(self.index(round))
            result = self.bernoulli_reward(self.arms[chosen])
            self.update_arm_average(chosen, result)
            self.next_regret(self.optimality_gaps[chosen])

    def index(self, round):
        c = 0.25 + eps
        """Calculate the Hellinger UCB index for each arm."""
        indices = []
        for arm in range(self.number_of_arms):
            if self.number_of_trials[arm] == 0:
                indices.append(float('inf'))  # Infinite index for untried arms
            else:
                indices.append(min(1, 
                hellinger_ucb(self.sample_means[arm], 
                1. - np.exp(-c * np.log(round)/self.number_of_trials[arm]), 
                precision=eps)))
        return indices

def hellinger_ucb(mean, log_term, precision=1e-15):
    upperbound = 1
    return hellinger(mean, log_term, lowerbound=-1, upperbound=upperbound, precision=precision)

def hellinger( mean, log_term, lowerbound, upperbound, precision=1e-6):
    """Calculate the Hellinger UCB index for a given mean and log term."""
    l = max(mean, lowerbound)
    u = upperbound
    while (u-l) > precision:
        m = (l+u)/2
        if hellinger_bern(mean, m) > log_term:
            u = m
        else:
            l = m
    return (l+u)/2
def hellinger_bern(x, y):
    x = min(max(x, eps), 1-eps)
    y = min(max(y, eps), 1-eps)
    return 1. - np.sqrt(x*y) - np.sqrt((1-x)*(1-y))

def H2UCBBernoulli(x, d, precision):
    """
    Close-form HellingerUCB for low latency
    """
    x = min(max(x, precision), 1-precision)
    m = 1 - d
    a = m**2 - 1 + x
    H2ucb = 2*x*m**2-a+np.sqrt((a-2*x*m**2)**2-a**2)
   
    return(H2ucb)