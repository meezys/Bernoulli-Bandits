from methods.method import Method
class AlphaBeta(Method):
    '''The Greedy and Thompson Sampling Methods make use of the pair (alpha,beta) which is the number of successes 
    and failure for each arm, respectively. .'''
    def __init__(self, horizon, arms):
        super().__init__(horizon, arms)
        self.alpha_beta = [[1, 1]] * self.number_of_arms
    
    """Update the alpha-beta values for an arm given a generated result."""
    def alpha_beta_update(self, arm, result):
        self.alpha_beta[arm] = [self.alpha_beta[arm][0] + result, self.alpha_beta[arm][1] + (1 - result)]
    
    def alpha_beta_mean(self, arm):
        """Return the mean of the Beta distribution for a given arm."""
        return self.alpha_beta[arm][0] / (self.alpha_beta[arm][0] + self.alpha_beta[arm][1])
    
    def alpha_beta_variance(self, arm):
        """Return the variance of the Beta distribution for a given arm."""
        alpha, beta = self.alpha_beta[arm]
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    
    def total(self,arm):
        """Return the total number of trials for a given arm."""
        return self.alpha_beta[arm][0] + self.alpha_beta[arm][1]