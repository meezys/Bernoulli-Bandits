from .alphabeta import AlphaBeta
import numpy as np
import random

class TS_UCB_A(AlphaBeta):
    def __init__(self, horizon, arms, m):
        random.seed(42)
        super().__init__(horizon, arms)
        self.m = m

    def run(self):
        m = self.m 
        K = self.number_of_arms
        
        for t in range(self.horizon):
            # 1. Current Posterior Parameters
            ab = np.array(self.alpha_beta)
            alphas = ab[:, 0]
            betas = ab[:, 1]
            
            # -------------------------------------------------------
            # Step A: Calculate f_t_tilde (The Target)
            # -------------------------------------------------------
            
            # Draw m independent samples from posterior q(H_t)
            # Shape: (m, K) - m simulations for each of the K arms
            sampled_params = np.random.beta(alphas, betas, size=(m, K))
            
            # Calculate f_i for each simulation (Max reward of best arm in that sample)
            # Shape: (m,)
            max_rewards = sampled_params.max(axis=1)
            
            # Calculate f_t_tilde: The average of these maximums
            # Eq: (1/m) * sum(f_i)
            f_t_tilde = max_rewards.mean()

            # -------------------------------------------------------
            # Step B: Calculate Psi (The Selection Criteria)
            # -------------------------------------------------------
            
            # Posterior Mean: mu_hat(a)
            mu_hat = alphas / (alphas + betas)
            
            # Count N_a (total pulls per arm)
            N = alphas + betas
            
            # Define Radius: U_t(a) - mu_hat(a)
            # In your previous code, you used 1/sqrt(N).
            # If using standard UCB radius, it would be sqrt(log(t)/N).
            # Below implements the logic consistent with your previous snippet:
            confidence_radius = 1.0 / np.sqrt(N)
            
            # Avoid division by zero if N is 0 (though priors usually prevent this)
            confidence_radius[confidence_radius == 0] = 1e-6 
            
            # Equation (1): (f_t_tilde - mu_hat) / radius
            psi = (f_t_tilde - mu_hat) / confidence_radius
            
            # -------------------------------------------------------
            # Step C: Update
            # -------------------------------------------------------
            
            # Minimize Psi
            chosen = int(np.argmin(psi))
            
            # Update history
            reward = self.bernoulli_reward(self.arms[chosen])
            self.alpha_beta_update(chosen, reward)
            self.next_regret(self.optimality_gaps[chosen])
    # Var is $$\frac{p(1-p)}{n}$$ so that Informaton is $$\frac{n}{p(1-p)}$$

    
    