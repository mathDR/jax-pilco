import jax.numpy as jnp
import objax
from scipy.stats import multivariate_normal


class RiskOfCollision(objax.Module):
    def __init__(self, state_dim, low, high):

        self.state_dim = state_dim
        self.low = low
        self.high = high

    def compute_reward(self, m, s):
        infl_diag_S = 2*objax.Vectorize(
            objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection()),
            objax.VarCollection(),
            )(s)

        dist1 = multivariate_normal(mean=m[0, 0], cov=infl_diag_S[0])
        dist2 = multivariate_normal(mean=m[0, 2], cov=infl_diag_S[2])
        risk = (
            dist1.cdf(self.high[0]) - dist1.cdf(self.low[0])
            ) * (
            dist2.cdf(self.high[1]) - dist2.cdf(self.low[1])
            )
        return risk, 0.0001 * jnp.ones(1)


class SingleConstraint(objax.Module):
    def __init__(self, dim, high=None, low=None, inside=True):
        if high is None:
            self.high = False
        else:
            self.high = high
        if low is None:
            self.low = False
        else:
            self.low = low
        if high is None and low is None:
            raise Exception(
                "At least one of bounds (high,low) "
                "has to be defined"
                )
        self.dim = int(dim)
        if inside:
            self.inside = True
        else:
            self.inside = False

    def compute_reward(self, m, s):
        # Risk refers to the space between the low and high value -> 1
        # otherwise self.in = 0
        if not self.high:
            dist = multivariate_normal(
                mean=m[0, self.dim], cov=s[self.dim, self.dim]
            )
            risk = 1 - dist.cdf(self.low)
        elif not self.low:
            dist = multivariate_normal(
                mean=m[0, self.dim], cov=s[self.dim, self.dim]
            )
            risk = dist.cdf(self.high)
        else:
            dist = multivariate_normal(
                mean=m[0, self.dim], cov=s[self.dim, self.dim]
            )
            risk = dist.cdf(self.high) - dist.cdf(self.low)
        if not self.inside:
            risk = 1 - risk
        return risk, 0.0001 * jnp.ones(1)


class ObjectiveFunction(objax.Module):
    def __init__(self, reward_f, risk_f, mu=1.0):
        self.reward_f = reward_f
        self.risk_f = risk_f
        self.mu = objax.StateVar(mu)

    def compute_reward(self, m, s):
        reward, var = self.reward_f.compute_reward(m, s)
        risk, _ = self.risk_f.compute_reward(m, s)
        return reward - self.mu * risk, var
