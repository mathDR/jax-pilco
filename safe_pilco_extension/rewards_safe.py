import jax.numpy as jnp
import objax
from scipy.stats import norm


class RiskOfCollision(objax.Module):
    def __init__(self, state_dim, low, high):

        self.state_dim = state_dim
        self.low = low
        self.high = high

    def compute_reward(self, m, s, low, high):
        infl_diag_S = jnp.diag(s, k=0)

        dist1 = norm(loc=m[0, 0], scale=infl_diag_S[0])
        dist2 = norm(loc=m[0, 2], scale=infl_diag_S[2])
        # risk = (
        #     dist1.cdf(self.high[0]) - dist1.cdf(self.low[0])
        #     ) * (
        #     dist2.cdf(self.high[1]) - dist2.cdf(self.low[1])
        #     )
        risk = (
            dist1.cdf(high[0]) - dist1.cdf(low[0])
            ) * (
            dist2.cdf(high[1]) - dist2.cdf(low[1])
        )
        return risk, 0.0001


class SingleConstraint(objax.Module):
    def __init__(self, dim, high=None, low=None, inside=True):
        if high is None:
            self.high = 1e9
        else:
            self.high = high
        if low is None:
            self.low = -1e9
        else:
            self.low = low
        if high is None and low is None:
            raise Exception(
                "At least one of bounds (high,low) "
                "has to be defined"
                )
        self.dim = int(dim)
        if inside:
            self.inside = 1.0
        else:
            self.inside = 0.0

    def compute_reward(self, m, s):
        # Risk refers to the space between the low and high value -> 1
        # otherwise self.in = 0
        dist = norm(loc=m[0, self.dim], scale=s[self.dim, self.dim])
        risk = dist.cdf(self.high) - dist.cdf(self.low)
        # TODO: add back inside somehow without breaking jit compile
        return risk, 0.0001


class ObjectiveFunction(objax.Module):
    def __init__(self, reward_f, risk_f, mu=1.0):
        self.reward_f = reward_f
        self.risk_f = risk_f
        self.mu = objax.StateVar(mu)

    def compute_reward(self, m, s):
        reward, var = self.reward_f.compute_reward(m, s)
        risk, _ = self.risk_f.compute_reward(m, s)
        return reward - self.mu.value * risk, var
