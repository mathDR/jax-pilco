import jax.numpy as jnp
import objax


class ExponentialReward(objax.Module):
    def __init__(self, state_dim, W=None, t=None):
        self.state_dim = state_dim
        if W is not None:
            self.W = objax.StateVar(jnp.reshape(W, (state_dim, state_dim)))
        else:
            self.W = objax.TrainVar(jnp.eye(state_dim))
        if t is not None:
            self.t = objax.StateVar(jnp.reshape(t, (1, state_dim)))
        else:
            self.t = objax.StateVar(jnp.zeros((1, state_dim)))

    def compute_reward(self, m, s):
        """
        Reward function, calculating mean and variance of rewards, given
        mean and variance of state distribution, along with the target State
        and a weight matrix.
        Input m : [1, k]
        Input s : [k, k]

        Output M : [1, 1]
        Output S : [1, 1]
        """

        SW = s @ self.W

        iSpW = jnp.transpose(
            jnp.linalg.solve((jnp.eye(self.state_dim) + SW), jnp.transpose(self.W))
        )

        muR = jnp.exp(-(m - self.t) @ iSpW @ jnp.transpose(m - self.t) / 2) / jnp.sqrt(
            jnp.linalg.det(jnp.eye(self.state_dim) + SW)
        )

        i2SpW = jnp.transpose(
            jnp.linalg.solve(
                (jnp.eye(self.state_dim) + 2 * SW),
                jnp.transpose(self.W),
            )
        )

        r2 = jnp.exp(-(m - self.t) @ i2SpW @ jnp.transpose(m - self.t)) / jnp.sqrt(
            jnp.linalg.det(jnp.eye(self.state_dim) + 2 * SW)
        )

        sR = r2 - muR @ muR
        return muR, sR


class LinearReward(objax.Module):
    def __init__(self, state_dim, W):
        self.state_dim = state_dim
        self.W = objax.StateVar(jnp.reshape(W, (state_dim, 1)))

    def compute_reward(self, m, s):
        muR = jnp.reshape(m, (1, self.state_dim)) @ self.W
        sR = jnp.transpose(self.W) @ s @ self.W
        return muR, sR


class CombinedRewards(objax.Module):
    def __init__(self, state_dim, rewards=[], coefs=None):
        self.state_dim = state_dim
        self.base_rewards = rewards
        if coefs is not None:
            self.coefs = objax.StateVar(coefs)
        else:
            self.coefs = objax.StateVar(jnp.ones(len(rewards)))

    def compute_reward(self, m, s):
        total_output_mean = 0
        total_output_covariance = 0
        for reward, coef in zip(self.base_rewards, self.coefs):
            output_mean, output_covariance = reward.compute_reward(m, s)
            total_output_mean += coef * output_mean
            total_output_covariance += coef**2 * output_covariance

        return total_output_mean, total_output_covariance
