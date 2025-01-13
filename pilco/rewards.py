import jax.numpy as jnp
import equinox as eqx
from jax import Array
from jax.typing import ArrayLike
from typing import Optional, Tuple


class ExponentialReward(eqx.Module):
    def __init__(
        self,
        state_dim: int,
        W: Optional[ArrayLike] = None,
        t: Optional[ArrayLike] = None,
    ):
        self.state_dim = state_dim
        if W is not None:
            self.W = W
        if t is not None:
            self.t = t

    def compute_reward(
        self,
        mean_state_distribution: ArrayLike,
        covariance_state_distibution: ArrayLike,
    ) -> Tuple[Array, Array]:
        """
        Reward function, calculating mean and variance of rewards, given
        mean and variance of state distribution, along with the target State
        and a weight matrix.
        Input mean_state_distribution : [1, k]
        Input s : [k, k]

        Output M : [1, 1]
        Output S : [1, 1]
        """
        # TODO: Clean up this

        SW = covariance_state_distibution @ self.W
        I_state_dim = jnp.eye(self.state_dim)
        I_plus_SW = I_state_dim + SW
        I_plus_2SW = I_state_dim + 2 * SW
        mu_minus_t = mean_state_distribution - self.t

        iSpW = jnp.transpose(jnp.linalg.solve(I_plus_SW, jnp.transpose(self.W)))

        mu_reward = jnp.exp(
            -0.5 * mu_minus_t @ iSpW @ jnp.transpose(mu_minus_t)
        ) / jnp.sqrt(jnp.linalg.det(I_plus_SW))

        i2SpW = jnp.transpose(
            jnp.linalg.solve(
                I_plus_2SW,
                jnp.transpose(self.W),
            )
        )

        r2 = jnp.exp(-mu_minus_t @ i2SpW @ jnp.transpose(mu_minus_t)) / jnp.sqrt(
            jnp.linalg.det(I_plus_2SW)
        )

        covariance_reward = r2 - muR @ muR
        return mu_reward, covariance_reward


class LinearReward(eqx.Module):
    def __init__(self, state_dim: int, W: ArrayLike):
        self.state_dim = state_dim
        self.W = W

    def compute_reward(
        self,
        mean_state_distribution: ArrayLike,
        covariance_state_distibution: ArrayLike,
    ) -> Tuple[Array, Array]:
        mu_reward = jnp.reshape(mean_state_distribution, (1, self.state_dim)) @ self.W
        covariance_reward = (
            jnp.transpose(self.W) @ covariance_state_distibution @ self.W
        )
        return mu_reward, covariance_reward


class CombinedRewards(eqx.Module):
    def __init__(
        self, state_dim: int, rewards: list = [], coefs: Optional[ArrayLike] = None
    ):
        self.state_dim = state_dim
        self.base_rewards = rewards
        if coefs is not None:
            self.coefs = coefs
        else:
            self.coefs = jnp.ones(len(rewards))

    def compute_reward(
        self,
        mean_state_distribution: ArrayLike,
        covariance_state_distibution: ArrayLike,
    ) -> Tuple[Array, Array]:
        total_output_mean = 0
        total_output_covariance = 0
        for reward, coef in zip(self.base_rewards, self.coefs):
            output_mean, output_covariance = reward.compute_reward(
                mean_state_distribution, covariance_state_distibution
            )
            total_output_mean += coef * output_mean
            total_output_covariance += jnp.square(coef) * output_covariance

        return total_output_mean, total_output_covariance
