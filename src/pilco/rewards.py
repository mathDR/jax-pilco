from flax import nnx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from typing import Tuple


class ExponentialReward(nnx.Module):
    """
    Compute expectation and variance, and their derivatives of an exponentiated negative quadratic cost
    exp( -(x-z).T * W * (x-z)/2 )
    where x ~ N(m,S)
    """

    def __init__(
        self,
        state_dim: int,
        weight_matrix: ArrayLike = None,
        target_state: ArrayLike = None,
    ):
        self.state_dim = state_dim
        if weight_matrix is not None:
            self.weight_matrix = weight_matrix.reshape(state_dim, state_dim)
        else:
            self.weight_matrix = jnp.eye(state_dim)

        if target_state is not None:
            self.target_state = target_state.reshape(1, state_dim)
        else:
            self.target_state = jnp.zeros((1, state_dim))

    def compute_reward(
        self, state_mean: ArrayLike, state_covariance: ArrayLike
    ) -> Tuple[Array, Array]:
        """
        Reward function, calculating mean and variance of rewards, given
        mean and variance of state distribution, along with the target State
        and a weight matrix.
        Input state_mean : [self.state_dim ,1]
        Input state_covariance : [self.state_dim, self.state_dim]

        Output expected_reward : [1, 1]
        Output reward_variance : [1, 1]
        """
        mu = state_mean - self.target_state
        eye_state_dim = jnp.eye(self.state_dim)

        State_Covar_times_Weight = jnp.dot(state_covariance, self.weight_matrix)

        iSpW = jnp.transpose(
            jnp.linalg.solve(
                (eye_state_dim + State_Covar_times_Weight),
                jnp.transpose(self.weight_matrix),
            )
        )

        expected_reward = jnp.exp(
            -0.5 * jnp.dot(mu, jnp.dot(iSpW, jnp.transpose(mu)))
        ) / jnp.sqrt(jnp.linalg.det(eye_state_dim + State_Covar_times_Weight))

        i2SpW = jnp.transpose(
            jnp.linalg.solve(
                (eye_state_dim + 2 * State_Covar_times_Weight),
                jnp.transpose(self.weight_matrix),
            )
        )

        r2 = jnp.exp(-jnp.dot(mu, jnp.dot(i2SpW, jnp.transpose(mu)))) / jnp.sqrt(
            jnp.linalg.det(eye_state_dim + 2 * State_Covar_times_Weight)
        )

        reward_variance = r2 - jnp.dot(state_mean, state_mean)
        return expected_reward, reward_variance
