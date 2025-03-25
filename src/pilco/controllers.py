from flax import nnx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike
from typing import Generator, List, Optional


class Controller(nnx.Module):
    """
    Superclass of controller objects
    """

    state_dim: int
    input_dim: int
    to_squash: bool
    max_action: float

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        to_squash: bool = False,
        max_action: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # set squashing function
        if to_squash:
            self.f_squash = lambda x: self.squashing(x)
        else:
            # assign the identity function
            self.f_squash = lambda x: x

    def compute_action(
        self, states: ArrayLike, time_for_action: float, key: Optional[ArrayLike]
    ) -> Array:
        raise NotImplementedError()

    def squashing(self, u: Array) -> ArrayLike:
        """
        Squash the inputs inside (-max_action, +max_action)
        """
        return self.max_action * jnp.tanh(u / self.max_action)


class RandomController(Controller):
    """Returns a random control output"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        to_squash: bool = False,
        max_action: float = 1.0,
    ):
        super().__init__(
            state_dim,
            action_dim,
            to_squash,
            max_action,
        )

    def compute_action(
        self, states: ArrayLike, time_for_action: float, key: Optional[ArrayLike] = None
    ) -> Array:
        """
        Simple random action
        IN: current state, time_for_action and key to use for random action
        OUT: the action value (uniform in (-max_action,+max_action))
        """
        if key is not None:
            key, subkey = jr.split(key)
        else:
            key = jr.key(123)
            key, subkey = jr.split(key)

        return jr.uniform(
            subkey,
            shape=(self.action_dim,),
            minval=-self.max_action,
            maxval=self.max_action,
        )


class Sum_of_Sinusoids(Controller):
    """
    Exploration policy: sum of 'num_sin' sinusoids with random amplitudes and frequencies
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        num_sin: int,
        omega_min: ArrayLike,
        omega_max: ArrayLike,
        amplitude_min: ArrayLike,
        amplitude_max: ArrayLike,
        to_squash: bool = False,
        max_action: float = 1.0,
        key: Optional[ArrayLike] = None,
    ):
        super(Sum_of_sinusoids, self).__init__()
        self.state_dim = (state_dim,)
        self.input_dim = (input_dim,)
        self.to_squash = (to_squash,)
        self.max_action = (max_action,)

        if key is None:
            key = jr.key(123)
        self.num_sin = num_sin
        # generate random parameters
        key, subkey = jr.split(key)
        self.amplitudes = jr.uniform(
            key, shape=(num_sin, input_dim), minval=amplitude_min, maxval=amplitude_max
        )
        key, subkey = jr.split(key)
        self.omega = jr.uniform(
            key, shape=(num_sin, input_dim), minval=omega_min, maxval=omega_max
        )
        key, subkey = jr.split(key)
        self.phases = jr.uniform(
            key, shape=(num_sin, input_dim), minval=jnp.pi, maxval=jnp.pi
        )

    def compute_action(
        self, states: List[ArrayLike], t: ArrayLike
    ) -> Generator[Array, None, None]:
        # returns the controller values at times t
        yield self.f_squash(
            jnp.sum(
                self.amplitudes * (jnp.sin(self.omega * t + self.phases)), axis=0
            ).reshape(-1, self.input_dim)
        )
