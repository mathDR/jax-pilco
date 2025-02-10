import jax.numpy as jnp
import jax.random as jr
from jax.typing import ArrayLike
from typing import List


class Controller:
    """
    Superclass of controller objects
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        to_squash: bool = False,
        max_action: float = 1.0,
    ):
        super(Controller, self).__init__()
        # model parameters
        self.state_dim = state_dim
        self.input_dim = input_dim

        # set squashing function
        if to_squash:
            self.f_squash = lambda x: self.squashing(x, max_action)
        else:
            # assign the identity function
            self.f_squash = lambda x: x

    def compute_action(self, states: List[Array], key: ArrayLike) -> float:
        raise NotImplementedError()

    def squashing(self, u: Array, u_max: float) -> ArrayLike:
        """
        Squash the inputs inside (-u_max, +u_max)
        """
        return u_max * jnp.tanh(u / u_max)


class RandomController(Controller):
    """Returns a random control output"""

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        to_squash: bool = False,
        max_action: float = 1.0,
    ):
        super(Controller, self).__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            to_squash=to_squash,
            max_action=max_action,
        )
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.max_action = max_action

    def compute_action(self, states: List[Array], key: ArrayLike) -> float:
        """
        Simple random action
        IN: current states and key to use for random action
        OUT: the action value
        """
        key, subkey = jr.split(key)
        return jr.uniform(key, minval=-max_action, maxval=max_action)


class Sum_of_sinusoids(Controller):
    """
    Exploration policy: sum of 'num_sin' sinusoids with random amplitudes and frequencies
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        num_sin: int,
        omega_min: Array,
        omega_max: Array,
        amplitude_min: Array,
        amplitude_max: Array,
        to_squash: bool = False,
        max_action: float = 1.0,
        key: Optiona[ArrayLike] = None,
    ):
        super(Sum_of_sinusoids, self).__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            to_squash=to_squash,
            max_action=max_action,
        )
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

    def forward(self, states: List[Array], t: Array) -> ArrayLike:
        # returns the controller values at times t
        return self.f_squash(
            jnp.sum(
                self.amplitudes * (jnp.sin(self.omega * t + self.phases)), axis=0
            ).reshape(-1, self.input_dim)
        )
