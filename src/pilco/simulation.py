import equinox as eqx
from typing import Callable
from jax import Array
from jax.typing import ArrayLike
from controllers import Controller
from scipy import signal

import gymnasium as gym


class Simulation(eqx.Module):
    """
    Dynamic System simulation
    """

    def __init__(self, env: gym.wrappers.common.TimeLimit):
        """
        env: the mujoco environment
        """
        self.env = env

    def rollout(
        self,
        initial_state: ArrayLike,
        policy: Controller,
        num_timesteps: int,
        dt: float,
    ) -> Tuple[Array, Array, Array]:
        """
        Generate a rollout of length num_timesteps.
        'noise' defines the standard deviation of a Gaussian measurement noise.
            initial state
            policy: policy function
            num_timesteps: length rollout (s)
            dt: sampling time (s)
        """
        state_dim = len(initial_state)
        discrete_times = np.linspace(0, num_timesteps, int(num_timesteps / dt) + 1)
        num_samples = len(discrete_times)

        # get first input
        u0 = policy.compute_action(initial_state, 0.0)
        num_inputs = len(u0)

        # init variables
        inputs = []
        states = []
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0

        states[0, :] = initial_state

        for i, t in enumerate(discrete_times[:-1]):
            # get input
            u = np.array(policy(states[i, :], t))
            actions[i, :] = u
            # get next state
            x_new, r, done, _ = env.step(np.array(u))
            ep_return_full += r
            if done:
                break
            states.append(xnew)

        # last u (only to have the same number of input and state samples)
        actions[-1, :] = np.array(policy.compute_action(states[-1, :], num_timesteps))

        return jnp.array(actions), jnp.array(states)
