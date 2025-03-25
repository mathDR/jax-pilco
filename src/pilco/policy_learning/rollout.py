import jax
import jax.random as jr
import typing as tp
from flax import nnx
from ..controller import Controller

Model = tp.TypeVar("Model", bound=nnx.Module)
from jaxtyping import ArrayLike
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)
from typing import Tuple

num_particles = 100


def one_rollout_step(carry: Tuple, t: ScalarFloat) -> Tuple[Tuple, ScalarFloat]:
    compute_action, predict_all_outputs, key, samples, total_cost = carry
    key, *subkeys = jr.split(key, num_particles + 1)
    u = jax.vmap(compute_action)(
        samples, jnp.tile(t, num_particles), jnp.array(subkeys)
    )
    this_state = jnp.hstack((samples, u))
    predictive_moments = predict_all_outputs(this_state)
    key, subkey = jr.split(key)
    samples = jnp.squeeze(
        vectorized_sample(
            key,
            predictive_moments[:, :, 0],
            jax.vmap(jnp.diag)(predictive_moments[:, :, 1]),
            1,
        )
    )
    cost = jnp.sum(jax.vmap(cart_pole_cost)(samples))
    return (compute_action, predict_all_outputs, key, samples, total_cost + cost), cost


def rollout(
    policy: Controller,
    model: Model,
    init_samples: ArrayLike,
    timesteps: ArrayLike,
    key: KeyArray = jr.PRNGKey(42),
) -> ScalarFloat:
    action = Partial(policy.compute_action)
    pao = Partial(model.predict_all_outputs)
    (action, pao, key, samples, total_cost), result = jax.lax.scan(
        one_rollout_step, (action, pao, key, init_samples, 0), timesteps
    )
    return total_cost / len(timesteps)
