import objax
import jax.numpy as jnp
from pilco.models import PILCO
from jax.lax import fori_loop


class SafePILCO(PILCO):
    def __init__(
        self, data, num_induced_points=None, horizon=30, controller=None,
        reward_add=None, reward_mult=None, m_init=None, S_init=None,
        trainable_likelihood_variance=True, name=None, mu=5.0
    ):
        super(SafePILCO, self).__init__(
            data,
            num_induced_points=num_induced_points,
            horizon=horizon,
            controller=controller,
            reward=reward_add,
            m_init=m_init,
            S_init=S_init,
            trainable_likelihood_variance=trainable_likelihood_variance
            )
        if reward_mult is None:
            raise exception("have to define multiplicative reward")

        self.mu = objax.StateVar(mu)

        self.reward_mult = reward_mult

    def predict(self, m_x, s_x, n, low, high):

        init_val = (m_x, s_x, 0.0, 1.0)

        def body_fun(i, v):
            m_x, s_x, reward_add, reward_mult = v
            return (
                *self.propagate(m_x, s_x),
                jnp.add(
                    reward_add,
                    jnp.squeeze(
                        self.reward.compute_reward(
                            m_x,
                            s_x,
                        )[0]
                    )
                ),
                jnp.multiply(
                    reward_mult,
                    jnp.squeeze(
                        self.reward_mult.compute_reward(
                            m_x,
                            s_x,
                            low,
                            high,
                        )[0]
                    )
                ),
            )

        val = fori_loop(0, n, body_fun, init_val)

        m_x, s_x, reward_add, reward_mult = val

        reward_total = reward_add + self.mu.value * (1.0 - reward_mult)
        return m_x, s_x, reward_total
