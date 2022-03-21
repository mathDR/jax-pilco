import jax.numpy as jnp
import objax
import gym
from bayesnewton.utils import softplus_inv
from pilco.models import PILCO
from pilco.controllers import RbfController
from pilco.rewards import ExponentialReward

from utils import rollout

seed = 0
objax.random.Generator(seed)


if __name__ == "__main__":
    SUBS = 3
    bf = 30
    maxiter = 50
    max_action = 2.0
    target = jnp.array([1.0, 0.0, 0.0])
    weights = jnp.diag(jnp.array([2.0, 2.0, 0.3]))
    m_init = jnp.reshape(jnp.array([-1.0, 0, 0.0]), (1, 3))
    S_init = jnp.diag(jnp.array([0.01, 0.05, 0.01]))
    T = 40
    T_sim = T
    J = 4
    N = 8
    restarts = 2

    env = gym.make("Pendulum-v1")

    # Initial random rollouts to generate a dataset
    X, Y, _, _ = rollout(
        env=env,
        pilco=None,
        timesteps=T,
        verbose=True,
        random=True,
        SUBS=SUBS,
        render=True,
    )
    for i in range(1, J):
        X_, Y_, _, _ = rollout(
            env=env,
            pilco=None,
            timesteps=T,
            verbose=True,
            random=True,
            SUBS=SUBS,
            render=True,
        )
        X = jnp.vstack((X, X_))
        Y = jnp.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim

    controller = RbfController(
        state_dim=state_dim,
        control_dim=control_dim,
        num_basis_functions=bf,
        max_action=max_action,
    )
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    pilco = PILCO(
        (X, Y),
        controller=controller,
        horizon=T,
        reward=R,
        m_init=m_init,
        S_init=S_init,
        trainable_likelihood_variance=False,
    )

    # for numerical stability, we can set the likelihood variance parameters
    # of the GP models
    for model in pilco.mgpr.models:
        model.likelihood.transformed_variance.value = softplus_inv(jnp.array(0.001))

    r_new = []
    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=restarts)
        pilco.optimize_policy(maxiter=maxiter, restarts=restarts)
        print("About to rollout")
        X_new, Y_new, _, _ = rollout(
            env=env,
            pilco=pilco,
            timesteps=T_sim,
            verbose=True,
            random=False,
            SUBS=SUBS,
            render=True,
        )

        # Update dataset
        X = jnp.vstack((X, X_new))
        Y = jnp.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))
