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

    env = gym.make("Pendulum-v1").env

    # Initial random rollouts to generate a dataset
    X, Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, render=True)
    for i in range(1, J):
        X_, Y_, _, _ = rollout(
            env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True, render=True
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
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)
        print("About to rollout")
        # X_new, Y_new, _, _ = rollout(
        #     env,
        #     pilco,
        #     timesteps=T_sim,
        #     verbose=True,
        #     random=False,
        #     SUBS=SUBS,
        #     render=True,
        # )
        X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T_sim)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        # for i in range(len(X_new)):
        #     r_new.append(
        #         R.compute_reward(X_new[i, None, :-1], 0.001 * jnp.eye(state_dim))[0]
        #     )
        # total_r = sum(r_new)
        # _, _, r = pilco.predict(X_new[0, None, :-1], 0.001 * S_init, T)
        # print("Total ", total_r, " Predicted: ", r)

        # Update dataset
        X = jnp.vstack((X, X_new))
        Y = jnp.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))
