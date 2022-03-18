import jax.numpy as jnp
import objax
import gym
from bayesnewton.utils import softplus_inv
from pilco.models import PILCO
from pilco.controllers import RbfController
from pilco.rewards import ExponentialReward

from utils import rollout, Normalised_Env

objax.random.Generator(0)

SUBS = 5
T = 25
env = gym.make("MountainCarContinuous-v0")
# Initial random rollouts to generate a dataset
X1, Y1, _, _ = rollout(
    env=env, pilco=None, random=True, timesteps=T, SUBS=SUBS, render=True
)
for i in range(1, 5):
    X1_, Y1_, _, _ = rollout(
        env=env, pilco=None, random=True, timesteps=T, SUBS=SUBS, render=True
    )
    X1 = jnp.vstack((X1, X1_))
    Y1 = jnp.vstack((Y1, Y1_))
env.close()

env = Normalised_Env(
    "MountainCarContinuous-v0", jnp.mean(X1[:, :2], 0), jnp.std(X1[:, :2], 0)
)
X = jnp.zeros(X1.shape)
X = X.at[:, :2].set(
    jnp.divide(X1[:, :2] - jnp.mean(X1[:, :2], 0), jnp.std(X1[:, :2], 0))
)
X = X.at[:, 2].set(X1[:, -1])  # control inputs are not normalised
Y = jnp.divide(Y1, jnp.std(X1[:, :2], 0))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
m_init = jnp.transpose(X[0, :-1, None])
S_init = 0.5 * jnp.eye(state_dim)
controller = RbfController(
    state_dim=state_dim, control_dim=control_dim, num_basis_functions=25
)

R = ExponentialReward(
    state_dim=state_dim,
    t=jnp.divide(jnp.array([0.5, 0.0]) - env.m, env.std),
    W=jnp.diag(jnp.array([0.5, 0.1])),
)
pilco = PILCO(
    (X, Y),
    controller=controller,
    horizon=T,
    reward=R,
    m_init=m_init,
    S_init=S_init,
    trainable_likelihood_variance=False,
)

best_r = 0
all_Rs = []
for i in range(X.shape[0]):
    all_Rs.append(R.compute_reward(X[i, None, :-1], 0.001 * jnp.eye(state_dim))[0])

ep_rewards = []
for i in range(len(X) // T):
    ep_rewards.append(sum(all_Rs[i * T : i * T + T]))

for model in pilco.mgpr.models:
    model.likelihood.transformed_variance.value = softplus_inv(jnp.array(0.05))

r_new = []
for rollouts in range(5):
    pilco.optimize_models()
    pilco.optimize_policy(maxiter=100, restarts=3)

    X_new, Y_new, _, _ = rollout(
        env=env, pilco=pilco, timesteps=T, SUBS=SUBS, render=True
    )

    for i in range(len(X_new)):
        r_new.append(
            R.compute_reward(X_new[i, None, :-1], 0.001 * jnp.eye(state_dim))[0]
        )
    total_r = sum(r_new)
    _, _, r = pilco.predict(m_init, S_init, T)

    print("Total ", total_r, " Predicted: ", r)
    X = jnp.vstack((X, X_new))
    Y = jnp.vstack((Y, Y_new))
    # all_Rs = jnp.vstack((jnp.array(all_Rs), jnp.array(r_new)))
    # ep_rewards = jnp.vstack(
    #    (jnp.array(ep_rewards), jnp.reshape(jnp.array(total_r), (1, 1)))
    # )
    pilco.mgpr.set_data((X, Y))
