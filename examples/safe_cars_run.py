import jax.numpy as jnp
import objax
from bayesnewton.utils import softplus_inv

from pilco.controllers import RbfController
from pilco.rewards import LinearReward

from linear_cars_env import LinearCars

from safe_pilco_extension.rewards_safe import RiskOfCollision

from safe_pilco_extension.safe_pilco import SafePILCO
from utils import rollout


class Normalised_Env():
    def __init__(self, m, std):
        self.env = LinearCars()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return jnp.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob = self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()


def safe_cars(seed=0):
    T = 25
    th = 0.10
    objax.random.Generator(seed)
    J = 5
    N = 5
    eval_runs = 5
    env = LinearCars()
    # Initial random rollouts to generate a dataset
    X1, Y1, _, _ = rollout(
        env,
        pilco=None,
        timesteps=T,
        verbose=True,
        random=True,
        render=False
    )
    for i in range(1, 5):
        X1_, Y1_, _, _ = rollout(
            env,
            pilco=None,
            timesteps=T,
            verbose=True,
            random=True,
            render=False
        )
        X1 = jnp.vstack((X1, X1_))
        Y1 = jnp.vstack((Y1, Y1_))

    env = Normalised_Env(jnp.mean(X1[:, :4], 0), jnp.std(X1[:, :4], 0))
    X, Y, _, _ = rollout(
        env,
        pilco=None,
        timesteps=T,
        verbose=True,
        random=True,
        render=False
    )
    for i in range(1, J):
        X_, Y_, _, _ = rollout(
            env,
            pilco=None,
            timesteps=T,
            verbose=True,
            random=True,
            render=False
        )
        X = jnp.vstack((X, X_))
        Y = jnp.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim

    m_init = jnp.transpose(X[0, :-1, None])
    S_init = 0.1 * jnp.eye(state_dim)

    controller = RbfController(state_dim=state_dim, control_dim=control_dim,
                               num_basis_functions=40, max_action=0.2)

    #w1 = np.diag([1.5, 0.001, 0.001, 0.001])
    #t1 = np.divide(np.array([3.0, 1.0, 3.0, 1.0]) - env.m, env.std)
    #R1 = ExponentialReward(state_dim=state_dim, t=t1, W=w1)
    # R1 = LinearReward(state_dim=state_dim, W=np.array([0.1, 0.0, 0.0, 0.0]))
    R1 = LinearReward(
        state_dim=state_dim,
        W=jnp.array([1.0 * env.std[0], 0., 0., 0])
    )

    bound_x1 = 1 / env.std[0]
    bound_x2 = 1 / env.std[2]
    low = [-bound_x1-env.m[0]/env.std[0], -bound_x2 - env.m[2] / env.std[2]]
    high = [bound_x1 - env.m[0]/env.std[0], bound_x2 - env.m[2] / env.std[2]]
    B = RiskOfCollision(
        2,
        [-bound_x1-env.m[0]/env.std[0], -bound_x2 - env.m[2] / env.std[2]],
        [bound_x1 - env.m[0]/env.std[0], bound_x2 - env.m[2] / env.std[2]]
    )

    pilco = SafePILCO(
        (X, Y),
        controller=controller,
        mu=-300.0,
        reward_add=R1,
        reward_mult=B,
        horizon=T,
        m_init=m_init,
        S_init=S_init,
        trainable_likelihood_variance=False
    )

    for model in pilco.mgpr.models:
        model.likelihood.transformed_variance.assign(
            softplus_inv(0.001)
        )

    # define tolerance
    new_data = True
    # init = tf.global_variables_initializer()
    #evaluation_returns_full = jnp.zeros((N, eval_runs))
    #evaluation_returns_sampled = jnp.zeros((N, eval_runs))
    #X_eval = []
    for rollouts in range(N):
        print("***ITERATION**** ", rollouts)
        if new_data:
            pilco.optimize_models(maxiter=100)
            new_data = False
        pilco.optimize_policy(maxiter=20, restarts=2)
        # check safety

        predicted_risks = []
        predicted_rewards = []

        for h in range(T):
            m_h, S_h, _ = pilco.predict(m_init, S_init, h, low, high)
            p_risk, _ = B.compute_reward(m_h, S_h)
            predicted_risks.append(p_risk)
            p_rew, _ = R1.compute_reward(m_h, S_h)
            predicted_rewards.append(p_rew)
        overall_risk = 1 - jnp.prod(1.0-jnp.array(predicted_risks))

        print("Predicted episode's return: ", sum(predicted_rewards))
        print("Overall risk ", overall_risk)
        print("Mu is ", pilco.mu)
        print("bound1 ", bound_x1, " bound1 ", bound_x2)

        if overall_risk < th:
            X_new, Y_new, _, _ = rollout(
                env,
                pilco=pilco,
                timesteps=T,
                verbose=True,
                render=False
            )
            new_data = True
            X = jnp.vstack((X, X_new))
            Y = jnp.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))
            if overall_risk < (th/4):
                pilco.mu.assign(0.75 * pilco.mu)

        else:
            X_new, Y_new, _, _ = rollout(
                env,
                pilco=pilco,
                timesteps=T,
                verbose=True,
                render=False
            )
            print("*********CHANGING***********")
            _, _, r = pilco.predict(m_init, S_init, T, low, high)
            print(r)
            # to verify this actually changes, run the reward wrapper before
            # and after on the same trajectory
            pilco.mu.assign(1.5 * pilco.mu.numpy())
            _, _, r = pilco.predict(m_init, S_init, T, low, high)
            print(r)


if __name__ == '__main__':
    safe_cars()
