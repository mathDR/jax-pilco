import jax.numpy as jnp
import objax
from jax.lax import fori_loop

import pandas as pd
import time

from .mgpr import MGPR

from .. import controllers
from .. import rewards


class PILCO(objax.Module):
    def __init__(
        self,
        data,
        num_induced_points=None,
        horizon=30,
        controller=None,
        reward=None,
        m_init=None,
        S_init=None,
        trainable_likelihood_variance=True,
        name=None,
    ):
        super(PILCO, self).__init__()
        # TODO: add back SMGPR?
        self.mgpr = MGPR(data, trainable_likelihood_variance)

        self.state_dim = data[1].shape[1]
        self.control_dim = data[0].shape[1] - data[1].shape[1]
        self.horizon = horizon

        if controller is None:
            self.controller = controllers.LinearController(
                self.state_dim, self.control_dim
            )
        else:
            self.controller = controller

        if reward is None:
            self.reward = rewards.ExponentialReward(self.state_dim)
        else:
            self.reward = reward

        if m_init is None or S_init is None:
            # If the user has not provided an initial state for the rollouts,
            # then define it as the first state in the dataset.
            self.m_init = data[0][0:1, 0 : self.state_dim]
            self.S_init = jnp.diag(jnp.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.S_init = S_init
        self.optimizer = None

    def training_loss(self):
        # This is for tuning controller's parameters
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return -reward

    def optimize_models(self, maxiter=1000, restarts=1):
        """
        Optimize GP models
        """
        self.mgpr.optimize(maxiter=maxiter, restarts=restarts)
        # ToDo: only do this if verbosity is large enough
        lengthscales = {}
        variances = {}
        noises = {}
        for i, model in enumerate(self.mgpr.models):
            lengthscales["GP" + str(i)] = jnp.array(model.kernel.lengthscale)
            variances["GP" + str(i)] = jnp.array([jnp.array(model.kernel.variance)])
            noises["GP" + str(i)] = jnp.array([jnp.array(model.likelihood.variance)])

        print("-----Learned models------")
        print("---Lengthscales---")
        print(pd.DataFrame(data=lengthscales))
        print("---Variances---")
        print(pd.DataFrame(data=variances))
        print("---Noises---")
        print(pd.DataFrame(data=noises))

    def optimize_policy(self, maxiter=1000, restarts=1):
        """
        Optimize controller's parameter's
        """
        lr_adam = 0.1
        if not self.optimizer:
            opt_hypers = objax.optimizer.Adam(self.controller.vars())
            energy = objax.GradValues(self.training_loss, self.controller.vars())

            def train_op(en=energy, oh=opt_hypers):
                dE, E = en()
                oh(lr_adam, dE)
                return E

            self.optimizer = objax.Jit(
                objax.Function(
                    train_op,
                    self.controller.vars() + opt_hypers.vars(),
                )
            )

            for i in range(maxiter):
                self.optimizer()
        else:
            for i in range(maxiter):
                self.optimizer()

        best_parameter_values = [jnp.array(param) for param in self.controller.vars()]
        best_reward = self.compute_reward()

        for restart in range(restarts):
            self.controller.randomize()

            for i in range(maxiter):
                self.optimizer()
            reward = self.compute_reward()
            if reward > best_reward:
                best_parameter_values = [
                    jnp.array(param) for param in self.controller.vars()
                ]
                best_reward = reward

        for i, param in enumerate(self.controller.vars()):
            param.assign(best_parameter_values[i])

    def compute_action(self, x_m):
        return self.controller.compute_action(
            x_m, jnp.zeros([self.state_dim, self.state_dim])
        )[0]

    def predict(self, m_x, s_x, n):
        init_val = (m_x, s_x, 0.0)

        def body_fun(i, v):
            m_x, s_x, reward = v
            return (
                *self.propagate(m_x, s_x),
                jnp.add(reward, jnp.squeeze(self.reward.compute_reward(m_x, s_x)[0])),
            )

        val = fori_loop(0, n, body_fun, init_val)

        m_x, s_x, reward = val
        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = jnp.concatenate([m_x, m_u], axis=1)
        s1 = jnp.concatenate([s_x, s_x @ c_xu], axis=1)
        s2 = jnp.concatenate([jnp.transpose(s_x @ c_xu), s_u], axis=1)
        s = jnp.concatenate([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x
        # TODO: cleanup the following line
        S_x = S_dx + s_x + s1 @ C_dx + C_dx.T @ s1.T

        # While-loop requires the shapes of the outputs to be fixed
        # M_x.set_shape([1, self.state_dim])
        # S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x

    def compute_reward(self):
        return -self.training_loss()

    @property
    def maximum_log_likelihood_objective(self):
        return -self.training_loss()
