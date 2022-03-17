import jax.numpy as jnp
import objax
from jax.lax import while_loop

import pandas as pd
import numpy as np
import time

from .mgpr import MGPR

from .. import controllers
from .. import rewards


class PILCO:
    def __init__(
        self,
        data,
        num_induced_points=None,
        horizon=30,
        controller=None,
        reward=None,
        m_init=None,
        S_init=None,
        name=None,
    ):
        super(PILCO, self).__init__()
        # TODO: add back SMGPR
        self.mgpr = MGPR(data)

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

    def optimize_models(self, restarts=1):
        """
        Optimize GP models
        """
        self.mgpr.optimize(restarts=restarts)
        # ToDo: only do this if verbosity is large enough
        lengthscales = {}
        variances = {}
        noises = {}
        for i, model in enumerate(self.mgpr.models):
            lengthscales["GP" + str(i)] = np.array(model.kernel.lengthscale)
            variances["GP" + str(i)] = np.array([np.array(model.kernel.variance)])
            noises["GP" + str(i)] = np.array([np.array(model.likelihood.variance)])

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
        lr_newton = 1.0
        if not self.optimizer:

            @objax.Function.with_vars(self.controller.vars())
            def train_op():
                dE, E = objax.GradValues(
                    self.training_loss, self.controller.vars()
                )()  # compute energy and its gradients w.r.t. hypers
                objax.optimizer.Adam(self.controller.vars())(lr_adam, dE)
                return E

            train_op = objax.Jit(train_op)
            self.optimizer = train_op
            for i in range(maxiter):
                self.optimizer()
        else:
            for i in range(maxiter):
                self.optimizer()

        best_reward = self.compute_reward()

        # TODO: maybe reimplement restarts?

    def compute_action(self, x_m):
        return self.controller.compute_action(
            x_m, jnp.zeros([self.state_dim, self.state_dim])
        )[0]

    def predict(self, m_x, s_x, n):
        loop_vars = [0, m_x, s_x, jnp.array([[0.0]])]

        def cond_fun(val):
            j, m_x, s_x, reward = val
            return j < n

        def body_fun(val):
            j, m_x, s_x, reward = val
            return (
                j + 1,
                *self.propagate(m_x, s_x),
                jnp.add(reward, self.reward.compute_reward(m_x, s_x)[0]),
            )

        while_loop(
            # Termination condition
            cond_fun,
            # Body function
            body_fun,
            loop_vars,
        )
        # while_loop(
        #     # Termination condition
        #     cond_fun,
        #     # Body function
        #     body_fun,
        #     loop_vars,
        # )
        _, m_x, s_x, reward = loop_vars
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
