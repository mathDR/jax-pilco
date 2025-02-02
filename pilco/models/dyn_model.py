from jax import config

config.update("jax_enable_x64", True)

from tinygp import kernels, GaussianProcess
import equinox as eqx
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import optax as ox
from dataset import Dataset


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1.0)


class DynamicalModel(eqx.Module):
    """The forward model of the system dynamics.

    Args:
        kernel (Kernel): The kernel function
        data (Dataset): The input data. This is either state-action pairs
            $(x_t, u_t)$, or (extension) will be observable-action pairs
            $(y_t, u_t).$
        targets (bool): if True, denotes if the targets are the next observation
            $x_{t+1}$, or if False, state differences $Delta_t = x_{t+1}-x_t.$

    """

    data: Dataset
    targets: bool

    def __init__(
        self,
        data: Dataset,
        targets: bool,
    ):
        super(DynamicalModel, self).__init__()
        # assume each element of data is of the same length
        self.num_outputs = data.y[0].shape[1]
        # Assume each element of data is of the same length
        self.num_dims = data.X[0].shape[1]
        self.num_datapoints = data.X.shape[0]

        # Parameters for the GP models
        self.theta = {
            "log_amps": np.zeros(self.num_outputs),
            "log_scales": np.zeros(self.num_outputs),
        }

        self.create_models(data)
        self.optimizers = []

    def create_models(self, data: Dataset):
        self.models = []
        amps = jnp.exp(self.theta["log_amps"])
        scales = jnp.exp(self.theta["log_scales"])
        for i in range(self.num_outputs):
            kern = amps[i] * kernels.ExpSquared(scales[i])

            self.models.append(GaussianProcess(kern, self.data.X))

    def optimize(self, maxiter=1000, restarts=1):
        lr_adam = 0.1

        if len(self.optimizers) == 0:  # This is the first call to optimize();
            for i, model in enumerate(self.models):
                self.optimizers.append(optax.lbfgs())
            for optimizer in self.optimizers:
                value_and_grad = optax.value_and_grad_from_state(
                    -model.log_probability(self.data.y[:, i])
                )
                params = jnp.array(
                    [self.theta["log_amps"][i], self.theta["log_scales"][i]]
                )
                opt_state = optimizer.init(params)
                for _ in range(maxiter):
                    value, grad = value_and_grad(params, state=opt_state)
                    updates, opt_state = solver.update(
                        grad,
                        opt_state,
                        params,
                        value=value,
                        grad=grad,
                        value_fn=-model.log_probability,
                    )
                    params = optax.apply_updates(params, updates)
        else:
            for optimizer in self.optimizers:
                value_and_grad = optax.value_and_grad_from_state(
                    -model.log_probability(self.data.y[:, i])
                )
                params = jnp.array(
                    [self.theta["log_amps"][i], self.theta["log_scales"][i]]
                )
                opt_state = optimizer.init(params)
                for _ in range(maxiter):
                    value, grad = value_and_grad(params, state=opt_state)
                    updates, opt_state = solver.update(
                        grad,
                        opt_state,
                        params,
                        value=value,
                        grad=grad,
                        value_fn=-model.log_probability,
                    )
                    params = optax.apply_updates(params, updates)

        kern = params[0] * kernels.ExpSquared(params[1])

        self.models.append(GaussianProcess(kern, self.data.X))

        for model, optimizer in zip(self.models, self.optimizers):
            best_params = {
                "k_lengthscale": model.kernel.lengthscale,
                "k_variance": model.kernel.variance,
                "l_variance": model.likelihood.variance,
            }
            best_loss = model.energy()
            for restart in range(restarts):
                randomize(model)
                for i in range(maxiter):
                    optimizer()
                loss = model.energy()
                if loss < best_loss:
                    best_params["k_lengthscale"] = model.kernel.lengthscale
                    best_params["k_variance"] = model.kernel.variance
                    best_params["l_variance"] = model.likelihood.variance
                    best_loss = loss
            model.kernel.transformed_lengthscale.assign(
                inverse_softplus(best_params["k_lengthscale"])
            )
            model.kernel.transformed_variance.assign(
                inverse_softplus(best_params["k_variance"])
            )
            model.likelihood.transformed_variance.assign(
                inverse_softplus(best_params["l_variance"])
            )

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

    def calculate_factorizations(self):
        K = self.K(self.X, self.X)
        batched_eye = jnp.expand_dims(jnp.eye(jnp.shape(self.X)[0]), axis=0).repeat(
            self.num_outputs, axis=0
        )
        L = jsp.linalg.cho_factor(
            K + self.noise[:, None, None] * batched_eye, lower=True
        )
        iK = jsp.linalg.cho_solve(L, batched_eye)
        Y_ = jnp.transpose(self.Y)[:, :, None]
        beta = jsp.linalg.cho_solve(L, Y_)[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = jnp.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = jnp.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection())(
            1 / self.lengthscales
        )
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + jnp.eye(self.num_dims)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = jnp.transpose(
            jnp.linalg.solve(B, jnp.transpose(iN, axes=(0, 2, 1))),
            axes=(0, 2, 1),
        )

        lb = jnp.exp(-0.5 * jnp.sum(iN * t, -1)) * beta
        tiL = t @ iL
        c = self.variance / jnp.sqrt(jnp.linalg.det(B))

        M = (jnp.sum(lb, -1) * c)[:, None]
        V = (jnp.transpose(tiL, axes=(0, 2, 1)) @ lb[:, :, None])[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        z = objax.Vectorize(
            objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection()),
            objax.VarCollection(),
        )(
            1.0 / jnp.square(self.lengthscales[None, :, :])
            + 1.0 / jnp.square(self.lengthscales[:, None, :])
        )

        R = (s @ z) + jnp.eye(self.num_dims)

        X = inp[None, :, :, :] / jnp.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :] / jnp.square(self.lengthscales[None, :, None, :])
        Q = 0.5 * jnp.linalg.solve(R, s)
        maha = (X - X2) @ Q @ jnp.transpose(X - X2, axes=(0, 1, 3, 2))

        k = jnp.log(self.variance)[:, None] - 0.5 * jnp.sum(jnp.square(iN), -1)
        L = jnp.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (
            jnp.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
            @ L
            @ jnp.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
        )[:, :, 0, 0]

        diagL = jnp.transpose(
            objax.Vectorize(
                objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection()),
                objax.VarCollection(),
            )(jnp.transpose(L))
        )
        S = S - jnp.diag(jnp.sum(jnp.multiply(iK, diagL), [1, 2]))
        S = S / jnp.sqrt(jnp.linalg.det(R))
        S = S + jnp.diag(self.variance)
        S = S - M @ jnp.transpose(M)

        return jnp.transpose(M), S, jnp.transpose(V)

    def centralized_input(self, m):
        return self.X - m
