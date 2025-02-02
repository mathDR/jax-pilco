__all__ = ["DynamicalModel"]

from jax import config

config.update("jax_enable_x64", True)

import gpjax as gpx
from gpjax.lower_cholesky import lower_cholesky
from cola.ops.operators import I_like
from cola.linalg.decompositions.decompositions import Cholesky
from cola.linalg.inverse.inv import (
    inv,
    solve,
)

from jax import Array, grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.typing import ArrayLike
from typing import Optional, Tuple
import optax as ox


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1.0)


class DynamicalModel:
    """The forward model of the system dynamics.

    Currently is a Multiple Gaussian Process regression with prediction on noisy inputs.

    Currently Creates a separate GP for every output dimension but
    TODO: use a multioutput kernel.

    Args:
        kernel (Kernel): The kernel function
        data (JAXArray): The input data. This is either state-action pairs
            $(x_t, u_t)$, or (extension) will be observable-action pairs
            $(y_t, u_t).$
        targets (bool): if True, denotes if the targets are the next observation
            $x_{t+1}$, or if False, state differences $Delta_t = x_{t+1}-x_t.$

    """

    data: gpx.Dataset
    targets: bool
    mean_func: Optional[gpx.mean_functions] = None
    name: Optional[str] = None

    def __init__(
        self,
        data: gpx.Dataset,
        targets: bool,
        mean_func: Optional[gpx.mean_functions] = None,
        name: Optional[str] = None,
    ) -> None:
        self.data = data
        self.targets = targets
        self.mean_func = mean_func

        self.num_outputs: int = data.y.shape[1]
        self.input_dimension: int = data.X.shape[1]
        self.num_datapoints: int = data.X.shape[0]

        if mean_func:
            self.mean_func = mean_func
        else:
            self.mean_func = [gpx.mean_functions.Zero()] * self.num_outputs

        self.create_models()
        self.optimizers: List = []

        if name:
            self.name = name

    def create_models(self) -> None:
        self.models = []
        for i in range(self.num_outputs):
            kern = gpx.kernels.RBF(
                variance=1.0, lengthscale=jnp.ones((self.input_dimension,))
            )
            meanf = self.mean_func[i]
            prior = gpx.gps.Prior(mean_function=meanf, kernel=kern)

            lik = gpx.likelihoods.Gaussian(num_datapoints=self.num_datapoints)

            self.models.append(prior * lik)

    def set_data(self, newdata: gpx.Dataset):  # Type hint added
        self.data = newdata  # Update the data
        self.num_datapoints = newdata.X.shape[0]  # Update num_datapoints
        for i in range(len(self.models)):
            # Recreate the likelihood with the new data's size
            old_prior = self.models[i].prior  # Access the prior
            new_likelihood = gpx.likelihoods.Gaussian(
                num_datapoints=self.num_datapoints
            )
            self.models[i] = old_prior * new_likelihood  # Rebuild the model

        self.optimizers = []  # Reset optimizers since model changed
        # No need to manually set X and Y. gpx.fit handles data association.

    def optimize(self, maxiter: int = 1000):
        key = jr.key(123)

        if not self.optimizers:  # More Pythonic way to check if list is empty
            for model in self.models:
                self.optimizers.append(ox.adam(1e-1))

        for i, model in enumerate(self.models):  # Iterate with index
            opt_posterior, history = gpx.fit(
                model=model,  # Use the current model
                objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
                train_data=gpx.Dataset(
                    self.data.X, self.data.y[:, i].reshape(-1, 1)
                ),  # Use self.data
                optim=self.optimizers[i],  # Use the correct optimizer
                num_iters=maxiter,
                safe=True,
                key=key,
            )
            self.models[
                i
            ] = opt_posterior  # Update the model with the optimized posterior.

    # def predict_on_noisy_inputs(
    #     self, input_mean: ArrayLike, input_covariance: ArrayLike
    # ) -> Tuple[Array, Array, Array]:
    #     """

    #     Args:
    #         input_mean: mean of input (1 x num_inputs)
    #         s: variance of inout (num_inputs x num_inputs)
    #         iK: (K + σ^{2} I)^{−1} (num_outputs x num_data_points x num_data_points)
    #         beta: (K + σ^{2} I)^{−1} y (num_data_points x num_data_points)

    #     Returns:
    #         mean (M) and variance (S) of the output and the input/output covariance (V)

    #     """
    #     iK, beta = self.calculate_factorizations()
    #     return self.predict_given_factorizations(input_mean, input_covaraince, iK, beta)

    # def calculate_factorizations(self):
    #     Kxx = jnp.stack([model.prior.kern.gram(self.data.X) for model in self.models])
    #     noise = jnp.stack(
    #         [
    #             model.prior.jitter + model.likelihood.obs_stddev.value**2
    #             for model in self.models
    #         ]
    #     )
    #     Sigma = Kxx + I_like(Kxx) * noise

    #     L = lower_cholesky(self.K(self.X, self.X) + noise, lower=True)
    #     iK = solve(L, batched_eye)
    #     y = jnp.stack(
    #         [
    #             self.data.y[i] - model.prior.mean_function(self.data.X)
    #             for i, model in enumerate(self.models)
    #         ]
    #     )
    #     beta = solve(Sigma, y, Cholesky())[:, :, 0]
    #     return iK, beta

    # def predict_given_factorizations(self, m, s, iK, beta):
    #     """
    #     Approximate GP regression at noisy inputs via moment matching
    #     IN: mean (m) (row vector) and (s) variance of the state
    #     OUT: mean (M) (row vector), variance (S) of the action
    #          and inv(s)*input-ouputcovariance
    #     """

    #     s = jnp.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
    #     inp = jnp.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

    #     # Calculate M and V: mean and inv(s) times input-output covariance
    #     iL = objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection())(
    #         1 / self.lengthscales
    #     )
    #     iN = inp @ iL
    #     B = iL @ s[0, ...] @ iL + jnp.eye(self.num_dims)

    #     # Redefine iN as in^T and t --> t^T
    #     # B is symmetric so its the same
    #     t = jnp.transpose(
    #         jnp.linalg.solve(B, jnp.transpose(iN, axes=(0, 2, 1))),
    #         axes=(0, 2, 1),
    #     )

    #     lb = jnp.exp(-0.5 * jnp.sum(iN * t, -1)) * beta
    #     tiL = t @ iL
    #     c = self.variance / jnp.sqrt(jnp.linalg.det(B))

    #     M = (jnp.sum(lb, -1) * c)[:, None]
    #     V = (jnp.transpose(tiL, axes=(0, 2, 1)) @ lb[:, :, None])[..., 0] * c[:, None]

    #     # Calculate S: Predictive Covariance
    #     z = objax.Vectorize(
    #         objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection()),
    #         objax.VarCollection(),
    #     )(
    #         1.0 / jnp.square(self.lengthscales[None, :, :])
    #         + 1.0 / jnp.square(self.lengthscales[:, None, :])
    #     )

    #     R = (s @ z) + jnp.eye(self.num_dims)

    #     X = inp[None, :, :, :] / jnp.square(self.lengthscales[:, None, None, :])
    #     X2 = -inp[:, None, :, :] / jnp.square(self.lengthscales[None, :, None, :])
    #     Q = 0.5 * jnp.linalg.solve(R, s)
    #     maha = (X - X2) @ Q @ jnp.transpose(X - X2, axes=(0, 1, 3, 2))

    #     k = jnp.log(self.variance)[:, None] - 0.5 * jnp.sum(jnp.square(iN), -1)
    #     L = jnp.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
    #     S = (
    #         jnp.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
    #         @ L
    #         @ jnp.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
    #     )[:, :, 0, 0]

    #     diagL = jnp.transpose(
    #         objax.Vectorize(
    #             objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection()),
    #             objax.VarCollection(),
    #         )(jnp.transpose(L))
    #     )
    #     S = S - jnp.diag(jnp.sum(jnp.multiply(iK, diagL), [1, 2]))
    #     S = S / jnp.sqrt(jnp.linalg.det(R))
    #     S = S + jnp.diag(self.variance)
    #     S = S - M @ jnp.transpose(M)

    #     return jnp.transpose(M), S, jnp.transpose(V)

    # def centralized_input(self, m: ArrayLike) -> Array:
    #     return self.X - m

    # def K(self, X1: ArrayLike, X2: ArrayLike | None = None):
    #     if X2 is None:
    #         X2 = X1.copy()

    # @property
    # def Y(self):
    #     return tf.concat([model.Y.parameter_tensor for model in self.models], axis=1)

    # @property
    # def X(self):
    #     return self.models[0].X.parameter_tensor

    # @property
    # def lengthscales(self):
    #     return tf.stack(
    #         [model.kern.lengthscales.constrained_tensor for model in self.models]
    #     )

    # @property
    # def variance(self):
    #     return tf.stack(
    #         [model.kern.variance.constrained_tensor for model in self.models]
    #     )

    # @property
    # def noise(self):
    #     return tf.stack(
    #         [model.likelihood.variance.constrained_tensor for model in self.models]
    #     )
