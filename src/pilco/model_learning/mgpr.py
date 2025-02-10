__all__ = ["DynamicalModel"]

from jax import config

config.update("jax_enable_x64", True)

import gpjax as gpx

from jax import Array, grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.typing import ArrayLike
from typing import List, Optional, Tuple
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
        self.models: List[gpjax.gps.ConjugatePosterior] = []
        # self.lower_cholesky_K = []
        # self.K_inverse_y = []

        for i in range(self.num_outputs):
            kern = gpx.kernels.RBF(
                variance=1.0,
                lengthscale=0.1
                * jnp.ones((self.input_dimension,)),  # makes an ARD kernel by default
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
            # Cache K^-1 y and the lower cholesky of K for use in predict
            # self.lower_cholesky_K[i] =
            # self.K_inverse_y[i] =

    def predict_all_outputs(
        self, test_inputs: Array
    ) -> List[Tuple[ArrayLike, ArrayLike]]:
        """
        Return the gp ouputs (mean and variance) for each output dimension
        """
        predictive_moments = []
        for i in range(self.num_outputs):
            latent_dist = self.models[i].predict(test_inputs, train_data=self.data)
            predictive_dist = self.models[i].likelihood(latent_dist)

            predictive_moments.append(
                (predictive_dist.mean(), predictive_dist.stddev())
            )
        return predictive_moments
