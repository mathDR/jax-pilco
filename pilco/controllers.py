import jax.numpy as jnp
import equinox as eqx
from jax import Array
from jax.typing import ArrayLike
from typing import Optional, Tuple
import gpjax as gpx
from .models import MGPR
from jax import random


def inverse_softplus(x: ArrayLike) -> Array:
    return jnp.log(jnp.exp(x) - 1.0)


def squash_sin(
    controls_mean: ArrayLike,
    controls_covariance: ArrayLike,
    max_action: Optional[ArrayLike] = None,
) -> Tuple[Array, Array, Array]:
    """
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean and covariance of the control input, along with max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    """
    k = jnp.shape(controls_mean)[1]
    if max_action is None:
        max_action = jnp.ones((1, k))  # squashes in [-1,1] by default
    else:
        max_action = max_action * jnp.ones((1, k))

    M = (
        max_action
        * jnp.exp(-0.5 * jnp.diag(controls_covariance))
        * jnp.sin(controls_mean)
    )

    lq = -0.5 * (
        jnp.diag(controls_covariance)[:, None] + jnp.diag(controls_covariance)[None, :]
    )
    q = jnp.exp(lq)
    mT = jnp.transpose(controls_mean, (1, 0))
    S = (jnp.exp(lq + controls_covariance) - q) * jnp.cos(mT - controls_mean) - (
        jnp.exp(lq - controls_covariance) - q
    ) * jnp.cos(mT + controls_mean)
    S = 0.5 * max_action * jnp.transpose(max_action, (1, 0)) * S

    C = max_action * jnp.diag(
        jnp.exp(-0.5 * jnp.diag(controls_covariance)) * jnp.cos(controls_mean)
    )

    return M, S, C.reshape((k, k))


class LinearController(eqx.Module):
    def __init__(self, state_dim: int, control_dim: int, max_action: float = 1.0):
        objax.random.Generator(0)
        self.W = objax.TrainVar(objax.random.uniform((control_dim, state_dim)))
        self.b = objax.TrainVar(objax.random.uniform((1, control_dim)))
        self.max_action = max_action

    def compute_action(self, m, s, squash=True):
        """
        Simple affine action:  M <- W(m-t) - b
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        """

        WT = jnp.transpose(self.W.value, (1, 0))
        M = m @ WT + self.b.value  # mean output
        S = self.W.value @ s @ WT  # output variance
        V = WT  # input output covariance
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def randomize(self):
        mean = 0
        sigma = 1
        self.W.assign(mean + sigma * objax.random.normal(self.W.shape))
        self.b.assign(mean + sigma * objax.random.normal(self.b.shape))


class RbfController(MGPR):
    """
    An RBF Controller implemented as a deterministic GP
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 5.3.2.
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        num_basis_functions: int,
        max_action: float = 1.0,
        key: Array,
    ):
        if not dtypes.issubdtype(key.dtype, dtypes.prng_key):
            raise TypeError("New-style typed JAX PRNG keys required")

        key, subkey1, subkey2 = random.split(key)

        MGPR.__init__(
            self,
            [
                random.normal(subkey1, shape=(num_basis_functions, state_dim)),
                0.1 * random.normal(subkey2, shape=(num_basis_functions, control_dim)),
            ],
            fixed_parameters,
        )

        self.fixed_parameters = fixed_parameters
        self.max_action = max_action

    def create_models(self, data: Dataset):
        self.models = []
        for i in range(self.num_outputs):
            kern = gpx.RBF(lengthscale=jnp.ones((data[0].shape[1],)), variance=1.0)
            meanf = gpx.mean_functions.Zero()
            prior = gpx.gps.Prior(mean_function=meanf, kernel=kern)

            lik = gpx.likelihoods.Gaussian(
                obs_stddev, 1e-4, num_datapoints=len(data[0])
            )
            # bayesnewton.likelihoods.Gaussian(
            #     variance=1e-4, fix_variance=self.fixed_parameters
            # )
            posterior = prior * lik
            # self.models.append(
            #     bayesnewton.models.VariationalGP(
            #         kernel=kern, likelihood=lik, X=data[0], Y=data[1][:, i : i + 1]
            #     )
            # )
            self.models.append(posterior)

    def compute_action(
        self, state_mean: ArrayLike, state_covariance: ArrayLike, squash: bool = True
    ) -> Tuple[Array < Array, Array]:
        """
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        """
        iK, beta = self.calculate_factorizations()
        M, S, V = self.predict_given_factorizations(
            state_mean, state_covariance, 0.0 * iK, beta
        )
        S = S - jnp.diag(self.variance - 1e-6)
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def randomize(self, key: Array):
        print("Randomizing controller")
        for m in self.models:
            m.X = jnp.array(objax.random.normal(m.X.shape))
            m.Y = jnp.array(0.1 * self.max_action * objax.random.normal(m.Y.shape))

            mean = 1.0
            sigma = 0.1
            m.kernel.transformed_lengthscale.assign(
                softplus_inv(
                    mean + sigma * objax.random.normal(m.kernel.lengthscale.shape)
                )
            )
