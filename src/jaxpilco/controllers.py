import jax.numpy as jnp
import objax

from jaxpilco.models import MGPR


def squash_sin(m, s, max_action=None):
    """
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean (m) and variance(s) of the control input, max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    """
    k = jnp.shape(m)[1]
    if max_action is None:
        max_action = jnp.ones((1, k))  # squashes in [-1,1] by default
    else:
        max_action = max_action * jnp.ones((1, k))

    M = max_action * jnp.exp(-0.5 * jnp.diag(s)) * jnp.sin(m)

    lq = -0.5 * (jnp.diag(s)[:, None] + jnp.diag(s)[None, :])
    q = jnp.exp(lq)
    mT = jnp.transpose(m, (1, 0))
    S = (jnp.exp(lq + s) - q) * jnp.cos(mT - m) - (jnp.exp(lq - s) - q) * jnp.cos(
        mT + m
    )
    S = 0.5 * max_action * jnp.transpose(max_action, (1, 0)) * S

    C = max_action * objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection())(
        jnp.exp(-0.5 * jnp.diag(s)) * jnp.cos(m)
    )
    return M, S, C.reshape((k, k))


class LinearController(objax.Module):
    def __init__(self, state_dim, control_dim, max_action=1.0):
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
        state_dim,
        control_dim,
        num_basis_functions,
        max_action=1.0,
        fixed_parameters=False,
    ):
        MGPR.__init__(
            self,
            [
                objax.random.normal((num_basis_functions, state_dim)),
                0.1 * objax.random.normal((num_basis_functions, control_dim)),
            ],
            fixed_parameters,
        )

        self.fixed_parameters = fixed_parameters
        self.max_action = max_action

    def create_models(self, data):
        self.models = []
        for i in range(self.num_outputs):

            kern = bayesnewton.kernels.Matern72(
                variance=1.0,
                lengthscale=jnp.ones((data[0].shape[1],)),
                fix_variance=self.fixed_parameters,
                fix_lengthscale=self.fixed_parameters,
            )

            lik = bayesnewton.likelihoods.Gaussian(
                variance=1e-4, fix_variance=self.fixed_parameters
            )
            self.models.append(
                bayesnewton.models.VariationalGP(
                    kernel=kern, likelihood=lik, X=data[0], Y=data[1][:, i : i + 1]
                )
            )

    def compute_action(self, m, s, squash=True):
        """
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        """
        iK, beta = self.calculate_factorizations()
        M, S, V = self.predict_given_factorizations(m, s, 0.0 * iK, beta)
        S = S - jnp.diag(self.variance - 1e-6)
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def randomize(self):
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
