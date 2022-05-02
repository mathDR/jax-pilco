from numpy.random import gamma
import bayesnewton
from bayesnewton.utils import softplus_inv
import jax.numpy as jnp
import jax.scipy as jsp

import objax


def randomize(model, mean=1, sigma=0.01):
    model.kernel.transformed_lengthscale.assign(
        softplus_inv(mean + sigma * objax.random.normal(model.kernel.lengthscale.shape))
    )
    model.kernel.transformed_variance.assign(
        softplus_inv(mean + sigma * objax.random.normal(model.kernel.variance.shape))
    )
    if isinstance(model.likelihood.transformed_variance, objax.TrainVar):
        model.likelihood.transformed_variance.assign(
            softplus_inv(mean + sigma * objax.random.normal(()))
        )


class MGPR(objax.Module):
    def __init__(
        self,
        data,
        trainable_likelihood_variance: bool = True,
        fixed_parameters: bool = False,
        name=None,
    ):
        super(MGPR, self).__init__()

        self.num_outputs = data[1].shape[1]
        self.num_dims = data[0].shape[1]
        self.num_datapoints = data[0].shape[0]
        self.trainable_likelihood_variance = trainable_likelihood_variance
        self.fixed_parameters = fixed_parameters
        self.create_models(data)
        self.optimizers = []

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
                variance=0.01, fix_variance=~self.trainable_likelihood_variance
            )
            self.models.append(
                bayesnewton.models.VariationalGP(
                    kernel=kern, likelihood=lik, X=data[0], Y=data[1][:, i : i + 1]
                )
            )

    def set_data(self, data):
        X_dim = self.models[0].X.shape
        Y_dim = self.models[0].Y.shape
        for i in range(len(self.models)):
            self.models[i].X = jnp.array(data[0])
            self.models[i].Y = jnp.array(data[1][:, i : i + 1])
            self.models[i].data = [self.models[i].X, self.models[i].Y]
        if (self.models[0].X.shape != X_dim) or (self.models[0].Y.shape != Y_dim):
            # Need to rebuild GP
            self.num_outputs = data[1].shape[1]
            self.num_dims = data[0].shape[1]
            self.num_datapoints = data[0].shape[0]
            self.create_models(data)
            self.optimizers = []

    def optimize(self, maxiter=1000, restarts=1):
        lr_adam = 0.1

        if len(self.optimizers) == 0:  # This is the first call to optimize();
            for i, model in enumerate(self.models):
                opt_hypers = objax.optimizer.Adam(model.vars())
                energy = objax.GradValues(model.energy, model.vars())

                def train_op(en=energy, oh=opt_hypers):
                    dE, E = en()
                    oh(lr_adam, dE)
                    return E

                self.optimizers.append(
                    objax.Jit(
                        objax.Function(
                            train_op,
                            model.vars() + opt_hypers.vars(),
                        )
                    )
                )
            for optimizer in self.optimizers:
                for i in range(maxiter):
                    optimizer()
        else:
            for optimizer in self.optimizers:
                for i in range(maxiter):
                    optimizer()

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
                softplus_inv(best_params["k_lengthscale"])
            )
            model.kernel.transformed_variance.assign(
                softplus_inv(best_params["k_variance"])
            )
            model.likelihood.transformed_variance.assign(
                softplus_inv(best_params["l_variance"])
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

    def K(self, X1, X2=None):
        return jnp.stack([model.kernel.K(X1, X2) for model in self.models])

    @property
    def Y(self):
        return jnp.concatenate([model.Y for model in self.models], axis=1)

    @property
    def X(self):
        return self.models[0].X

    @property
    def lengthscales(self):
        return jnp.stack([model.kernel.lengthscale for model in self.models])

    @property
    def variance(self):
        return jnp.stack([model.kernel.variance for model in self.models])

    @property
    def noise(self):
        return jnp.stack([model.likelihood.variance for model in self.models])

    @property
    def data(self):
        return (self.X, self.Y)
