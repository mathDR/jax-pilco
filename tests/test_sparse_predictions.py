from pilco.models import SMGPR
import jax.numpy as jnp
import objax
import os
import oct2py

octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
octave.addpath(dir_path)

from gpflow import config

float_type = config.default_float()


def test_sparse_predictions():
    objax.random.Generator(0)
    d = 3  # Input dimension
    k = 2  # Number of outputs

    # Training Dataset
    X0 = objax.random.uniform((100, d + k))
    A = objax.random.uniform((d + k, d))

    Y0 = jnp.sin(X0).dot(A) + 1e-3 * (objax.random.uniform((100, k)) - 0.5)
    smgpr = SMGPR((X0, Y0), num_induced_points=30)

    smgpr.optimize()

    # Generate input
    m = objax.random.uniform((1, d))  # But MATLAB defines it as m'
    s = objax.random.uniform((d, d))
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, V = smgpr.predict_on_noisy_inputs(m, s)

    # convert data to the struct expected by the MATLAB implementation
    lengthscales = jnp.stack([model.kernel.lengthscales for model in smgpr.models])
    variance = jnp.stack([model.kernel.variance for model in smgpr.models])
    noise = jnp.stack([model.likelihood.variance for model in smgpr.models])

    hyp = jnp.log(
        jnp.hstack(
            (lengthscales, jnp.sqrt(variance[:, None]), jnp.sqrt(noise[:, None]))
        )
    ).T

    gpmodel = oct2py.io.Struct()
    gpmodel.hyp = hyp
    gpmodel.inputs = X0
    gpmodel.targets = Y0
    gpmodel.induce = smgpr.Z.numpy()

    # Call function in octave
    M_mat, S_mat, V_mat = octave.gp1(gpmodel, m.T, s, nout=3)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    assert jnp.allclose(M, M_mat.T, rtol=1e-4)
    assert jnp.allclose(S, S_mat, rtol=1e-4)
    assert jnp.allclose(V, V_mat, rtol=1e-4)


if __name__ == "__main__":
    test_sparse_predictions()
