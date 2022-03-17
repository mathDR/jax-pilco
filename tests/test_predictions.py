from pilco.models import MGPR
import jax.numpy as jnp
import numpy as np
import objax
import os
import oct2py

oc = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
oc.addpath(dir_path)


def test_predictions():
    objax.random.Generator(0)
    d = 3  # Input dimension
    k = 2  # Number of outputs
    b = 100  # Number of basis functions

    # Training Dataset
    X0 = objax.random.uniform((b, d))
    A = objax.random.uniform((d, k))

    Y0 = jnp.sin(X0).dot(A) + 1e-3 * (objax.random.uniform((b, k)) - 0.5)
    mgpr = MGPR((X0, Y0))

    mgpr.optimize()

    # Generate input
    m = objax.random.uniform((1, d))  # But MATLAB defines it as m'
    s = objax.random.uniform((d, d))
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, V = mgpr.predict_on_noisy_inputs(m, s)

    # Change the dataset and predict again. Just to make sure that we don't cache something we shouldn't.
    X0 = 5 * objax.random.uniform((b, d))
    mgpr.set_data((X0, Y0))

    M, S, V = mgpr.predict_on_noisy_inputs(m, s)

    # convert data to the struct expected by the MATLAB implementation
    lengthscales = np.stack(
        [np.array(model.kernel.lengthscale) for model in mgpr.models]
    )
    variance = np.stack([np.array(model.kernel.variance) for model in mgpr.models])
    noise = np.stack([np.array(model.likelihood.variance) for model in mgpr.models])

    hyp = np.log(
        np.hstack((lengthscales, np.sqrt(variance[:, None]), np.sqrt(noise[:, None])))
    ).T

    gpmodel = oct2py.io.Struct()
    gpmodel.hyp = hyp
    gpmodel.inputs = X0
    gpmodel.targets = Y0

    # Call function in octave
    M_mat, S_mat, V_mat = oc.gp0(gpmodel, m.T, s, nout=3)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    # Probably fails since there is no Squared Exponenital kernel in BayesNewton
    assert jnp.allclose(M, M_mat.T, rtol=1e-4)
    assert jnp.allclose(S, S_mat, rtol=1e-4)
    assert jnp.allclose(V, V_mat, rtol=1e-4)


if __name__ == "__main__":
    test_predictions()
