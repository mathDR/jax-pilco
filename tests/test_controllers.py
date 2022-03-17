from pilco.controllers import LinearController, RbfController, squash_sin
import jax.numpy as jnp
import numpy as np
import objax
import os
import oct2py

oc = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
print(dir_path)
oc.addpath(dir_path)


def test_rbf():
    objax.random.Generator(0)
    d = 3  # Input dimension
    k = 2  # Number of outputs
    b = 100  # basis functions

    # Training Dataset
    X0 = objax.random.uniform((b, d))
    A = objax.random.uniform((d, k))

    Y0 = jnp.sin(X0).dot(A) + 1e-3 * (objax.random.uniform((b, k)) - 0.5)

    rbf = RbfController(d, k, b)
    rbf.set_data((X0, Y0))

    # Generate input
    m = objax.random.uniform((1, d))  # But MATLAB defines it as m'
    s = objax.random.uniform((d, d))

    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, V = rbf.compute_action(m, s, squash=False)

    # convert data to the struct expected by the MATLAB implementation
    lengthscales = np.stack(
        [np.array(model.kernel.lengthscale) for model in rbf.models]
    )
    variance = np.stack([np.array(model.kernel.variance) for model in rbf.models])
    noise = np.stack([np.array(model.likelihood.variance) for model in rbf.models])

    hyp = np.log(
        np.hstack((lengthscales, np.sqrt(variance[:, None]), np.sqrt(noise[:, None])))
    ).T

    gpmodel = oct2py.io.Struct()
    gpmodel.hyp = hyp
    gpmodel.inputs = X0
    gpmodel.targets = Y0

    # Call gp0 in octave
    M_mat, S_mat, V_mat = oc.gp0(gpmodel, m.T, s, nout=3)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    # The following fail because BayesNewton does not have a
    # Squared Exponential Kernel
    assert jnp.allclose(M, M_mat.T, rtol=1e-4)
    assert jnp.allclose(S, S_mat, rtol=1e-4)
    assert jnp.allclose(V, V_mat, rtol=1e-4)


def test_linear():
    objax.random.Generator(0)
    d = 3  # Input dimension
    k = 2  # Output dimension
    # Generate input
    m = objax.random.uniform((1, d))
    s = objax.random.uniform((d, d))
    s = s.dot(s.T)  # Make s positive semidefinite

    W = objax.random.uniform((k, d))
    b = objax.random.uniform((1, k))

    linear = LinearController(d, k)
    linear.W.assign(W)
    linear.b.assign(b)

    M, S, V = linear.compute_action(m, s, squash=False)

    # convert data to the struct expected by the MATLAB implementation
    policy = oct2py.io.Struct()
    policy.p = oct2py.io.Struct()
    policy.p.w = W
    policy.p.b = b.T

    # Call function in octave
    M_mat, S_mat, V_mat = oc.conlin(policy, m.T, s, nout=3)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    # np.testing.assert_allclose(M, M_mat.T, rtol=1e-4)
    assert jnp.allclose(S, S_mat, rtol=1e-4)
    assert jnp.allclose(V, V_mat, rtol=1e-4)


def test_squash():
    objax.random.Generator(0)
    d = 3  # Control dimensions

    m = objax.random.uniform((1, d))
    s = objax.random.uniform((d, d))

    s = s.dot(s.T)
    e = 7.0

    M, S, V = squash_sin(m, s, e)

    M_mat, S_mat, V_mat = oc.gSin(m.T, s, e, nout=3)
    M_mat = jnp.asarray(M_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape

    jnp.allclose(M, M_mat.T, rtol=1e-4)
    jnp.allclose(S, S_mat, rtol=1e-4)
    jnp.allclose(V, V_mat, rtol=1e-4)


if __name__ == "__main__":
    test_rbf()
    test_linear()
    test_squash()
