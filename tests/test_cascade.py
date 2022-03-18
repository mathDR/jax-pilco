from pilco.models.pilco import PILCO
import jax.numpy as jnp
import numpy as np
import objax
import os
import oct2py
import logging

oc = oct2py.Oct2Py(logger=oct2py.get_log())
oc.logger = oct2py.get_log("new_log")
oc.logger.setLevel(logging.INFO)
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
oc.addpath(dir_path)


def test_cascade():
    objax.random.Generator(0)
    d = 2  # State dimenstion
    k = 1  # Controller's output dimension
    b = 100
    horizon = 10
    e = jnp.array(
        [[10.0]]
    )  # Max control input. Set too low can lead to Cholesky failures.

    # Training Dataset
    X0 = objax.random.uniform((b, d + k))
    A = objax.random.uniform((d + k, d))

    Y0 = jnp.sin(X0).dot(A) + 1e-3 * (objax.random.uniform((b, d)) - 0.5)
    pilco = PILCO((X0, Y0))
    pilco.controller.max_action = e

    pilco.optimize_models(restarts=5)
    pilco.optimize_policy(restarts=5)

    # Generate input
    m = objax.random.uniform((1, d))

    s = objax.random.uniform((d, d))
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, reward = pilco.predict(m, s, horizon)

    # convert data to the struct expected by the MATLAB implementation
    policy = oct2py.io.Struct()
    policy.p = oct2py.io.Struct()
    policy.p.w = np.array(pilco.controller.W)
    policy.p.b = np.array(pilco.controller.b).T
    policy.maxU = e

    # convert data to the struct expected by the MATLAB implementation
    lengthscales = np.stack(
        [np.array(model.kernel.lengthscale) for model in pilco.mgpr.models]
    )
    variance = np.stack(
        [np.array(model.kernel.variance) for model in pilco.mgpr.models]
    )
    noise = np.stack(
        [np.array(model.likelihood.variance) for model in pilco.mgpr.models]
    )

    hyp = np.log(
        np.hstack((lengthscales, np.sqrt(variance[:, None]), np.sqrt(noise[:, None])))
    ).T

    dynmodel = oct2py.io.Struct()
    dynmodel.hyp = hyp
    dynmodel.inputs = X0
    dynmodel.targets = Y0

    plant = oct2py.io.Struct()
    plant.angi = np.zeros(0)
    plant.angi = np.zeros(0)
    plant.poli = np.arange(d) + 1
    plant.dyni = np.arange(d) + 1
    plant.difi = np.arange(d) + 1

    # Call function in octave
    M_mat, S_mat = oc.pred(
        policy, plant, dynmodel, m.T, s, horizon, nout=2, verbose=True
    )
    # Extract only last element of the horizon
    M_mat = M_mat[:, -1]
    S_mat = S_mat[:, :, -1]

    assert jnp.allclose(M[0], M_mat.T, rtol=2e-4)
    assert jnp.allclose(S, S_mat, rtol=2e-4)


if __name__ == "__main__":
    test_cascade()
