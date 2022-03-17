from pilco.rewards import ExponentialReward
import jax.numpy as jnp
import numpy as np
import objax
import os
import oct2py

oc = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
oc.addpath(dir_path)


def test_exponential_reward():
    """
    Test reward function by comparing to reward.m
    """
    objax.random.Generator(0)

    k = 2  # state dim
    m = objax.random.uniform((1, k))  # But MATLAB defines it as m'
    s = objax.random.uniform((k, k))
    s = s.dot(s.T)

    reward = ExponentialReward(k)
    W = np.array(reward.W)
    t = np.array(reward.t)

    M, S = reward.compute_reward(m, s)

    M_mat, _, _, S_mat = oc.reward(m.T, s, t.T, W, nout=4)

    assert jnp.allclose(M, M_mat)
    assert jnp.allclose(S, S_mat)


if __name__ == "__main__":
    test_exponential_reward()
