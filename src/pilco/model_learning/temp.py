from jax import config
import jax.numpy as jnp
import jax.random as jr

from mgpr import DynamicalModel

config.update("jax_enable_x64", True)
import gpjax as gpx

key = jr.key(123)

d = 3  # Input dimension
k = 2  # Number of outputs
b = 100  # Number of basis functions

key, subkey = jr.split(key)
x0 = jr.uniform(key=subkey, shape=(b, d))
key, subkey = jr.split(key)
A = jr.uniform(key=subkey, shape=(d, k))

key, subkey = jr.split(key)
y0 = jnp.sin(x0).dot(A) + 1e-3 * (jr.uniform(key=subkey, shape=(b, k)) - 0.5)

D = gpx.Dataset(X=x0, y=y0)

model = DynamicalModel(data=D, targets=True)

# model.optimize()

# Generate input
key, subkey = jr.split(key)
input_mean = jr.uniform(key=subkey, shape=(1, d))  # But MATLAB defines it as m'
key, subkey = jr.split(key)
s = jr.uniform(key=subkey, shape=(d, d))
input_covariance = s.dot(s.T)  # Make s positive semidefinite

iK, beta = model.calculate_factorizations()

a, b, c = model.predict_given_factorizations(input_mean, input_covariance, iK, beta)
