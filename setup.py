from setuptools import setup
from setuptools import find_packages

# Dependencies of PILCO
requirements = ["bayesnewton"]

packages = find_packages(".")
setup(
    name="pilco",
    version="0.1",
    author="Daniel Marthaler",
    author_email="dan.marthaler@gmail.com",
    description=("A JAX implementation of PILCO"),
    license="MIT License",
    keywords="reinforcement-learning model-based-rl gaussian-processes jax",
    url="",
    packages=packages,
    install_requires=requirements,
    include_package_data=True,
    test_suite="tests",
)
