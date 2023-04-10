import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Dependencies of PILCO
requirements = ["bayesnewton"]

setuptools.setup(
    name="jax-pilco",
    version="0.1",
    author="Daniel Marthaler",
    author_email="dan.marthaler@gmail.com",
    description=("A JAX implementation of PILCO"),
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-clarity",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    keywords="reinforcement-learning model-based-rl gaussian-processes jax",
    url="https://github.com/mathDR/jax-pilco",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    test_suite="tests",
)
