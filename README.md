# jax-pilco
An Implementation of the Probabilistic Inference for Learning Control (PILCO) in Jax

This repo rewrites [nrontsis/PILCO](https://github.com/nrontsis/PILCO) making use of both jax and Gaussian Processes from [AaltoML/BayesNewton](https://github.com/AaltoML/BayesNewton)

Initially a rewrite is initiated, but eventually we will extend the repo with:
1. Full multi-output GPs (instead of modelling each output independently)
2. Using MCMC for fitting the hyperparameters of the GPs
	a. Will still take the mean of the posterior as the point estimate (just not the MLE)
	b. Will still fit the Gaussian to the posterior
3. Use a Laplace Likelihood to constrain the posterior to be Gaussian.
4. ???

