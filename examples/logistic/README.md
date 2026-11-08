# Bayesian logistic regression, approximated four ways

A logistic regression gives each class a probability that depends on the features through a few weights. A Bayesian one does
not pick a single set of weights: it keeps a distribution over them (the posterior), so its predictions carry the
uncertainty about the weights. For logistic regression the posterior has no closed form, so it is approximated.

## What the program does

- It draws 60 points from two blobs. The generator knows the true class boundary.
- It writes the model once as a probabilistic program (a Gaussian prior on an intercept and two slopes, and a Bernoulli
  likelihood), then infers the posterior four ways: NUTS, the sampler, as the reference; the Laplace approximation; and
  mean-field and full-rank variational inference.
- It also fits a probit version, which changes one line.

## What to look for

- Left: the 95% region of the two slope weights for each method, over the NUTS draws (grey dots). Full-rank VI is close to
  NUTS. Mean-field VI treats the weights as independent, so its region is too small and has no tilt. Laplace sits on the
  most likely weights, and the posterior is skewed, so its region is off to one side.
- Right: the probability of class 1 from the NUTS draws, as a heat map, with contours where it is 5, 25, 50, 75 and 95%. The
  50% contour is the decision boundary, and the dashed line is the true boundary. The contours fan out away from the data:
  the posterior is less sure there, so the model is less sure of the class.

Posterior mean ± standard deviation of the intercept and the two weights are in the top-level README and in the notes the
program prints. The tape is the full-rank variational run (its ELBO).

To run it again: `make readme-examples ARGS="logistic"`. Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
