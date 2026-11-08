# Bijections between distributions

A bijector is an invertible map with a known log-Jacobian. Pushing a distribution through one gives a new distribution
whose density follows from the change of variables, so `logProb` stays exact and sampling is cheap.

## What the program does

- It takes a standard Normal and pushes it through `exp` with `transformed`. The result is the log-normal. Its `logProb`
  at 0.5, 1 and 2 matches the closed form to six decimals (-0.466018, -0.918939, -1.852312).
- It draws 4,000 samples from it. Their mean is 1.631, against exp(1/2) = 1.649 for the true distribution.
- It builds an affine coupling layer, a map that shifts the second coordinate by a function of the first, and pushes a
  2-D Gaussian through it. The result is a banana-shaped density, with an exact `logProb` (-1.8379 at (0, -0.8)).

## What to look for

Every distribution is drawn on the density scale, so you can compare them:
- Top left: the Normal, as a histogram of draws with its density. Top right: the map `y = exp(x)`, which stretches the
  right side and squeezes the left.
- Bottom left: the log-normal, `p(y) = p(x) / |dy/dx|`, as the density that `logProb` gives with a histogram of draws under
  it. The two agree, which is the point of the change of variables. Bottom right: the banana, as a heat map of its density
  with contours of the same density.

The bijectors that map the real line onto each kind of support (`bijectors.unconstrained`) are what let NUTS, Laplace and
variational inference run on constrained parameters.

To run it again: `make readme-examples ARGS="bijectors"`. Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
