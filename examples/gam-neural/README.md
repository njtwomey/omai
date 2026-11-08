# A neural additive model, from its spline bases

This is the model of the proof of concept "Generalised additive model, neural" in `apps/pocs`, run without its page. A
generalised additive model fits `y` as a sum of smooth functions of the features. Here there is one feature, and its
function is a learnable cubic B-spline: a sum of basis functions, each multiplied by a coefficient. The coefficients and
the positions of the knots are trained by an optimiser, as the weights of a network are.

## What the program does

- It draws 16 noisy points from a sine curve (`sineCurve`, noise 0.2) and keeps the true curve for comparison.
- It trains the spline with 11 interior knots (learnable) and a smoothness penalty of zero, for 600 steps. With no
  penalty nothing stops the curve from following the points, and with 15 coefficients for 16 points it can pass through
  all of them.
- It asks the training session, at 21 chosen steps, for the curve, for the weighted basis functions (the pieces that add
  up to the curve) and for the knots. These come from the tape of the run.
- For comparison it fits a classic GAM of the same data, whose smoothing is chosen by REML, with 7 bases.

## What to look for

- `learning.gif` shows the pale orange weighted bases and their sum in red. At step 0 the sum is almost flat. As the
  steps go on the bases grow and move, the knots (the red ticks) shift, and the red curve is pulled up to the points.
  The loss under the plot falls from 0.4 to about 1e-6.
- `figure.svg` shows the start, the end, and the loss. At the end the red curve goes through every point (the root mean
  square distance from the points is 0.003, where the noise is 0.2) and is wiggly between them. The blue classic GAM
  is smooth and stays near the dotted true curve (the red one is 0.27 away from it). That is what a penalty is for.

The tape of the run is in `tape.json`. To run it again: `make readme-examples ARGS="gam-neural"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg). Animation: [learning.gif](learning.gif).
