# A neural additive model, learning its shapes

![The three fitted shape functions moving toward the true ones while the loss falls](learning.gif)

A generalised additive model writes the output as a sum of one function per feature, `y = f0(x0) + f1(x1) + f2(x2)`, so
what it learns can be drawn as three curves. A neural additive model gives each feature its own small network (two
hidden layers of 12 units here) and adds their outputs.

## The data and the model

`additiveModel` generates 200 rows with three uniform features. `y` is the sum of `2 sin(2 pi x0)`, `exp(2 x1)` (shifted)
and `4 (x2 - 1/2)^2` (shifted), plus noise of standard deviation 0.3. The generator hands back the true functions, so the
learned curves can be compared with them. `additiveRegressor` with a `netTerm` per feature is trained on the squared
error by Adam, as a session, for 250 steps.

## The animation

`learning.gif` has one frame for each of 17 steps (0, 1, 2, 4, 7, ... 250), and holds on the last. Each frame asks the
session for the three curves at that step: `probe(step, 'predictor/0/term/<feature>/curve')`. Left to right the panels
are features 0, 1 and 2. The grey curve is the truth, the blue one what the network has learned. Under them is the loss
(on a log scale) up to this step, and a bar that fills as training goes. The frames contain shapes only: the
rasteriser the library uses outside a browser draws no text.

## What to look for

- At step 0 the curves are flat. The simple shapes (the quadratic, the exponential) are found first; the sine takes longer.
- The loss starts at 5.4 and settles at 0.092, which is the noise variance (0.09): the network cannot do better than the
  noise, and it has not overfitted much.
- `figure.svg` is the last frame drawn with axes: the true and learned curves and the loss. The root mean square gap
  between learned and true curve is 0.40, 0.18 and 0.04. Feature 0 is the weakest: ReLU networks draw
  piecewise-linear curves, and a sine needs more segments than 12 units give.

## Run it

`make readme-examples ARGS="gam-neural"` writes `learning.gif`, `figure.svg` and `tape.json`. The program is
[main.tsx](main.tsx). The shared figure code is in [../_shared](../_shared).
