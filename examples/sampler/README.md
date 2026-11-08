# A sampler on a banana

Haario's banana is a Gaussian bent into a curve: `x1 ~ N(0, 3^2)` and `x2 | x1 ~ N(0.1 x1^2 - 0.9, 1)`. Its density is
known exactly, which makes it a good test of whether a sampler explores a curved shape. The example runs the No-U-Turn
sampler (NUTS) from `mdml-engine/sampling` on it.

![Draws of four chains over the true density contours, and the trace of x[0]](figure.svg)

## The program

`banana({ curvature: 0.1, scale: 3 })` is the target, with its gradient. `nuts` runs four chains, each with 500 warm-up
iterations in which the step size is tuned and then 500 kept draws. `summary` computes the diagnostics from the draws.

## What to look for

- Left: every tenth draw of each chain (a colour each), over contours of the true density. The contours are where the
  density is 0.6, 0.14 and 0.01 of its peak, which for a Gaussian would be one, two and three standard deviations out. The
  draws follow the curve, including the arms.
- Right: the trace of `x[0]` for the four chains. They overlap and none drifts, which is what a converged run looks like.

## Printed values

- `x[0]`: mean -0.08 (true 0), standard deviation 2.86 (true 3), R-hat 1.009, bulk effective sample size 339 of 2000.
- `x[1]`: mean -0.08 (true 0), standard deviation 1.46 (true 1.62), R-hat 1.014, bulk effective sample size 527.

R-hat compares the chains with one another; values near 1 mean they agree. The effective sample size is how many
independent draws the 2000 correlated ones are worth.

## Run it

`make readme-examples ARGS="sampler"` writes `figure.svg` and `tape.json`. The program is [main.tsx](main.tsx).
