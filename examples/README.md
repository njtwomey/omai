# Examples

Each folder is a program. `main.tsx` computes with `mdml-engine` (a traced run, an estimator session) and draws the
result with `mdml-render`; `figure.svg` is its figure and `tape.json` the tape of the run behind it, which the
site's player can replay. `make readme-examples` runs them all and writes these files (`make readme-examples-check`
fails when they are stale). To add an example, add a folder with a `main.tsx` that follows
`_shared/contract.ts`; nothing else lists it. This file is generated.

# In the README

## A Gaussian process, tuned by a gradient

grad of the marginal likelihood written as a plain function, and a Gaussian process fitted on that gradient.

![A Gaussian process fitted to noisy sine data, with its uncertainty band](gp/figure.svg)

[gp/main.tsx](gp/main.tsx), tape: [tape.json](gp/tape.json)

## An optimiser leaves a trace

Three optimisers on the Rosenbrock function; the paths over its contours are drawn from the recorded runs.

![Three optimisers on the Rosenbrock function: their paths over the contours](optimisers/figure.svg)

[optimisers/main.tsx](optimisers/main.tsx), tape: [tape-gradient-descent.json](optimisers/tape-gradient-descent.json), tape: [tape-momentum.json](optimisers/tape-momentum.json), tape: [tape-adam.json](optimisers/tape-adam.json)

## Pause, jump back, compare with the truth

A support vector classifier as a session: its decision surface at step 3 and at the end, and the best possible accuracy.

![The decision surface of a support vector classifier at step 3 and at the end](session/figure.svg)

[session/main.tsx](session/main.tsx), tape: [tape.json](session/tape.json)

