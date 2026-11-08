# Examples

Each folder is a program. `main.tsx` computes with `mdml-engine` (a traced run, an estimator session) and draws the
result with `mdml-render`; `figure.svg` is its figure and `tape.json` the tape of the run behind it, which the
site's player can replay. `make readme-examples` runs them all and writes these files (`make readme-examples-check`
fails when they are stale). To add an example, add a folder with a `main.tsx` that follows
`_shared/contract.ts`; nothing else lists it. This file is generated.

# In the README

## Bijections between distributions

A Normal through exp is a log-normal; a coupling layer turns a Gaussian into a banana. The density stays exact.

![A Normal, the map exp, the log-normal it gives, and a coupling layer turning a Gaussian into a banana](bijectors/figure.svg)

[bijectors/main.tsx](bijectors/main.tsx)

## A Gaussian process, tuned by a gradient

grad of the marginal likelihood written as a plain function, and a Gaussian process fitted on that gradient.

![A Gaussian process fitted to noisy sine data, with its uncertainty band](gp/figure.svg)

[gp/main.tsx](gp/main.tsx), tape: [tape.json](gp/tape.json)

## Bayesian logistic regression, approximated four ways

The weight posterior by NUTS, Laplace and two variational fits, and the predicted probability beside the true boundary.

![The weight posterior of a Bayesian logistic regression by NUTS, Laplace and two variational fits, and the predicted probability with the true boundary](logistic/figure.svg)

[logistic/main.tsx](logistic/main.tsx), tape: [tape.json](logistic/tape.json)

## Second order: grad of grad, Hessians and Newton

grad(grad(f)), hessian and hvp; Newton reaches the minimum in 12 steps where gradient descent needs 2662.

![Newton's path against gradient descent on the Rosenbrock function, with the curvature at two points](newton/figure.svg)

[newton/main.tsx](newton/main.tsx), tape: [tape-newton.json](newton/tape-newton.json), tape: [tape-gradient-descent.json](newton/tape-gradient-descent.json)

## An optimiser leaves a trace

Three optimisers on the Rosenbrock function; the paths over its contours are drawn from the recorded runs.

![Three optimisers on the Rosenbrock function: their paths over the contours](optimisers/figure.svg)

[optimisers/main.tsx](optimisers/main.tsx), tape: [tape-gradient-descent.json](optimisers/tape-gradient-descent.json), tape: [tape-momentum.json](optimisers/tape-momentum.json), tape: [tape-adam.json](optimisers/tape-adam.json)

## An agent learning

Q-learning on the cliff-walking gridworld: the return of each episode, drawn from the recorded run.

![Episode returns while Q-learning learns to walk along the cliff](qlearning/figure.svg)

[qlearning/main.tsx](qlearning/main.tsx), tape: [tape.json](qlearning/tape.json)

## Pause, jump back, compare with the truth

A support vector classifier as a session: its decision surface at step 3 and at the end, and the best possible accuracy.

![The decision surface of a support vector classifier at step 3 and at the end](session/figure.svg)

[session/main.tsx](session/main.tsx), tape: [tape.json](session/tape.json)

# More

## A metric curve

The ROC curve and AUC of a support vector classifier on held-out data.

![The ROC curve of the support vector classifier on 300 test points](roc/figure.svg)

[roc/main.tsx](roc/main.tsx), tape: [tape.json](roc/tape.json)

