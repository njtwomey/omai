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

![A Gaussian process on noisy sine data, before and after its hyperparameters are learned, each with its uncertainty band](gp/figure.svg)

[gp/main.tsx](gp/main.tsx), tape: [tape.json](gp/tape.json)

## Bayesian logistic regression, approximated four ways

The weight posterior by NUTS, Laplace and two variational fits, and the predicted probability beside the true boundary.

![The weight posterior of a Bayesian logistic regression by NUTS, Laplace and two variational fits, and the predicted probability with the true boundary](logistic/figure.svg)

[logistic/main.tsx](logistic/main.tsx), tape: [tape.json](logistic/tape.json)

## Second order: grad of grad, Hessians and Newton

grad(grad(f)), hessian and hvp; Newton reaches the minimum in 12 steps where gradient descent needs 2662.

![Newton's path against gradient descent on the Rosenbrock function, with the curvature at two points](newton/figure.svg)

[newton/main.tsx](newton/main.tsx), [newton/README.md](newton/README.md), tape: [tape-gradient-descent.json](newton/tape-gradient-descent.json), tape: [tape-newton.json](newton/tape-newton.json)

## Optimisers, first and second order

Gradient descent, momentum and Adam against BFGS, L-BFGS and Newton on the Rosenbrock function: the paths and the loss from the recorded runs.

![Six optimisers on the Rosenbrock function: their paths over the function and the loss against the step on a log axis](optimisers/figure.svg)

[optimisers/main.tsx](optimisers/main.tsx), [optimisers/README.md](optimisers/README.md), tape: [tape-adam.json](optimisers/tape-adam.json), tape: [tape-bfgs.json](optimisers/tape-bfgs.json), tape: [tape-gradient-descent.json](optimisers/tape-gradient-descent.json), tape: [tape-lbfgs.json](optimisers/tape-lbfgs.json), tape: [tape-momentum.json](optimisers/tape-momentum.json), tape: [tape-newton.json](optimisers/tape-newton.json)

## An agent learning

Q-learning on the cliff-walking gridworld: the return of each episode, drawn from the recorded run.

![Episode returns while Q-learning learns to walk along the cliff](qlearning/figure.svg)

[qlearning/main.tsx](qlearning/main.tsx), tape: [tape.json](qlearning/tape.json)

## The metrics of a classifier

The ROC curve, the precision-recall curve, the precision-recall gain curve and the contingency table of a support vector classifier on held-out data.

![The ROC, precision-recall and precision-recall gain curves of a classifier, and its contingency table with totals and rates](roc/figure.svg)

[roc/main.tsx](roc/main.tsx), [roc/README.md](roc/README.md), tape: [tape.json](roc/tape.json)

## Pause, jump back, compare with the truth

A support vector classifier as a session: its decision surface at step 3 and at the end, and the best possible accuracy.

![The decision surface of a support vector classifier at step 3 and at the end](session/figure.svg)

[session/main.tsx](session/main.tsx), tape: [tape.json](session/tape.json)

# More

## Which arm to pull

Epsilon-greedy, UCB1 and Thompson sampling on a five-armed bandit: the regret curves and how often each arm was pulled.

![Cumulative regret of three bandit strategies over 400 pulls, and the number of pulls of each of the five arms](bandit/figure.svg)

[bandit/main.tsx](bandit/main.tsx), [bandit/README.md](bandit/README.md), tape: [tape-epsilon-greedy.json](bandit/tape-epsilon-greedy.json), tape: [tape-thompson.json](bandit/tape-thompson.json), tape: [tape-ucb1.json](bandit/tape-ucb1.json)

## A Gaussian mixture, step by step

Expectation-maximisation for three overlapping Gaussians: the density, the log likelihood and the soft assignments.

![A Gaussian mixture fitted by EM: the density it gives the plane with the rows coloured by component, the log likelihood over the steps, and the largest responsibility of each row](clustering/figure.svg)

[clustering/main.tsx](clustering/main.tsx), [clustering/README.md](clustering/README.md), tape: [tape.json](clustering/tape.json)

## A decision tree and the regions it cuts

A classification tree on two iris measurements, drawn by the library as its splits and as the regions of the plane.

![A decision tree on iris petal length and width, drawn as a tree of splits beside the regions of the plane it makes](decision-tree/figure.svg)

[decision-tree/main.tsx](decision-tree/main.tsx), [decision-tree/README.md](decision-tree/README.md), tape: [tape.json](decision-tree/tape.json)

## A rolled-up sheet, flattened

A Swiss roll as it is, by PCA and by t-SNE, each point coloured by its true position along the sheet.

![Three panels of a Swiss roll coloured by position along the sheet: the roll seen end-on as a spiral, its PCA projection where layers overlap, and its t-SNE embedding, with the t-SNE objective under them](embedding/figure.svg)

[embedding/main.tsx](embedding/main.tsx), [embedding/README.md](embedding/README.md), tape: [tape.json](embedding/tape.json)

## A neural additive model, from its spline bases

A learnable spline with no smoothness penalty: the weighted basis functions and their sum come up to the points as it trains (learning.gif).

![A spline fitted with no smoothness penalty: its weighted basis functions and their sum against the data, the truth and the classic GAM, and the training loss](gam-neural/figure.svg)

[gam-neural/main.tsx](gam-neural/main.tsx), [gam-neural/README.md](gam-neural/README.md), animation: [learning.gif](gam-neural/learning.gif), tape: [tape.json](gam-neural/tape.json)

## A small neural network, trained and watched

A multilayer perceptron on the two moons: the loss at every step, and the decision surface at the start, the middle and the end.

![The training loss of a small network on two moons, and its decision surface at the start, the middle and the end of training](neural-network/figure.svg)

[neural-network/main.tsx](neural-network/main.tsx), [neural-network/README.md](neural-network/README.md), tape: [tape.json](neural-network/tape.json)

## Principal components of the iris data

PCA found one component per step: the flowers in the first two components, the variance each explains, the loadings.

![PCA of the iris data: the rows in the plane of the first two components coloured by species, the variance explained and the loadings](pca/figure.svg)

[pca/main.tsx](pca/main.tsx), [pca/README.md](pca/README.md), tape: [tape.json](pca/tape.json)

## A logic program and its proof search

A small Prolog family database; the query is answered by SLD resolution and the search tree is drawn from the run.

![The search tree of the query grandparent(tom, W): the answers, and the branches that fail and backtrack](prolog/figure.svg)

[prolog/main.tsx](prolog/main.tsx), [prolog/README.md](prolog/README.md), tape: [tape.json](prolog/tape.json)

## A sampler on a banana

NUTS on a twisted Gaussian: the draws over the true density, the trace of one coordinate, R-hat and effective sample size.

![Draws of four NUTS chains over the contours of the banana density, and the trace of the first coordinate](sampler/figure.svg)

[sampler/main.tsx](sampler/main.tsx), [sampler/README.md](sampler/README.md), tape: [tape.json](sampler/tape.json)

## A SAT solver at work

Six pigeons in five holes as a Boolean formula: DPLL and CDCL both prove it unsatisfiable, and their traces differ.

![Conflicts and decision level against steps for the DPLL and CDCL solvers on the pigeonhole formula with six pigeons and five holes](sat/figure.svg)

[sat/main.tsx](sat/main.tsx), [sat/README.md](sat/README.md), tape: [tape-cdcl.json](sat/tape-cdcl.json), tape: [tape-dpll.json](sat/tape-dpll.json)

## Support vectors and the margin

A kernel support vector classifier with a small and a large C: the score, the boundary, the margins and the support vectors.

![Two support vector classifiers on the two-moons data: the score as a heat map with the boundary and margins as contours, and the support vectors ringed](svm/figure.svg)

[svm/main.tsx](svm/main.tsx), [svm/README.md](svm/README.md), tape: [tape-large-c.json](svm/tape-large-c.json), tape: [tape-small-c.json](svm/tape-small-c.json)

