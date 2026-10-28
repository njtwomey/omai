# Optimisation (part 1, mathematical foundations)

Optimisers watched on landscapes: gradient, accelerated, coordinate and proximal methods, Newton and quasi-Newton, constrained problems and duality, stochastic gradients and variance reduction, derivative-free search and global optimisation, as sessions over `optim`, `autodiff` and named objectives. Optimiser races and landscape studies live here; the same optimisers fill the solver slot of every model elsewhere.

## Capabilities

- First-order methods: gradient descent with every step-size rule, momentum, Nesterov, coordinate descent, proximal gradient, mirror descent; options: the objective (quadratic bowls, Rosenbrock, Beale, a logistic loss on data, any `losses` objective), the start, the schedule; log the iterate, the value, the gradient norm.
- Second-order: Newton, line searches (Armijo, Wolfe), BFGS and L-BFGS, trust regions, Gauss-Newton and Levenberg-Marquardt on least squares.
- Constrained: Lagrangians and KKT on small problems, penalty and barrier methods, projected gradient, linear and quadratic programs stepped.
- Stochastic: SGD with batch size and noise, learning-rate schedules, SVRG and SAGA, Adam and friends; the loss-landscape noise shown.
- Derivative-free and global: Nelder-Mead, simulated annealing, CMA-ES and evolution strategies (`search`), Bayesian optimisation (`tune`), random restarts; basins and saddle points.
- Optimal transport: Sinkhorn iterations on two point clouds, the coupling forming.
- Races: several optimisers on one objective from one start with one key, overlaid.

## Examples to make

- Momentum against plain descent on an elongated bowl (the gallery's live-algorithm recipe, generalised).
- Line search: each Armijo backtrack drawn along the search direction.
- The condition number as the knob: convergence rate curves for gradient descent and conjugate gradient.
- SGD's batch size against gradient noise, the iterate's cloud shrinking.
- Sinkhorn: the transport plan as a heatmap converging.
- The optimiser race page: pick any objective, any set of optimisers, the same start.

## Rendering

- Contours with the trajectory (`fields` + `trajectory` panel), value curves, the gradient norm on a log axis, 3-D surfaces with the path.
- Missing: the comparison of several runs on one axis (the race), with per-run colouring; a loss-landscape slice panel (one direction, or two, through the iterate).

## Shared

- `shared/`: the objective catalogue (named landscapes with their minima, plus any model loss on data), the optimiser slot table for the whole library (every `optim` optimiser with its knobs and schedule), the race base.
- From `methods/shared`: the solver slot definitions are this chapter's, used everywhere.

## Problems

- `optimisation/`, `evolutionary/` (strategies) and `search/annealing` meet here; swarm and population search stays in `search-and-planning`.
- A race is one session driving several: the comparison panel and the same-key rule are the prerequisites.
