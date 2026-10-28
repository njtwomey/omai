# Probability (part 1, mathematical foundations)

Probability simulated: sampling and simulation (inverse transform, rejection, Box-Muller, Monte Carlo integration, importance sampling, the gradient estimators), limit theorems, stochastic processes and stochastic calculus, and the classic problems, as sessions over `random`, `dists`, `sampling` and `stochastic`. Markov chain Monte Carlo as inference lives in `probabilistic-inference`.

## Capabilities

- Sampling by transformation: inverse transform, rejection, Box-Muller, Gumbel-max, from any `dists` family; options: the target, the proposal, the sample count; log acceptance, the running estimate and its error.
- Monte Carlo estimation: plain, importance, control variates, antithetic, stratified; options: the integrand, the proposal; log the estimate, the variance, the effective sample size.
- Limit theorems: the law of large numbers and the central limit theorem by simulation; concentration bounds against empirical tails; options: the distribution, the sample size schedule.
- Stochastic processes: random walks, Markov chains (stationary distribution, mixing), Poisson and renewal processes, branching, queues, Brownian motion and its paths, Itô integrals, SDEs by Euler-Maruyama and Milstein, Ornstein-Uhlenbeck, Fokker-Planck against histograms; options: the process and its parameters, the discretisation, the number of paths.
- Classic problems simulated against their exact answer: birthday, Monty Hall, gambler's ruin, coupon collector, the dishonest casino.

## Examples to make

- Rejection sampling with the proposal envelope and the accepted and rejected points accumulating.
- Importance sampling with a mismatched proposal: the weights' distribution and the effective sample size collapsing.
- The central limit theorem: the sum's histogram sharpening into a Gaussian as n grows, for any family.
- Brownian paths with the exact mean and variance envelope; an SDE's paths against Fokker-Planck.
- A Markov chain's distribution converging to stationarity, the total variation on the tape.

## Rendering

- Histograms and densities against exact curves (`statistical` charts), paths bundles (`stochastic/diffusionPaths` exists), the running estimate with its error band.
- Missing: a sample as points accumulating in the plane with acceptance colouring at the viewed step (a `rows`-like panel for samples, not training rows).

## Shared

- `shared/`: the target slot (any `dists` family or a named log density), the proposal slot, the estimate-with-error record.
- From `methods/shared`: the procedure base, sweeps (an estimate against the sample size).

## Problems

- `probability/transformSampling`, `stochastic/diffusionPaths` and `monte-carlo/` move here; MCMC (`sampling/mcmc`) moves to `probabilistic-inference`.
- Many paths on the tape: a bundle of 1,000 paths at 1,000 steps is a tensor per step; what detail to keep.

## What is here now

- `transformSampling` (built this week); `stochastic/diffusionPaths` and `sampling/mcmc` join or leave this chapter as the mapping in `methods/README.md` says.
