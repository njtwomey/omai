# Probabilistic inference (part 4, principles of learning)

Inference watched: graphical models and exact inference (variable elimination, belief propagation on trees, junction trees), Markov chain Monte Carlo (Metropolis, Gibbs, HMC, NUTS, slice, with diagnostics), variational inference (mean field, CAVI, stochastic VI, black-box VI, normalising-flow posteriors), message-passing approximations (loopy BP, expectation propagation), probabilistic programming, skill rating and model-based machine learning, as sessions over `pgm`, `sampling`, `probprog` and `dists`. The hidden Markov model, topic models and other named probabilistic models sit here too.

## Capabilities

- Exact inference: elimination order chosen and factors multiplied step by step; belief propagation's messages passing on a tree; the junction tree built.
- MCMC: every kernel in `sampling` as the sampler slot on any log density (named targets, a posterior of a small model, a probabilistic program); diagnostics (trace, autocorrelation, R-hat, effective sample size) on the tape; options: the target, the kernel and its knobs, chains, warm-up.
- Variational: CAVI on a mixture, SVI with batches, the ELBO rising, the posterior against MCMC on the same model.
- Message passing: loopy BP on a grid oscillating or converging; EP's site updates.
- Probabilistic programs: a `probprog` program as the model slot, run by any engine.
- Named models: the hidden Markov model by Baum-Welch (exists as `sequence/hmm`), topic models (LDA by Gibbs and by VI), conjugate models, rating systems (Elo, TrueSkill), linear-Gaussian latent models.

## Examples to make

- Metropolis against HMC on a banana density, the same budget, the chains side by side.
- NUTS's trajectory tree on a 2-D Gaussian at the viewed step.
- Belief propagation's messages on a small tree, one message per step.
- CAVI on a Gaussian mixture against EM and against Gibbs: three sessions, one data set.
- TrueSkill updating after each match of a small league.

## Rendering

- Chains in the plane (`trajectory`), trace plots, posterior grids (`posteriorGrid`), state sequences, graphical models with messages (structure view), ELBO curves.
- Missing: a message-on-edge overlay for the structure view; the chain comparison panel.

## Shared

- `shared/`: the target slot (named densities, posteriors of small models, programs) shared with `probability`; the sampler slot table (every `sampling` kernel); the diagnostics record.

## Problems

- `sampling/mcmc`, `sequence/hmm` and `models/` move here. Probabilistic programs as the model slot need the program's latent names as paths on the tape: the contract becomes a function of the model.
