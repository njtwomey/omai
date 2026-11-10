# evidence/pmcmc-ar1

Particle marginal Metropolis-Hastings and particle Gibbs (with and without ancestor sampling) against the exact posterior of the AR(1) coefficient, from the Kalman filter's marginal likelihood on a grid.

Run on 2026-11-10 at commit 908edcb3; seed 0, budget scale 1 (iterations 3000, chains 4), 38.3 s.

**Passed**: every chain reproduces the exact posterior mean (within four Monte Carlo standard errors) and spread (within 30 percent)

AR(1) benchmark data (12 observations), uniform prior on phi in (-0.99, 0.99), known noise. Exact posterior of phi from the Kalman marginal likelihood on a grid: mean 0.8476, sd 0.0876. 3000 iterations each, first 20 percent discarded; z is the mean error in Monte Carlo standard errors (the sd over the square root of the chain's ESS).

| method                                             | mean  | exact mean | z     | sd    | exact sd | ESS | accept | path refresh | verdict |
| -------------------------------------------------- | ----- | ---------- | ----- | ----- | -------- | --- | ------ | ------------ | ------- |
| PMMH, 50 particles, random walk                    | 0.856 | 0.848      | 1.51  | 0.082 | 0.088    | 246 | 0.15   | -            | ok      |
| PMMH, 50 particles, adaptive                       | 0.861 | 0.848      | 2.48  | 0.082 | 0.088    | 261 | 0.22   | -            | ok      |
| particle Gibbs, ancestor sampling, 24 particles    | 0.845 | 0.848      | -0.80 | 0.087 | 0.088    | 895 | 0.31   | 1.00         | ok      |
| particle Gibbs, no ancestor sampling, 24 particles | 0.846 | 0.848      | -0.56 | 0.092 | 0.088    | 913 | 0.32   | 0.38         | ok      |

Path refresh is the fraction of times at which a conditional SMC sweep changed the state path: ancestor sampling keeps it near 1 with few particles, while without it the early path stays stuck to the reference (the path degeneracy that ancestor sampling removes).
