# Statistics (part 1, mathematical foundations)

Statistics watched: estimators and intervals, hypothesis tests and their errors, Bayesian updating, resampling, regression diagnostics, survival analysis and experimental design, as sessions over `stats`, `dists` and `datasets`. The methods are studies of behaviour (coverage, power, the bootstrap distribution forming), not the tests themselves, which are library functions.

## Capabilities

- Estimation: an estimator's sampling distribution by simulation (bias, variance, consistency), confidence intervals and their coverage; options: the estimator, the distribution, n.
- Hypothesis testing: power curves, Type I and II error by simulation, multiple testing corrections, sequential tests; options: the test, the effect size, the design.
- Bayesian statistics: conjugate updating one observation at a time (beta-binomial, normal-normal, gamma-Poisson), credible intervals, Bayes factors; options: the prior, the data.
- Resampling: the bootstrap and jackknife distributions forming, permutation tests; options: the statistic, the resample count.
- Regression analysis: residual diagnostics, leverage and influence, multicollinearity on a fitted linear model (from `supervised-learning`).
- Survival analysis: Kaplan-Meier and Nelson-Aalen estimators as the data arrive, the log-rank test, Cox proportional hazards fitted iteratively; options: censoring, the hazard.
- Experimental design: randomisation, blocking, factorial designs, and causal effects from observational data (with `causal-inference`).

## Examples to make

- Coverage: a hundred confidence intervals drawn as the samples arrive, the ones that miss highlighted.
- Power: the test's rejection rate against the effect size, the sample size as the knob.
- A beta posterior updating coin by coin, with the prior as the knob.
- The bootstrap distribution of a median filling in, resample by resample.
- Kaplan-Meier stepping down as events arrive, censored subjects marked.

## Rendering

- Curves with bands, histograms against densities, a posterior over a grid (`posteriorGrid`), step functions with marks.
- Missing: an interval strip (many intervals as horizontal bars against a truth line); a table of test results with the decision rule shown.

## Shared

- `shared/`: the sampling-distribution study (repeat an estimator over fresh samples, a base for coverage and power), the conjugate updating base.
- From `methods/shared`: sweeps, the data slot with truth.

## Problems

- `survival/` and the statistical half of `experimentation/` move here.
- Studies by repetition need many runs per step (a hundred samples per point of a power curve): the sweep base must keep the inner work off the tape.
