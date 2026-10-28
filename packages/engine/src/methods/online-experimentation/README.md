# Online experimentation (part 7, application domains)

Experiments run online: A/B test analysis (means, proportions, variance reduction by CUPED, stratification), validity threats and biased samples, the classic pitfalls simulated, sequential testing and safe peeking (always-valid p-values, mSPRT), multi-armed bandits (epsilon-greedy, UCB, Thompson sampling, best-arm identification) and contextual bandits with off-policy evaluation, as sessions over `stats`, `dists`, `agents` (bandits) and `rl`.

## Capabilities

- An A/B test as a stream: users arriving, the estimate and its interval, the p-value over time with the peeking problem shown; options: the effect, the variance, the allocation.
- Variance reduction: CUPED's covariate adjustment shrinking the interval.
- Sequential tests: the always-valid interval against the fixed-horizon one.
- Bandits: regret and the arm's posteriors over time, against A/B's fixed allocation; best-arm identification's stopping.
- Contextual bandits: a linear bandit on synthetic contexts, off-policy estimates (inverse propensity, doubly robust) against the truth.
- Pitfalls: novelty effects, sample-ratio mismatch, interference between units, each planted.

## Examples to make

- Peeking: the p-value crossing 0.05 by chance on a null experiment, many runs overlaid, against the sequential test.
- CUPED: the interval before and after adjustment, the covariate's correlation as the knob.
- Thompson sampling against A/B: cumulative regret and the posteriors over arms.
- Sample-ratio mismatch detected by the chi-squared statistic over time.

## Rendering

- Curves over arrivals with crossing marks, posteriors over arms as bars with intervals, many-runs overlays, bandit episodes.
- Missing: the many-runs overlay panel.

## Shared

- `shared/`: the arrivals stream (shared with `online-learning`), the bandit slot (from `agents`), the planted-effect slot.

## Problems

- `experimentation/` splits: the statistical half to `statistics`, the online half here.
