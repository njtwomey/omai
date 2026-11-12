# EP quality, second round: multimodal posteriors, loops, stochastic volatility, evidence (2026-11-11, `agent-pp193-multimodal`)

Four of the places where EP is known to be wrong or narrow, each measured before and after on the same seeds, what was kept and what was
not. Every number below was produced by the scripts of this session (the data are simulated from stated truths, the seeds are
stated; "coverage" is the share of true values inside the 90% interval, 0.9 being right). The machine was loaded by other agents, so
times are only comparative within a table.

## 1. Multimodal posteriors: relabelling starts and a canonical read-out (`epMultiStartMethod`)

**What the existing multi-start did on the basic changepoint program** (`ep-program-changepoint`: 80 steps, stay 0.95, rates 3, 9, 22
with the broad gamma priors of the page, seeds 1 to 24, so 72 intervals): plain EP covers the true rates in 0.778 of the intervals (the
root mean square of error over sd is 4.55). The multi-start of section 15 of `ep-changelog.md` (24 starts, evidence weights) covers 0.931
(root mean square 1.45) and is right to be wider where a regime is hardly visited, but it is **3.9 times wider than the Gibbs sampler on
average (median 3.0)** on every seed, including the 12 where every regime has at least 5 steps and plain EP is exact: the 24 starts
find the relabellings of the same reading (the broad priors prefer one labelling by only 1.4 to 8 nats), and the mixture over labels is
the posterior of the model as written, whose rate of "regime 0" is not one number.

**What was added** (`probprog/inference/ep-multistart.ts`, `_relabel.ts`): a knob `labels`.
- `keep` (the default, the behaviour of before): the components are the model's labels.
- `permute`: after every new reading found (a converged fixed point whose canonical form no earlier one has), start also from every
  relabelling of its discrete messages (all `k!` permutations of the category count that matches a vector of parameters, at most 4
  categories), so the labellings the priors allow are found on purpose and not by chance. The read-out is unchanged.
- `sort`: the same starts, and the posterior is read with the components sorted by the mean of the first vector of continuous
  parameters of length `k` (the rates), every site that is indexed by the components moved with them (a vector, both axes of a
  matrix, a covariance, the probabilities of a discrete site, which are now mixed as probabilities and not as an expected index). The
  weights are the evidence of each fixed point, so the evidence of the model is the same in all three.

**Measured, the same 24 seeds and 72 intervals, 24 starts** (Gibbs is the reference sampler of the page, started in the data-ordered
mode; its coverage is 0.931):

| EP                   | coverage of the true rates | rms (error / sd) | EP sd over Gibbs sd, mean (median) | time (s, relative) |
| -------------------- | -------------------------- | ---------------- | ---------------------------------- | ------------------ |
| one run              | 0.778                      | 4.55             | 0.69 (0.94)                        | 14                 |
| 24 starts, `keep`    | 0.931                      | 1.45             | 3.91 (3.02)                        | 335                |
| 24 starts, `permute` | 0.944                      | 1.41             | 4.31 (4.27)                        | 760                |
| 24 starts, `sort`    | **0.889**                  | 1.68             | **0.83 (0.95)**                    | 337                |

`sort` is the first EP here whose intervals are as wide as the sampler's where every regime is visited (sd ratios 0.84 to 1.01 on seeds
1, 3, 6, 9, 12, 13, 15, 16, 21 to 24) and wider than one run's where a regime is not (on seeds 2, 5, 8, 11, 14, 18 the unvisited
regime's rate has 0.77 to 1.27 of the sampler's spread, where one run had 0.05 to 0.15), with the coverage of one run restored from 0.778
to 0.889 (visited seeds 0.861, barely visited 0.917; plain 0.806 and 0.750). `keep` and `permute` cover more by being wide everywhere:
as the model is written (the labels carry the priors' names) that is the honest posterior, and a reading of the labels as "the
low, the middle, the high regime" is what `sort` gives. The default stays `keep`: it does not change what a site named `rate[0]`
means, and the choice of `sort` is the user's (it is right when the components are exchangeable up to their parameters, as the
changepoint regimes and mixture components are).

**What it does not fix: a regime that is visited a little.** Seeds 4, 7, 10 (a regime with 6, 4 and 4 of 80 steps) and 19 (one regime
only) are overconfident with every variant: rate sd 0.2 to 0.35 of the sampler's, errors of 2 to 4.5 sds on seeds 4 and 10, and the
truth outside the interval on 6 of the 72 intervals of `sort`. The mode that is missing is not a relabelling: all 24 starts reach
the reading in which those few steps belong to that regime, whose rate EP then believes to the precision of the steps it assigned,
and the other readings (those steps absorbed by their neighbours) are not fixed points of EP at all (the runs cycle, change 0.3 to 0.4 at
the sweep cap with damping 0.5 and 0.8). It is the loop of gap `loopy-overconfidence` again, between a rate shared by all the steps and
a discrete state per step; a second attempt, a damped retry of the starts that did not settle (damping 0.5, then 0.8), converted none
of them and was removed. What would close it is the structure: the chain summed out given the rates (a forward algorithm as one
site over the three rates, a three-dimensional numeric message), not a better start.

**The circular mixture** (`ep-program-circular`, the data form, two components, seed 2 is the bad mode): the multi-start posterior is
empty there, because the model has no EP evidence at the fixed point (more than a hundred of its row gates have an improper cavity: a negative
precision on the means from the other rows), so there is nothing to weight the starts by. Left as it is and named in the gap.
The relabelling counts the categories that match a vector of parameters (here 2 components, not the 3 copies of a reading), so it
applies to this model as soon as it has an evidence.

**The extended changepoint program** (Dirichlet rows, 100 steps, stay 0.93, seeds 1, 2, 3, 10, 24): see the table at the end of this
section.

Truth inside the 90% interval, the three rates and the nine entries of the transition matrix (12 per seed), one run against 12 starts
with `sort` (the regimes' visits in brackets):

| seed (visits)  | one run | 12 starts, `sort` | the rate of the thin regime, one run -> `sort` (truth) |
| -------------- | ------- | ----------------- | ------------------------------------------------------ |
| 1 (14, 47, 39) | 11/12   | 11/12             |                                                        |
| 2 (83, 2, 15)  | 10/12   | 11/12             | regime 1: 3.4 +- 1.1 -> 6.5 +- 7.2 (9)                 |
| 3 (16, 50, 34) | 10/12   | 10/12             |                                                        |
| 10 (0, 22, 78) | 10/12   | 9/12              | regime 0: 9.3 +- 1.0 -> 7.4 +- 2.0 (3)                 |
| 24 (2, 82, 16) | 10/12   | 12/12             | regime 0: 6.3 +- 3.0 -> 7.3 +- 3.4 (3)                 |
| all five       | 51/60   | 53/60             |                                                        |

Five seeds is a smoke measurement, not a calibration: the 60 intervals cannot tell 0.85 from 0.88. What is visible is the same
picture as the basic program (the thin regime's spread opens up, e.g. 1.1 -> 7.2), at 4 to 8 times the time of one run (12 to 28 s against 3 to 5 s).

## 2. Loops: power EP on the softmax slopes, and where it does not help

The class slopes of the softmax classifier (`softmax-regression-learned`, three slopes with N(0, 2²) priors, 24 points, grid reference)
are strongly coupled (a common shift of the scores is invisible to the softmax; only the prior breaks it), one Gaussian per slope is
too narrow, and EP does not settle in most runs at damping 0.5 (56 of 81). Power EP (every factor with a power, `power` now a knob
of `epSpecMethod`; fractional EP when below 1) makes the spreads wider, monotonically, and the run settles. The test of calibration
draws the true slopes from the prior (so that the exact posterior is calibrated by construction: its coverage is 0.96) and counts how
often the true slope is inside EP's 90% interval, over 80 seeds (40 for 48 points):

| points | plain EP        | power 0.5    | power 0.3    | power 0.2    | power 0.1    | exact |
| ------ | --------------- | ------------ | ------------ | ------------ | ------------ | ----- |
| 10     | 0.798 (sd 0.74) | 0.918 (0.89) | 0.963 (1.00) |              |              | 0.963 |
| 24     | 0.630 (sd 0.55) |              | 0.872 (0.82) | 0.934 (0.92) |              | 0.959 |
| 48     | 0.455 (sd 0.40) |              |              | 0.764 (0.74) | 0.919 (0.92) | 0.959 |

(in brackets, EP's spread over the exact spread). The power that gives the exact coverage is about 5 divided by the number of rows
(0.5, 0.2 to 0.25, 0.1), a rule fitted to this one model, not a derivation. The price is the mean: it moves by 0.04 exact sds at power 1,
0.11 at 0.5, 0.2 at 0.2 (on the default data of the learned atom 0.47 sds), so the Kolmogorov-Smirnov distance to the exact belief is
not better, only the intervals are honest. The two softmax atoms now run at power 0.5 (ten points) and 0.25 (24), with their
tolerances restated (`ep-atoms-multiclass.ts`); the `power` knob is on the EP spec method only.

**It does not generalise.** Beta outcomes with two random shapes (`pair-factors.test.ts`, the exact posterior by a 400 by 400 grid):
spreads 0.65 of the exact at power 1, 0.49 at 0.7, 0.40 at 0.5, 0.30 at 0.3 (worse: a power below 1 narrows these) and 1.5 at 1.3;
the exact spread is between power 1.1 and 1.15 (0.76 to 1.0 of the exact, with the mean of the first shape 0.1 to 0.2 exact sds off), so
the sign of the correction depends on the model. The learned logistic and probit regressions are already packed into one joint vector
and are calibrated (spreads 1.03 and 1.00 of the exact ones), and a power changes nothing there. The tournament skills do not respond
to a power or an alpha (section 11 of `ep-changelog.md`). So the default is not changed for any other model, and no linear-response or
variance-inflation correction was built: where the loop is a missing correlation (the slopes), the right fix is a joint vector
over the classes (a vector-valued softmax site, the gap `softmax-many-classes`); power EP is the measured stopgap. The "damping-aware
variance inflation" was not tried: damping does not change the fixed point, only how it is reached.

## 3. Stochastic volatility with the level, the persistence and the precision unknown

Model: a log-variance AR(1) around an unknown level `mu` (N(-1, 1)), persistence `phi` (N(0.8, 0.3)) and innovation precision `tau`
(gamma(3, 0.3)), 60 returns, truth (-1, 0.9, 11.1); 30 seeds; EP at damping 0.5 and tolerance 1e-4 (the loops never meet 1e-8).
Coverage of the truth by the 90% interval:

| EP                                             | level | persistence | precision | pooled |
| ---------------------------------------------- | ----- | ----------- | --------- | ------ |
| one run                                        | 0.484 | 0.581       | 0.633     | 0.566  |
| inside a grid over the persistence (21 points) | 0.968 | 0.677       | 1.000     | 0.882  |

The reference NUTS run (seeds 1 and 2): level -0.918 +- 0.425 and -1.108 +- 0.586, persistence 0.731 +- 0.199 and 0.826 +- 0.173, precision
11.5 +- 6.1 and 12.5 +- 6.1; one EP run says the level to +- 0.13 (a fifth to a third of the sampler's), the grid -0.914 +- 0.655 and
-1.102 +- 0.748 (the means within 0.02, the spread 1.3 to 1.5 of the sampler's), persistence 0.760 +- 0.187 and 0.838 +- 0.141 (0.7 to 0.94 of
the sampler's), precision 11.5 +- 5.7 and 12.4 +- 5.9. This needed no new rule: `epGridMethod(bound, { over: ['phi'] })` conditions
the persistence, the level and the precision stay inside each EP run (the chain given the persistence is a tree but for the shared precision).
A grid over both the level and the persistence was tried and is worse (the level needs 70 points: the grid of 5 by 5 put all the weight on one
node, sd 0.007). The persistence is still undercovered (0.68, mean sd 0.12 against the sampler's 0.17 to 0.2): the nodes at high persistence
often do not settle, and only the ones that settled weigh the grid (below), which truncates the posterior on that side; seeds 26 and 27 put the
persistence at 1.02 and 1.14 with sd 0.012. With damping 0.8 and 400 sweeps (to settle more nodes) a collapsed fixed point (persistence -0.2,
precision 0.002, a high evidence) took the whole weight on 3 of 3 seeds, so the damping was left at 0.5.

Engine fixes found on the way (`ep-grid.ts`): a posterior asked for while no node has an evidence yet threw `no run has a weight` and now says
`failed`; and **a node that did not settle no longer weighs the grid when some did** (the EP energy at a state that cycles or has collapsed
was 20 to 30 nats above the settled nodes' and put the whole posterior on a degenerate node: persistence 0.019, precision 0.002 on 4 of the first
21 seeds). Tests: `ep-grid.test.ts`, `ep-grid-sv.test.ts` (seed 1 against the NUTS numbers).

Not done: the joint rule over the level and the persistence (a block update on the chain); the grid is the measured route, 25 s per fit on
60 returns, and the persistence's coverage says it is not enough, so the setup is still not offered as a learned-parameters page.

Update 2026-11-12 (`agent-ad2a59-svchain`): the two defects the grid left were in the scalar chain, not in the grid. With the persistence known
the level's belief is twice too wide (the cavities of the previous state and of the level are correlated through the level and taken as
independent), also in a linear Gaussian copy of the model; the precision's sixty messages cycle at a low persistence. The steps of the series
are one `gaussian-chain` factor now (`message-passing/operators/gaussian/chain.ts`, written by `factor-graph/_chains.ts`); against the
point-mass filter every one of 30 seeds has the three means within 0.2 sd and the spreads within 0.12 of the reference's. The full account is
in `ep-changelog.md` (2026-11-12, `agent-ad2a59-svchain`).

## 4. Evidence of a plated factor with an improper cavity

The learned logistic regression compiles to one plated factor for all rows, and the scale mixture on the weights leaves the plate a
cavity with a negative precision, so the plate has no EP energy (`diagnostics.evidenceUndefinedAt`). When a converged run has that defect and
the plates were vectorised, `epSpecMethod` now runs the same model with a factor per row (each cavity proper) and reports its energy as the
evidence, named `ep-energy (rows expanded)`; the marginals stay those of the plated run. Against the grid evidence of the same model:
within 0.1 nats (0.046 measured), as `plates: 'expand'` had given (`ep-evidence.test.ts`). The extra run happens only at a converged state and
once per state, and is skipped when the rows-expanded run does not converge (then the evidence is undefined with its reason, as before).
An alpha-divergence run still has no EP energy.

## 5. Not started

More than six softmax classes by a cheaper site, and random ordinal cut points, were not started; the other items took the time.
