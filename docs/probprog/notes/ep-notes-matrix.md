# EP and covariance matrices (2026-11-11, `agent-pp191-matrix`)

Closes the gap `matrix-families`: a learned covariance (correlated random effects, a multivariate normal with an unknown precision
matrix) is now a latent under EP. What was built, how it was checked, and where it is weaker. Remainders are the gaps `matrix-larger` and
`hierarchical-matrix-few-groups` in `ep-gaps.ts`.

## 1. What exists

**Families** (`message-passing/families-matrix.ts`). `wishart(d)`: messages `[a, B]` with `a = (nu - d - 1) / 2`, `B = -S^-1 / 2`, sufficient statistics
`(log |L|, L)`, the matrix analogue of the gamma (`d = 1` is the gamma, tested). `fromMoments` matches `E log |L|` and `E L` (the KL projection) by a
safeguarded Newton step on `d log k - psi_d(k)`. `normalWishart(d)`: the joint of a mean and a precision matrix, the analogue of the normal-gamma.
Tested against the library's own `wishart` density (`log q = eta . T - A` at random matrices), its mean, variance and entropy, and the
`d = 1` families.

**Operators** (`operators/covariance.ts`, `correlation.ts`; shared helpers in `_sides.ts`, `_bartlett.ts`, `_gridn.ts`).
- `mvn-wishart`: `x ~ N(mean, L^-1)`, ports `x`, `mean` (Gaussian vectors, observed or random) and `L` (Wishart). The fully known case is the exact
  conjugate message `[1/2, -r r' / 2]` in any dimension. With Gaussian cavities the vectors integrate out (`N(r; 0, V + L^-1)`) and EP matches the
  Wishart to the tilted moments by the **Bartlett quadrature** of the Wishart (`L = C A A' C'`, gamma and normal one-dimensional rules; 12 nodes per
  coordinate at `d = 2`, 5 at `d = 3`). Two things make it accurate: the exact conjugate part of the factor (the scatter, and the difference
  `x - mean` when its cavity is narrow) is moved into the base measure, so the quadrature integrates a smooth remainder (exact when the vector is known), and
  the logarithm's bias under the rule (`E log |L|` is not a polynomial) is measured and removed. A factor may stand for `count` rows with a given
  `scatter` (a plate): the likelihood of i.i.d. rows with a shared mean depends on the rows only through their mean and scatter, so the graph does not grow with the data.
  Variational rule: closed form (expected residual outer product). Evidence: the integral against the cavities.
- `mvn-normal-wishart`: the exact conjugate update of `(mu, L)` (marginal of the mean a multivariate t; checked against the textbook posterior and the t density).
- `mvn-sd-corr` (bivariate): `Sigma` from two gamma precisions and a correlation `2u - 1` with `u` a beta variable (`Beta(1, 1)` uniform, `Beta(eta, eta)`
  the 2 by 2 LKJ prior). The three coordinates are integrated on a zooming panelled grid (`_gridn.ts`: log precisions, logit of `u`), the vector and mean in
  closed form given them. A known sd or correlation is one node, so known sds make it a one-dimensional integral. EP only: the expected log of a beta
  correlation has no closed form (coverage's `behindEp` 6 to 8).
- `numerics/quadrature/gaussLaguerre(n, alpha)` gained the generalised weight (`x^alpha e^-x`).

**Model layer.** Families `wishart`, `inverseWishart` (matrix-valued, event rank 2; the library's own distributions and fixtures), `correlation(eta)`
(the LKJ marginal, `transformed(beta, affine)`), `bivariateNormal(loc, { scale1, scale2, rho })` and `multivariateNormal(loc, { precision | covariance }, { over: axis })`
(the vector's axis named, the other dimensions plates). **Compiler** (`factors.ts`, new methods only): a Wishart latent is one variable; `multivariateNormal` with a
`precision` that is a Wishart latent (or `covariance` an inverse-Wishart one) maps as observed rows (one plated `mvn-wishart` factor for a shared mean, one per row
otherwise) or as one Gaussian vector per group (`dims: [group, axis]`, a factor per vector, coordinates readable as `theta.at(group)` summed against a design over the axis);
the mean is known or one Gaussian vector variable. `bivariateNormal` with `1 / sqrt(tau)` scales (gamma latents) and a `correlation` latent maps to `mvn-sd-corr`. The posterior of a
matrix site is the library `wishart` (or `inverseWishart`) with the matched parameters, of a correlation the transformed beta. `constantFamily` (graph.ts) now gives a
constant vector its own length.

## 2. Atoms and accuracy

Six atoms (`ep-atoms-covariance.ts`, references in `ep-atoms-covariance-ref.ts`), each with a region snippet in the user's style and a page `ep-<name>`. The references are independent of the rule: the
closed-form Wishart posterior; a one-dimensional integral for the correlation; and three-dimensional grids over the entries of the precision matrix (or the two precisions and the correlation)
with the means and effects integrated out in closed form (Woodbury for the random effects). EP against them (mean error in reference sds / sd ratio):

| atom | result |
| --- | --- |
| `mvn-wishart` (mean known) | exact, 2 sweeps (z 0, ratio 1.000) |
| `mvn-wishart-learned` (mean unknown, 25 to 40 rows) | z 0.000, ratio 1.000 to 1.001 (one plated factor: EP matches the exact moments); also at 2 or 3 rows and 1.2 degrees of freedom |
| `bivariate-correlation` | mean z 0.02, **sd ratio 0.83** (eight pairs, rho about 0.8) |
| `bivariate-correlation-learned` | z 0.000 to 0.002, ratio 1.00 to 0.99 (rho 0.37); mean, precisions exact |
| `random-intercepts-slopes` (mean known, 6 groups) | matrix and every group: z 0.000, ratio 0.99 to 1.003 |
| `random-intercepts-slopes-learned` (10 groups) | mean ratio 0.93, matrix ratio 1.02, groups 1.000, z at most 0.04 |

Calibration (ground truth recovered over seeds, 90% intervals): `mvn-wishart-learned` 40 seeds coverage 0.85 to 0.97 (zrms 0.83 to 1.19); `bivariate-correlation-learned` 40 seeds
0.88 to 1.00 (the correlation 0.88, zrms 1.32); `random-intercepts-slopes-learned` 30 seeds 0.87 to 1.00 for the mean and matrix, 0.90 and 0.91 for the 300 intercepts and 300 slopes. Note the lesson of
the first runs: with a prior of 4 degrees of freedom centred on the identity and a truth five times away from it, the *exact* posterior had coverage 0.00 to 0.5 for the matrix (the prior pulls); EP agreed with the
exact posterior to 2%, so the calibration defaults were changed (prior of 3 degrees of freedom, truths nearer the prior), not the floors.

Cost: EP is 0.1 to 0.7 s per atom at `d = 2` (the reference grids are the slow part, and are lazy, so a calibration seed does not pay for one). Dimension 3 works end to end (a conjugate and a flat-mean
model against closed forms; a hierarchical model with eight groups of three effects converges in 30 sweeps, about 12 s).

## 3. Honest limits (measured)

- **Few groups, shared mean.** With the population mean unknown and 3 to 4 groups EP's spread for the mean is 0.77 to 0.89 of the exact and the precision matrix is 1.05 to 1.25 times too wide (1.5 to 1.7
  with a prior of 1.2 degrees of freedom, where it also needs 30 to 40 sweeps); at 6 to 10 groups 0.93 and 1.02. The means stay within 0.5 posterior sd. A single plated factor is exact. (`hierarchical-matrix-few-groups`.)
- **A strong correlation from few pairs.** The beta that matches the exact marginal on `E log u` and `E log (1 - u)` has a spread 0.81 to 0.92 of the exact at rho 0.9 to 0.99 (6 to 12 pairs), 0.97 at eta 4.
  The mean is right to 0.07 sd. The atom's tolerance says so (sd ratio floor 0.75). The belief is drawn as a normal on the correlation, which adds to its KS distance.
- **A Wishart with few degrees of freedom** is not a problem for EP when the rows are plated or observed (exact), and the prior is: at 1.2 to 2.2 degrees of freedom the prior's pull on the matrix dominates a handful
  of rows, which is the model, not the rule.
- **Beyond the reach**: non-conjugate Wishart above dimension 3, an LKJ of order 3 or more, a latent bivariate vector with its own sds and correlation (a grid per group), and a normal-Wishart variable written
  inside a larger model: `matrix-larger`.
- **Cost of dimension 3**: 15,625 nodes per call, three ports per group, 12 s for eight groups; a shared tilt across the ports of one factor would cut it threefold.

## 4. Searched and reused

Searched for Wishart, LKJ, normal-Wishart, `gaussian-mv`, `eigh`, Gauss-Laguerre, `transformed`, `affine`, the pair and likelihood grids. Reused: the library's `wishart`, `inverseWishart`, `lkj` distributions and their fixtures
(no new fixtures: no new library distribution), `gaussianMv`, `defineFamily`, `gammaAxis` (`_grid.ts`), `gaussLegendre`/`gaussHermite`/`gaussLaguerre`, `transformed` and `bijectors.affine` for the correlation,
the `plate` and `addPlated` machinery, `eventVector`, the atom scaffold (`defineAtom`, `atomCalibration`, `hyperPosterior`-style exact marginals), `linear-regression-row` for the readings.

## 5. Update (2026-11-12, `agent-a3e7a0-matrix`)

Sections 3's two limits moved: with a random population mean the group vectors, the mean and the matrix are now one factor (`mvn-wishart-group`), which removes the loop that made the mean too narrow and the
matrix too wide with few groups (the changelog has the numbers), and dimension 4 is allowed (three nodes per coordinate). The gap `hierarchical-matrix-few-groups` became `correlation-strong-few-pairs` (the beta
belief of a strong correlation); `matrix-larger` lists what stays out (dimension 5, LKJ of order 3, a latent bivariate vector with its own sds and correlation, a normal-Wishart variable in a larger model).

## 6. Update (2026-11-12, `agent-a6b87c-corr`)

**The correlation's belief.** Section 3's strong-correlation limit is fixed for the spread: `mvn-sd-corr` matches the beta on `(rho + 1) / 2` to the tilted mean and variance of the
correlation (`match: 'moments'`, the default), not to `E log u` and `E log (1 - u)` (the KL projection, `match: 'log-moments'`). Measured against the exact grid posterior (known sds,
unit variances, 4 seeds each, 2 to 10 pairs): the KL beta had a sd ratio (mean over seeds) 0.54 to 0.98 at rho 0.7 to 0.99 (0.91 to 0.99 at 0.3 and 0.5; single seeds down to 0.45), the new one 1.000 everywhere, mean z 0.00 (0.02 to 0.03
before), evidence within 0.001 nats, two sweeps. The belief's curve is the cost: at rho 0.99 from five pairs the 5% and 95% points of the beta with the right spread are 0.4 of an exact
sd off, and its own Kolmogorov distance 0.4 (0.15 for the KL beta). A Gaussian on the Fisher transform (the logit of `u`) with the beta prior as a numeric site was built and measured, and dropped: its
curve is the best (0.04 to 0.19), but the two-site fixed point is off 1% to 8% in spread, 0.005 to 0.08 nats in evidence, and slow at rho 0.99 from two to ten pairs (10 to 60 sweeps).

**The learned atom's reference was wrong for a strong correlation.** The atom `bivariate-correlation-learned` is checked against a three-pass grid with equal steps in the precisions and in rho,
and the grid's spread of a correlation of 0.95 from eight pairs was 0.033 where an independent grid in the logs and the Fisher transform (100 to 200 nodes a side, stable to five figures) has 0.0429 (EP:
0.0417); its mean mixture kept the heaviest 6000 nodes and was 3% to 5% off. `posterior3` takes a `warp` per coordinate (`log`, `correlation`: equal steps in `2 atanh(rho)`) and a
`mixtureNodes` limit; the learned bivariate atom uses them, no other atom does (their digest lines are identical to the bit). The 0.89 of the exact spread this atom showed at rho 0.9 was against this reference; the KL beta's true ratio there was about 0.83.

**The group form.** `mvn-sd-corr-group` is the joint site of `J` bivariate vectors with a shared Gaussian mean and the two gamma precisions and the beta correlation of one covariance: the closed-form
integral over the vectors and the mean at a node (the evaluator of `mvn-wishart-group`, moved to `_group.ts`), a panelled grid over the three parameters (`mvn-sd-corr`'s, `_sdcorr.ts`), each vector and
the mean matched to the mixture of their conditionals over the nodes, one tilt for all ports of a sweep. The compiler maps a latent `bivariateNormal` over `dims: [group, coef]` to it when the mean is
one Gaussian vector variable (a known mean: a factor per vector). Tested against a dense joint-Gaussian grid (the operator, two and three vectors; `J = 1` is `mvn-sd-corr` to 1e-9) and, as a model of
four groups with readings, against a grid over the parameters with the vectors and the mean integrated out densely: the mean, the correlation, each precision and a vector within 3% of the exact spread
and 0.06 sd of the exact mean, 20 sweeps, under a second. What stays out: an LKJ prior of order 3 (the correlations of a larger matrix), vectors of more than two coordinates with their own sds and
correlation, and a second factor on the same precisions or correlation (a loop, as everywhere in EP).
