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
