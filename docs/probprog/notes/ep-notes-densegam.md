# Notes: the dense non-centred scale (agent-pp192-densegam, 2026-11-11)

Package: take the per-coefficient `apply-product` posterior of the non-centred Bayesian GAM (band 17% wide, about 200 sweeps without
meeting the tolerance) to one Gaussian vector, as `gaussian-mv-precision-scale` does for a gamma smoothing parameter.

## What was built

- `message-passing/operators/function-scale.ts`, operator `gaussian-mv-function-scale`: `beta_j = z_j u_j(x)` for a Gaussian vector,
  independent standard normals (integrated out) and one Gaussian or gamma variable `x`; `u_j` is a chain of unary steps per
  coefficient (the same `chainOf` as `apply`). Given `x` the vector is Gaussian with diagonal precision `1 / u_j(x)^2`, so against the
  Gaussian cavity of the vector the integral is closed form and what is left is one scalar integral over `x`.
- The grid and the mixture of conditionals are not a second version: `operators/scale.ts` (`gaussian-mv-precision-scale`) was
  refactored so its tilt is `vectorScaleTilt(h, J, model)` over a `VectorScaleModel` (the precisions at `t`, the cavity of the mixing
  variable as a density in `t`, the range), and `vectorMessageOf` (the dense message). The gamma multiplier and the Gaussian or gamma
  input of a chain are two models for the same code. One addition to the shared grid: when the mass is narrower than two grid steps
  it zooms onto the window that holds it (a wide Gaussian cavity, sd 3 in `log lambda`, needs it; it does not change the gamma runs
  that were resolved).
- Compiler (`probprog/model/factors.ts`, localised): a plan step `findNonCentred` marks a derived node `D = z / S` (z a latent vector of
  independent standard normals that no other node reads, `S` depending on a latent, `D` read by a `matmul`); `denseNonCentred` makes `D`
  one `gaussian-mv` variable with the factor when every divisor is a chain from the same variable (a known divisor is a constant
  chain), and `z` is then not a variable. Otherwise (`z` is compiled as it was) `tiedToVector` makes the per-coefficient scalar
  products the coordinates of one Gaussian vector `D` (a `dot-constant` factor each): the ARD form. `sharedScale: 'elements'`
  switches both off. The GAM model is not reworded.

## Measured (sine, 60 rows, 10 coefficients; reference at 60 points per hyperparameter, ARD against two NUTS chains of 800 draws)

| form                 | band ratio before | after | log smoothing shift before | after | sweeps before | after | time before | after |
| -------------------- | ----------------- | ----- | -------------------------- | ----- | ------------- | ----- | ----------- | ----- |
| penalty, `logNormal` | 1.17 to 1.19      | 1.025 | 0.25                       | 0.03  | 200 (cap)     | 23    | 8.2 s       | 1.2 s |
| ARD                  | 1.06              | 0.99  |                            |       | 64            | 24    | 1.8 s       | 1.0 s |

The curve error is 0.004 (was 0.005) and 0.009 (was 0.011) response sd. Times are one core under `nice`, a single run each.

ARD: the dense treatment of the likelihood helps, by less than the log-normal form (the band was 1.06 against a long sampler, the
1.09 quoted earlier was against a short NUTS run): what it removes is the loop of scalar sums (`linear` x 60 and `equal` x 10 become
`dot-constant` x 70), which is what shortened the run. The prior itself (one latent per coefficient) cannot be integrated into a
single scalar, so the apply-product stays per coefficient.

## Things worth knowing

- The page's exact reference at its default of 28 points per hyperparameter under-resolves the noise: its log noise sd is 0.041
  against 0.098 at 60 or 120 points. EP's is 0.111, so about 13% wide, and the log smoothing's 0.694 against 0.619 is 12% wide: what
  is left of the error is a moment match of a skewed marginal, not the correlations. The EP test now uses 60 points.
- The evidence of the new operator is the log mass against the unnormalised messages (tested against a grid); the posterior's
  evidence diagnostics read it like the other operators'.
- Variational message passing has a rule for the dense factor since 2026-11-12 (the mean field of the vector given `x`, no standard
  normals); ARD has none (gap `non-centred-scale`, now only that).
