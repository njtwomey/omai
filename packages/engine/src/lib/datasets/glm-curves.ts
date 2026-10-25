/**
 * One-dimensional responses of generalised linear models with a log link: counts (Poisson, negative binomial) and
 * positive skewed targets (gamma, inverse Gaussian, Tweedie) whose mean is `exp(intercept + slope x)`, for fitting and
 * checking GLMs.
 */

import { type DatasetGenerator, type RegressionData, xy } from '../data/index.ts'
import {
  type Distribution,
  gamma as gammaDist,
  inverseGaussian,
  negativeBinomial,
  poisson as poissonDist,
} from '../dists/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { add, exp, mul, sliceAxis, squeeze, toFlat, type Value } from '../tensor/index.ts'
import { matrix, stamp, vector } from './_util.ts'

/** The response families of {@link glmCurve}. */
export type GlmCurveFamily = 'poisson' | 'negativeBinomial' | 'gamma' | 'inverseGaussian' | 'tweedie'

/** Draws of one family with means `mu` and a dispersion (see {@link glmCurve} for each family's meaning). */
function draws(k: Key, family: string, mu: readonly number[], dispersion: number, power: number): number[] {
  const draw = (d: Distribution, key: Key = k): number[] => toFlat(d.sample(key))
  switch (family) {
    case 'negativeBinomial': {
      // NB2: variance mu + alpha mu^2, so r = 1 / alpha successes with success probability r / (r + mu) (scipy's).
      const r = 1 / dispersion
      return draw(negativeBinomial(vector(mu.map(() => r)), vector(mu.map((m) => r / (r + m)))))
    }
    case 'gamma':
      // Shape 1 / dispersion (the squared coefficient of variation), rate shape / mu.
      return draw(gammaDist(vector(mu.map(() => 1 / dispersion)), vector(mu.map((m) => 1 / (dispersion * m)))))
    case 'inverseGaussian':
      // Variance dispersion mu^3: shape lambda = 1 / dispersion.
      return draw(inverseGaussian(vector(mu), vector(mu.map(() => 1 / dispersion))))
    case 'tweedie': {
      // Compound Poisson-gamma (1 < p < 2): N ~ Poisson(mu^(2-p) / (phi (2-p))) gamma(alpha, theta) jumps with
      // alpha = (2-p) / (p-1) and theta = phi (p-1) mu^(p-1); the sum of N jumps is gamma(N alpha, theta), 0 when N = 0.
      const alpha = (2 - power) / (power - 1)
      const counts = draw(
        poissonDist(vector(mu.map((m) => m ** (2 - power) / (dispersion * (2 - power))))),
        foldIn(k, 0),
      )
      const sums = draw(
        gammaDist(
          vector(counts.map((c) => Math.max(c, 1) * alpha)),
          vector(mu.map((m) => 1 / (dispersion * (power - 1) * m ** (power - 1)))),
        ),
        foldIn(k, 1),
      )
      return sums.map((s, i) => ((counts[i] ?? 0) > 0 ? s : 0))
    }
    default:
      return draw(poissonDist(vector(mu)))
  }
}

/**
 * A one-dimensional response of a generalised linear model with the log link: x uniform on [0, 3], mean
 * `mu = exp(intercept + slope x)`, and y drawn from the family with that mean. `dispersion` is the negative binomial's
 * alpha (variance mu + alpha mu^2), the gamma's squared coefficient of variation (variance dispersion mu^2), the inverse
 * Gaussian's 1 / lambda (variance dispersion mu^3) or the Tweedie phi (variance phi mu^p, a compound Poisson-gamma
 * with exact zeros); the Poisson ignores it. The regression function is the mean curve.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { glmCurve } from 'whitebox-engine/datasets'
 * const { data } = generate(glmCurve, { knobs: { n: 50, family: 'poisson' } })
 * Array.from(data.y.data).every((v) => Number.isInteger(v) && v >= 0) // => true
 * ```
 */
export const glmCurve: DatasetGenerator<RegressionData> = {
  name: 'glmCurve',
  kind: 'regression',
  description: 'Counts or positive skewed targets whose mean is exp(intercept + slope x), for GLM regressors.',
  knobs: dict({
    n: int(2, 100_000, { default: 150, label: 'samples', description: 'The number of rows.' }),
    family: choice(['poisson', 'negativeBinomial', 'gamma', 'inverseGaussian', 'tweedie'], {
      default: 'poisson',
      label: 'response family',
      description: 'Poisson or negative binomial counts, gamma or inverse Gaussian positive values, or Tweedie.',
    }),
    intercept: real(-3, 3, {
      default: 0,
      label: 'intercept',
      description: 'The log of the mean at x = 0.',
    }),
    slope: real(-2, 2, {
      default: 0.8,
      label: 'slope',
      description: 'How fast the log of the mean grows with x.',
    }),
    dispersion: real(0.01, 5, {
      default: 0.5,
      log: true,
      label: 'dispersion',
      description: 'Extra spread beyond the Poisson: NB alpha, gamma CV², inverse Gaussian 1/λ or Tweedie φ.',
    }),
    power: real(1.05, 1.95, {
      default: 1.5,
      label: 'Tweedie power',
      description: 'The variance power p of the Tweedie family (variance φ μ^p), between Poisson (1) and gamma (2).',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const family = knobs['family'] as string
    const intercept = knobs['intercept'] as number
    const slope = knobs['slope'] as number
    const dispersion = knobs['dispersion'] as number
    const power = knobs['power'] as number
    const xs = Float64Array.from(uniform(foldIn(k, 0), [n], { low: 0, high: 3 }).data)
    const mu = Array.from(xs, (x) => Math.exp(intercept + slope * x))
    const ys = draws(foldIn(k, 1), family, mu, dispersion, power)
    const f = (v: Value): Value => exp(add(mul(squeeze(sliceAxis(v, -1, 0, 1), -1), slope), intercept))
    return {
      data: xy(matrix(xs, n, 1), vector(ys), {
        ...stamp('glmCurve', glmCurve.description, knobs),
        featureNames: ['x'],
      }),
      truth: {
        regressionFunction: f,
        relevantFeatures: [0],
        extra: { family, intercept, slope, dispersion, ...(family === 'tweedie' ? { power } : {}), link: 'log' },
      },
    }
  },
}
