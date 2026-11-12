/**
 * The conjugate leaf models of a Bayesian tree: a Normal-Gamma model of a real target (regression) and a Dirichlet model
 * of a label (classification; with two classes it is the Beta-Bernoulli model). A leaf keeps additive sufficient
 * statistics (a count, a sum and a sum of squares; or a count per class), so the posterior of a leaf, the exact marginal
 * likelihood of its rows and the posterior predictive are closed forms over a trailing statistics axis, evaluated for
 * many leaves or many candidate splits at once with tensor operations. The precision of a leaf (how much evidence it
 * holds) and its step size, the reciprocal of that precision, are read off the same statistics.
 */

import { OmaiError } from '../../base/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import {
  add,
  asTensor,
  cast,
  div,
  log,
  mul,
  oneHot,
  ones,
  sliceAxis,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type TensorLike,
  toArray,
  toFlat,
  type Value,
} from '../../tensor/index.ts'

const f64 = (x: TensorLike): Value => cast(asTensor(x), 'float64')

/** What a leaf's posterior says, in plain numbers (the leaf model's `describe`). */
export interface LeafPosterior {
  /** Rows the leaf holds. */
  readonly n: number
  /** The posterior mean of the target (one number), or the class probabilities. */
  readonly value: readonly number[]
  /** Posterior standard deviation of the mean (regression only). */
  readonly sd?: number
  /** The standard deviation of one new observation around the mean (regression only). */
  readonly noise?: number
  /** The evidence the leaf holds, in observations: the prior strength plus `n`. */
  readonly precision: number
  /** One over the precision: how far one new observation moves the leaf's mean (an adaptive step size). */
  readonly stepSize: number
}

/**
 * A conjugate leaf model. Everything takes statistics with a trailing axis of length `size`, so a call handles one leaf,
 * every leaf of a tree, or every candidate split of a node at once.
 *
 * @index structure A conjugate leaf model: statistics, exact evidence, posterior summary.
 */
export interface LeafModel {
  readonly kind: 'normal-gamma' | 'dirichlet'
  /** Length of the statistics axis. */
  readonly size: number
  /** Length of the summary axis (what a prediction averages over trees). */
  readonly outputs: number
  /** The prior, as plain data. */
  readonly prior: Readonly<Record<string, number | readonly number[]>>
  /** Per-row statistics of a target: a matrix `[n, size]`; summing rows gives a leaf's statistics. */
  statistics(y: TensorLike): Value
  /** The number of rows behind statistics `[..., size]`, as `[...]`. */
  count(stats: Value): Value
  /** The exact log marginal likelihood of the rows behind statistics `[..., size]`, as `[...]`. */
  logEvidence(stats: Value): Value
  /** The posterior-predictive summary `[..., outputs]` that a prediction averages over trees (linear in the summary). */
  summary(stats: Value): Value
  /** A leaf's posterior in numbers, from one statistics vector. */
  describe(stats: readonly number[]): LeafPosterior
  /** A prediction from a summary averaged over trees `[g, outputs]`: the mean, its spread, or the class probabilities. */
  read(averaged: Value): LeafPrediction
}

/** A prediction read from an averaged summary: one row per query point. */
export interface LeafPrediction {
  /** The predictive mean (regression), or the probability of the last class (classification, two classes). */
  readonly mean: readonly number[]
  /** The standard deviation of the mean function over trees and leaf posteriors (regression only). */
  readonly sd?: readonly number[]
  /** The standard deviation of a new observation (regression only). */
  readonly observationSd?: readonly number[]
  /** The class probabilities, one row per point (classification only). */
  readonly probabilities?: readonly (readonly number[])[]
}

/** The hyperparameters of the Normal-Gamma leaf: `y | mu, tau ~ N(mu, 1/tau)`, `mu | tau ~ N(m0, 1/(kappa0 tau))`, `tau ~ Gamma(alpha0, beta0)`. */
export interface NormalGammaPrior {
  /** The prior mean of a leaf. */
  readonly m0: number
  /** The prior strength of the mean, in observations. */
  readonly kappa0: number
  /** The prior shape of the noise precision; above 1 so that variances exist. */
  readonly alpha0: number
  /** The prior rate of the noise precision: `beta0 / alpha0` is the prior guess of the noise variance. */
  readonly beta0: number
}

/**
 * The Normal-Gamma leaf model for a real target, with statistics `[n, sum of z, sum of z squared]` for `z = y - m0`. The
 * posterior of a leaf is `kappa = kappa0 + n`, `m = m0 + sum / kappa`, `alpha = alpha0 + n / 2`,
 * `beta = beta0 + (sumsq - sum^2 / kappa) / 2`; the exact log evidence is
 * `lgamma(alpha) - lgamma(alpha0) + alpha0 log beta0 - alpha log beta + (log kappa0 - log kappa) / 2 - n log(2 pi) / 2`;
 * the mean function of a leaf has a Student-t posterior and a new observation a wider one.
 *
 * @param prior - The hyperparameters (see {@link regressionPrior} for ones taken from the scale of the data).
 * @returns The leaf model.
 * @throws OmaiError with code `value` unless `kappa0 > 0`, `alpha0 > 1` and `beta0 > 0`.
 * @concept bayesian-trees
 * @index function Normal-Gamma leaf model: exact evidence and predictive for a regression leaf.
 * @example
 * ```ts
 * import { normalGammaLeaf } from 'omai-engine/trees'
 * import { sum, toFlat } from 'omai-engine/tensor'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 2 })
 * const leaf = sum(model.statistics([1, 2, 3]), 0)
 * toFlat(leaf) // => [3, 6, 14]
 * model.describe(toFlat(leaf)).stepSize // => 0.25
 * ```
 */
export function normalGammaLeaf(prior: NormalGammaPrior): LeafModel {
  const { m0, kappa0, alpha0, beta0 } = prior
  if (!(kappa0 > 0 && alpha0 > 1 && beta0 > 0)) {
    throw new OmaiError(
      'value',
      `normalGammaLeaf: need kappa0 > 0, alpha0 > 1 and beta0 > 0, got ${JSON.stringify(prior)}`,
    )
  }
  const parts = (stats: Value): { n: Value; s: Value; q: Value } => ({
    n: lastAxis(stats, 0),
    s: lastAxis(stats, 1),
    q: lastAxis(stats, 2),
  })
  const posterior = (stats: Value): { kappa: Value; mean: Value; alpha: Value; beta: Value; n: Value } => {
    const { n, s, q } = parts(stats)
    const kappa = add(n, kappa0)
    const beta = add(mul(sub(q, div(square(s), kappa)), 0.5), beta0)
    return { kappa, mean: add(div(s, kappa), m0), alpha: add(mul(n, 0.5), alpha0), beta, n }
  }
  return {
    kind: 'normal-gamma',
    size: 3,
    outputs: 3,
    prior: { ...prior },
    statistics: (y) => {
      const z = sub(f64(y), m0)
      return stack([ones(z.shape), z, square(z)], -1)
    },
    count: (stats) => lastAxis(stats, 0),
    logEvidence: (stats) => {
      const { kappa, alpha, beta, n } = posterior(stats)
      const terms = sub(lgamma(alpha), lgamma(alpha0))
      const scale = sub(alpha0 * Math.log(beta0), mul(alpha, log(beta)))
      const mean = mul(sub(Math.log(kappa0), log(kappa)), 0.5)
      return add(add(terms, scale), sub(mean, mul(n, 0.5 * Math.log(2 * Math.PI))))
    },
    summary: (stats) => {
      const { kappa, mean, alpha, beta } = posterior(stats)
      const meanVariance = div(beta, mul(sub(alpha, 1), kappa))
      const observationVariance = mul(meanVariance, add(kappa, 1))
      return stack([mean, add(meanVariance, square(mean)), add(observationVariance, square(mean))], -1)
    },
    describe: (stats) => {
      const p = posterior(f64(stats))
      const [kappa, mean, alpha, beta] = [p.kappa, p.mean, p.alpha, p.beta].map((v) => toFlat(v)[0] ?? 0) as [
        number,
        number,
        number,
        number,
      ]
      return {
        n: stats[0] ?? 0,
        value: [mean],
        sd: Math.sqrt(beta / ((alpha - 1) * kappa)),
        noise: Math.sqrt(beta / (alpha - 1)),
        precision: kappa,
        stepSize: 1 / kappa,
      }
    },
    read: (averaged) => {
      const mean = lastAxis(averaged, 0)
      const meanVariance = sub(lastAxis(averaged, 1), square(mean))
      const observationVariance = sub(lastAxis(averaged, 2), square(mean))
      return {
        mean: toFlat(mean),
        sd: toFlat(sqrt(meanVariance)),
        observationSd: toFlat(sqrt(observationVariance)),
      }
    },
  }
}

/**
 * Hyperparameters of the Normal-Gamma leaf taken from the scale of a target, a weakly informative choice: the prior mean
 * is the mean of `y`, the mean is worth `kappa0` observations, and the noise variance is guessed as `noiseGuess` times
 * the variance of `y` with the weight of `2 alpha0` observations.
 *
 * @param y - The target.
 * @param options - `kappa0` (default 1), `alpha0` (default 2) and `noiseGuess` (default 0.5).
 * @returns The hyperparameters.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { regressionPrior } from 'omai-engine/trees'
 * regressionPrior([0, 2]) // => { m0: 1, kappa0: 1, alpha0: 2, beta0: 1 }
 * ```
 */
export function regressionPrior(
  y: TensorLike,
  options: { readonly kappa0?: number; readonly alpha0?: number; readonly noiseGuess?: number } = {},
): NormalGammaPrior {
  const t = f64(y)
  const m0 = toFlat(div(sum(t), t.shape[0] ?? 1))[0] ?? 0
  const spread = toFlat(div(sum(square(sub(t, m0))), t.shape[0] ?? 1))[0] ?? 1
  const alpha0 = options.alpha0 ?? 2
  return { m0, kappa0: options.kappa0 ?? 1, alpha0, beta0: alpha0 * (options.noiseGuess ?? 0.5) * spread }
}

/**
 * The Dirichlet leaf model for a class label in `0..K-1`, with the class counts as statistics. The posterior of a leaf has
 * concentration `a + counts`, its mean is the class probabilities, and the exact log evidence of the labels is
 * `lgamma(A) - lgamma(A + n) + sum_k (lgamma(a_k + n_k) - lgamma(a_k))` with `A = sum a`. With two classes this is the
 * Beta-Bernoulli model.
 *
 * @param classes - The number of classes.
 * @param concentration - The prior concentration per class (default 1, the uniform prior; for two classes `Beta(1, 1)`).
 * @returns The leaf model.
 * @throws OmaiError with code `value` unless there are at least two classes and the concentration is positive.
 * @concept bayesian-trees
 * @index function Dirichlet leaf model (Beta-Bernoulli for two classes): exact evidence for a classification leaf.
 * @example
 * ```ts
 * import { dirichletLeaf } from 'omai-engine/trees'
 * import { sum, toFlat } from 'omai-engine/tensor'
 * const model = dirichletLeaf(2)
 * const leaf = sum(model.statistics([0, 0, 1]), 0)
 * toFlat(leaf) // => [2, 1]
 * model.describe(toFlat(leaf)).value // => [0.6, 0.4]
 * ```
 */
export function dirichletLeaf(classes: number, concentration = 1): LeafModel {
  if (!(Number.isInteger(classes) && classes >= 2 && concentration > 0)) {
    throw new OmaiError('value', `dirichletLeaf: need an integer number of classes >= 2 and a positive concentration`)
  }
  const total = concentration * classes
  const summary = (stats: Value): Value => div(add(stats, concentration), add(sum(stats, -1, true), total))
  return {
    kind: 'dirichlet',
    size: classes,
    outputs: classes,
    prior: { classes, concentration },
    statistics: (y) => {
      // A label outside 0..K-1 would be a row of zeros: dropped from the counts, the evidence and the minimum leaf size.
      const bad = toFlat(asTensor(y)).find((c) => !Number.isInteger(c) || c < 0 || c >= classes)
      if (bad !== undefined)
        throw new OmaiError(
          'value',
          `dirichletLeaf: label ${bad} is not a class: the labels are the integers 0 to ${classes - 1} (for ${classes} classes)`,
          { label: bad, classes },
        )
      return oneHot(y, classes)
    },
    count: (stats) => sum(stats, -1),
    logEvidence: (stats) => {
      const n = sum(stats, -1)
      const head = sub(lgamma(total), lgamma(add(n, total)))
      const body = sum(sub(lgamma(add(stats, concentration)), lgamma(concentration)), -1)
      return add(head, body)
    },
    summary,
    describe: (stats) => {
      const n = stats.reduce((a, b) => a + b, 0)
      const precision = n + total
      return {
        n,
        value: toFlat(summary(f64(stats))),
        precision,
        stepSize: 1 / precision,
      }
    },
    read: (averaged) => {
      const grid = toArray(averaged) as number[][]
      return { mean: grid.map((p) => p[classes - 1] ?? 0), probabilities: grid }
    },
  }
}

/** Slice index `i` of the last axis, as `[...]`. */
function lastAxis(x: Value, i: number): Value {
  const v = sliceAxis(x, -1, i, i + 1)
  return sum(v, -1)
}
