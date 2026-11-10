/**
 * The generalised linear model benchmarks: Bayesian logistic and probit regression with an intercept and two covariates,
 * and Poisson and negative binomial regression with a log link on one covariate. Each is a spec whose posterior is a
 * smooth log-concave density in two or three coordinates, so the truth is a grid quadrature (`gridPosterior`) and the
 * methods can be ranked against it: NUTS and tempered SMC agree, Laplace is close because the posterior is near
 * Gaussian, and a mean-field normal guide is a little too narrow where the coefficients are correlated. The probit
 * model is the first vertical slice of the plan (section 7.2). Data are simulated at fixed coefficients.
 */

import { negativeBinomial, poisson as poissonDist } from '../../dists/index.ts'
import { ndtr } from '../../numerics/special/index.ts'
import { normal as drawNormal, uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { item, tensor, toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type BenchmarkData, type BenchmarkModel, defineBenchmark } from './benchmark.ts'
import { importancePlan, laplacePlan, nutsPlan, sviPlan, temperedPlan } from './plans.ts'
import { gridTruth } from './truth.ts'

/** An intercept column and `d - 1` standard normal covariates, `n` rows. */
function designMatrix(key: Key, n: number, features: number): number[][] {
  const z = toFlat(drawNormal(key, [n, features - 1]))
  return Array.from({ length: n }, (_, i) => [
    1,
    ...Array.from({ length: features - 1 }, (_, j) => z[i * (features - 1) + j] ?? 0),
  ])
}

const dot = (row: readonly number[], w: readonly number[]): number => row.reduce((a, v, j) => a + v * (w[j] ?? 0), 0)

/** Binary labels from a link: label `i` is 1 with probability `link(eta_i)`. */
function labels(
  key: Key,
  X: readonly (readonly number[])[],
  w: readonly number[],
  link: (eta: number) => number,
): number[] {
  const u = toFlat(drawUniform(key, [X.length]))
  return X.map((row, i) => ((u[i] ?? 1) < link(dot(row, w)) ? 1 : 0))
}

const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v))
const probitLink = (v: number): number => item(ndtr(v))

const LOGISTIC_W = [0.3, 1.2, -0.8]

/**
 * Bayesian logistic regression: `y_i ~ Bernoulli(sigmoid(x_i . w))` with an intercept and two covariates, `w ~ N(0, 2²)`
 * (50 rows at the canonical size, generated at `w = (0.3, 1.2, -0.8)`). Three coordinates, so the truth is a grid; EP and
 * variational bounds are the methods the plan expects to compare later, and the Laplace fit and a mean-field guide
 * are the cheap approximations to read against it.
 *
 * @concept posterior-validation
 * @index dataset Bayesian logistic regression with an intercept and two covariates; the truth is a 3-D grid.
 * @example
 * ```ts
 * import { logisticRegression } from 'omai-engine/probprog/models'
 * logisticRegression.bound().sizes // => { row: 50, feature: 3 }
 * ```
 */
export const logisticRegression: BenchmarkModel = defineBenchmark({
  name: 'logistic-regression',
  title: 'Bayesian logistic regression',
  description:
    'Logistic regression with an intercept and two covariates and a N(0, 4) prior on the weights: a smooth three-dimensional posterior with a grid truth.',
  tags: ['glm', 'classification', 'logistic', 'quadrature', 'three-dimensional'],
  references: [
    'Gelman et al., Bayesian Data Analysis, 3rd ed., chapter 16 (generalised linear models)',
    'Minka (2001), A family of algorithms for approximate Bayesian inference, section 4 (Bayes point machine, EP on a classifier)',
  ],
  definition: defineModel('logistic-regression', (m) => {
    const n = m.range('row')
    const f = m.range('feature')
    const X = m.data('X', { dims: [n, f], role: 'covariate' })
    const y = m.data('y', { dims: [n], support: 'binary', role: 'response' })
    const w = m.latent('w', d.normal(0, 2), { dims: [f], doc: 'the intercept and the two slopes' })
    m.observe('y', d.bernoulli({ logits: X.matmul(w) }), y)
  }),
  generate: (key: Key, size = 50): BenchmarkData => {
    const X = designMatrix(foldIn(key, 0), size, 3)
    return { data: { X, y: labels(foldIn(key, 1), X, LOGISTIC_W, sigmoid) }, generating: { w: LOGISTIC_W } }
  },
  truth: (data, quality) =>
    gridTruth(
      logisticRegression.program(data),
      quality === 'fast' ? { points: 17, maxRefinements: 0 } : { points: 31, maxRefinements: 1 },
    ),
  plans: (scale) => [
    nutsPlan(scale),
    temperedPlan(scale),
    importancePlan(scale, {
      expect: 'report',
      note: 'weights from the prior: the posterior is concentrated, so the effective sample size is small',
    }),
    laplacePlan(scale, { note: 'near Gaussian at this sample size: the mean is close, the spread slightly off' }),
    sviPlan(scale, {
      note: 'a mean-field normal ignores the correlation of the intercept and the slopes: spread a little too small',
    }),
  ],
})

const PROBIT_W = [0.3, 1, -0.7]

/**
 * Bayesian probit regression, the slice model of the plan: `y_i ~ Bernoulli(Phi(x_i . w))`, an intercept and two
 * covariates, `w ~ N(0, 2²)` (50 rows, generated at `w = (0.3, 1, -0.7)`). Written with the `probit` operator, it is the
 * model EP is exact-ish on (the projection of each site is a normal), so the first thing message passing is checked
 * against; the truth is a grid.
 *
 * @concept posterior-validation
 * @index dataset Bayesian probit regression (the plan's slice model); the truth is a 3-D grid.
 * @example
 * ```ts
 * import { probitRegression } from 'omai-engine/probprog/models'
 * probitRegression.definition?.spec.nodes.map((n) => n.name) // => ['X', 'y', 'w', 'y']
 * ```
 */
export const probitRegression: BenchmarkModel = defineBenchmark({
  name: 'probit-regression',
  title: 'Bayesian probit regression',
  description:
    'Probit regression with an intercept and two covariates and a N(0, 4) prior: the slice model of the plan, the test of expectation propagation, with a grid truth.',
  tags: ['glm', 'classification', 'probit', 'quadrature', 'three-dimensional', 'slice'],
  references: [
    'Albert and Chib (1993), Bayesian analysis of binary and polychotomous response data',
    'Minka (2001), Expectation propagation for approximate Bayesian inference, section on the Bayes point machine',
    'Rasmussen and Williams (2006), Gaussian Processes for Machine Learning, section 3.6 (EP for probit classification)',
  ],
  definition: defineModel('probit-regression', (m) => {
    const n = m.range('row')
    const f = m.range('feature')
    const X = m.data('X', { dims: [n, f], role: 'covariate' })
    const y = m.data('y', { dims: [n], support: 'binary', role: 'response' })
    const w = m.latent('w', d.normal(0, 2), { dims: [f], doc: 'the intercept and the two slopes' })
    m.observe('y', d.bernoulli({ probs: ops.probit(X.matmul(w)) }), y)
  }),
  generate: (key: Key, size = 50): BenchmarkData => {
    const X = designMatrix(foldIn(key, 0), size, 3)
    return { data: { X, y: labels(foldIn(key, 1), X, PROBIT_W, probitLink) }, generating: { w: PROBIT_W } }
  },
  truth: (data, quality) =>
    gridTruth(
      probitRegression.program(data),
      quality === 'fast' ? { points: 17, maxRefinements: 0 } : { points: 31, maxRefinements: 1 },
    ),
  plans: (scale) => [
    nutsPlan(scale),
    temperedPlan(scale),
    importancePlan(scale, {
      expect: 'report',
      note: 'weights from the prior: the posterior is concentrated, so the effective sample size is small',
    }),
    laplacePlan(scale, { note: 'near Gaussian at this sample size: the mean is close, the spread slightly off' }),
    sviPlan(scale, {
      note: 'a mean-field normal ignores the correlation of the intercept and the slopes: spread a little too small',
    }),
  ],
})

const POISSON = { intercept: 0.3, slope: 0.8 }

/** Covariates uniform on [0, 3] and the log-link mean `exp(intercept + slope x)`. */
function logLinkCurve(key: Key, n: number, intercept: number, slope: number): { x: number[]; mu: number[] } {
  const x = Array.from(toFlat(drawUniform(key, [n], { low: 0, high: 3 })))
  return { x, mu: x.map((v) => Math.exp(intercept + slope * v)) }
}

/**
 * Poisson regression with a log link: `k_i ~ Poisson(exp(a + b x_i))`, `a, b ~ N(0, 2²)` (80 rows, `x` uniform on [0, 3],
 * generated at `a = 0.3`, `b = 0.8`, the curve of the library's `glmCurve`). Two coordinates, a grid truth; the
 * posterior is close to Gaussian, so Laplace and a mean-field guide do well, and importance sampling from the prior
 * has a small effective sample size.
 *
 * @concept posterior-validation
 * @index dataset Poisson regression with a log link on one covariate; the truth is a 2-D grid.
 * @example
 * ```ts
 * import { poissonRegression } from 'omai-engine/probprog/models'
 * poissonRegression.bound().sizes // => { row: 80 }
 * ```
 */
export const poissonRegression: BenchmarkModel = defineBenchmark({
  name: 'poisson-regression',
  title: 'Poisson regression',
  description:
    'Counts with a log-linear mean in one covariate and N(0, 4) priors on the intercept and the slope: a two-dimensional posterior with a grid truth.',
  tags: ['glm', 'counts', 'poisson', 'log-link', 'quadrature', 'two-dimensional'],
  references: [
    'Gelman et al., Bayesian Data Analysis, 3rd ed., section 16.3 (Poisson regression)',
    'the `glmCurve` generator of omai-engine/datasets (same mean curve)',
  ],
  definition: defineModel('poisson-regression', (m) => {
    const n = m.range('row')
    const x = m.data('x', { dims: [n], role: 'covariate' })
    const k = m.data('k', { dims: [n], support: 'count', role: 'response' })
    const a = m.latent('a', d.normal(0, 2), { doc: 'the log rate at x = 0' })
    const b = m.latent('b', d.normal(0, 2), { doc: 'the slope of the log rate' })
    // The linear predictor is clipped to [-30, 30]: a wild leapfrog step of the step-size search must give a huge but finite
    // rate, since the distribution constructors refuse an infinite one (the clip never binds where the posterior has mass).
    m.observe('k', d.poisson(ops.exp(ops.clip(a.add(b.mul(x)), -30, 30))), k)
  }),
  generate: (key: Key, size = 80): BenchmarkData => {
    const { x, mu } = logLinkCurve(foldIn(key, 0), size, POISSON.intercept, POISSON.slope)
    const k = Array.from(toFlat(poissonDist(tensor(mu)).sample(foldIn(key, 1))))
    return { data: { x, k }, generating: { ...POISSON } }
  },
  truth: (data, quality) =>
    gridTruth(
      poissonRegression.program(data),
      quality === 'fast' ? { points: 41, maxRefinements: 0 } : { points: 81, maxRefinements: 1 },
    ),
  plans: (scale) => [
    nutsPlan(scale),
    temperedPlan(scale),
    importancePlan(scale, {
      expect: 'report',
      note: 'weights from the prior: the posterior is concentrated, so the effective sample size is small',
    }),
    laplacePlan(scale, { note: 'near Gaussian: close to the grid' }),
    sviPlan(scale, {
      note: 'a mean-field normal: close in the means, spread a little too small where a and b are correlated',
    }),
  ],
})

const NEGBIN = { intercept: 0.3, slope: 0.8, alpha: 0.5 }

/**
 * Negative binomial regression with a log link: `k_i ~ NegBin(mean exp(a + b x_i), variance mean + alpha mean²)`,
 * `a, b ~ N(0, 2²)`, `alpha ~ HalfNormal(1)` (100 rows, generated at `a = 0.3`, `b = 0.8`, `alpha = 0.5`). The
 * overdispersion `alpha` is a positive scale with a skewed posterior, so the Laplace fit on its log and a mean-field
 * guide miss in the tail where NUTS and tempered SMC agree with the three-dimensional grid.
 *
 * @concept posterior-validation
 * @index dataset Negative binomial regression with an overdispersion parameter; the truth is a 3-D grid.
 * @example
 * ```ts
 * import { negativeBinomialRegression } from 'omai-engine/probprog/models'
 * negativeBinomialRegression.definition?.spec.nodes.filter((n) => n.kind === 'latent').map((n) => n.name) // => ['a', 'b', 'alpha']
 * ```
 */
export const negativeBinomialRegression: BenchmarkModel = defineBenchmark({
  name: 'negative-binomial-regression',
  title: 'Negative binomial regression',
  description:
    'Overdispersed counts with a log-linear mean and a HalfNormal(1) overdispersion: three coordinates, one of them a skewed positive scale, with a grid truth.',
  tags: ['glm', 'counts', 'negative-binomial', 'overdispersion', 'log-link', 'quadrature', 'three-dimensional'],
  references: [
    'Gelman et al., Bayesian Data Analysis, 3rd ed., section 16.3 (overdispersed Poisson and negative binomial regression)',
    'the `glmCurve` generator of omai-engine/datasets (same mean curve and dispersion)',
  ],
  definition: defineModel('negative-binomial-regression', (m) => {
    const n = m.range('row')
    const x = m.data('x', { dims: [n], role: 'covariate' })
    const k = m.data('k', { dims: [n], support: 'count', role: 'response' })
    const a = m.latent('a', d.normal(0, 2), { doc: 'the log mean at x = 0' })
    const b = m.latent('b', d.normal(0, 2), { doc: 'the slope of the log mean' })
    const alpha = m.latent('alpha', d.halfNormal(1), { doc: 'overdispersion: variance = mean + alpha mean^2' })
    // The linear predictor and the overdispersion are clipped ([-30, 30] and [1e-6, 1e6]) in the likelihood so that a wild
    // leapfrog step of the step-size search gives a huge but finite mean and a positive probability, which the
    // distribution constructors require; the clip never binds where the posterior has mass.
    const mean = ops.exp(ops.clip(a.add(b.mul(x)), -30, 30))
    const over = ops.clip(alpha, 1e-6, 1e6)
    // NB2: r = 1 / alpha successes, success probability r / (r + mean) = 1 / (1 + alpha mean).
    m.observe('k', d.negativeBinomial(ops.div(1, over), ops.div(1, ops.add(1, over.mul(mean)))), k)
  }),
  generate: (key: Key, size = 100): BenchmarkData => {
    const { x, mu } = logLinkCurve(foldIn(key, 0), size, NEGBIN.intercept, NEGBIN.slope)
    const r = 1 / NEGBIN.alpha
    const k = Array.from(
      toFlat(negativeBinomial(tensor(mu.map(() => r)), tensor(mu.map((v) => r / (r + v)))).sample(foldIn(key, 1))),
    )
    return { data: { x, k }, generating: { ...NEGBIN } }
  },
  truth: (data, quality) =>
    gridTruth(
      negativeBinomialRegression.program(data),
      quality === 'fast' ? { points: 17, maxRefinements: 0 } : { points: 31, maxRefinements: 1 },
    ),
  plans: (scale) => [
    nutsPlan(scale),
    temperedPlan(scale),
    importancePlan(scale, {
      expect: 'report',
      note: 'weights from the prior: the posterior is concentrated, so the effective sample size is small',
    }),
    laplacePlan(scale, { note: 'a normal on log alpha: misses the skew of the overdispersion' }),
    sviPlan(scale, {
      note: 'a mean-field normal on log alpha: misses the skew, and the correlation with the intercept',
    }),
  ],
})
