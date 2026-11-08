/**
 * Estimators of the gradient of an expectation `E_{x ~ p_theta}[f(x)]` with respect to the parameters of the
 * distribution: the score-function (REINFORCE, likelihood-ratio) estimator with an optional baseline, the pathwise
 * (reparameterisation) estimator, the straight-through estimator for discrete variables, and the Gumbel-softmax
 * relaxation. Each draws `n` independent samples from keys derived from one key and returns the mean gradient together
 * with the per-sample variance and the standard error, so the estimators can be compared with each other and with an exact
 * gradient. The distributions and their `rsample` come from `dists`; the differentiation from `autodiff`.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import { meanOf, OmaiError, varianceOf } from '../base/index.ts'
import { type Distribution, relaxedOneHotCategorical, straightThrough } from '../dists/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { asTensor, mul, stopGradient, sum, type TensorLike, toFlat, type Value } from '../tensor/index.ts'

/** The parameters of the distribution, by name; the gradient comes back with the same names. */
export type DistributionParams<P extends string = string> = Readonly<Record<P, TensorLike>>

/** Options shared by the gradient estimators. */
export interface GradientEstimatorOptions<P extends string = string> {
  /** Builds the distribution from the (traced) parameters. */
  readonly dist: (params: Record<P, Value>) => Distribution
  /** The function whose expectation is differentiated, of one sample (shape batch + event), returning a scalar. */
  readonly fn: (x: Value) => Value
  /** The parameters at which the gradient is taken. */
  readonly params: DistributionParams<P>
  /** Random key; sample `i` uses `foldIn(key, i)`. */
  readonly key: Key
  /** Number of samples (at least 2). */
  readonly n: number
}

/** A gradient estimate. */
export interface GradientEstimate {
  /** The estimator's name. */
  readonly method: string
  /** Samples used. */
  readonly n: number
  /** The mean gradient, flattened, by parameter name. */
  readonly gradient: Readonly<Record<string, Float64Array>>
  /** The variance of a single-sample gradient estimate, by parameter name and coordinate. */
  readonly variance: Readonly<Record<string, Float64Array>>
  /** The standard error of the mean gradient: `sqrt(variance / n)`. */
  readonly standardError: Readonly<Record<string, Float64Array>>
  /** The baseline subtracted from `f` (score-function estimators with a constant or batch-mean baseline). */
  readonly baseline?: number
}

/** The baseline of the score-function estimator. */
export type Baseline = 'none' | 'mean' | 'leave-one-out' | number

/** Options for {@link scoreFunctionGradient}. */
export interface ScoreFunctionOptions<P extends string = string> extends GradientEstimatorOptions<P> {
  /**
   * The baseline `b` in `(f - b) grad log p`: `'none'` (default), `'mean'` (the batch mean of `f`, with an `O(1/n)`
   * bias), `'leave-one-out'` (for sample `i` the mean of the others, which is exactly unbiased) or a constant.
   */
  readonly baseline?: Baseline
}

function tensors<P extends string>(params: DistributionParams<P>): Record<P, Value> {
  return Object.fromEntries(
    Object.entries<TensorLike>(params).map(([name, v]) => [name, asTensor(v) as Value]),
  ) as Record<P, Value>
}

function checkEstimator<P extends string>(options: GradientEstimatorOptions<P>, who: string): void {
  if (!(Number.isInteger(options.n) && options.n >= 2)) {
    throw new OmaiError('value', `${who}: n must be an integer of at least 2, got ${options.n}`)
  }
}

function summarise(
  method: string,
  names: readonly string[],
  rows: readonly Record<string, number[]>[],
  baseline?: number,
): GradientEstimate {
  const n = rows.length
  const gradient: Record<string, Float64Array> = {}
  const variance: Record<string, Float64Array> = {}
  const standardError: Record<string, Float64Array> = {}
  for (const name of names) {
    const width = rows[0]?.[name]?.length ?? 0
    gradient[name] = new Float64Array(width)
    variance[name] = new Float64Array(width)
    standardError[name] = new Float64Array(width)
    for (let c = 0; c < width; c++) {
      const column = rows.map((r) => r[name]?.[c] ?? 0)
      const v = varianceOf(column, 1)
      ;(gradient[name] as Float64Array)[c] = meanOf(column)
      ;(variance[name] as Float64Array)[c] = v
      ;(standardError[name] as Float64Array)[c] = Math.sqrt(v / n)
    }
  }
  return { method, n, gradient, variance, standardError, ...(baseline === undefined ? {} : { baseline }) }
}

function flatGradient<P extends string>(g: Record<P, Value>, names: readonly P[]): Record<string, number[]> {
  return Object.fromEntries(names.map((name) => [name, toFlat(g[name] as Value)]))
}

/**
 * The score-function (REINFORCE, likelihood-ratio) gradient estimate `mean_i (f(x_i) - b) grad_theta log p_theta(x_i)`
 * with `x_i ~ p_theta`. It needs only `sample` and `logProb`, so it works for discrete distributions and for functions
 * `f` that are not differentiable, at the price of a large variance that a baseline reduces without adding bias (the
 * score has mean zero, so a constant `b` leaves the expectation unchanged).
 *
 * @param options - Distribution builder, function, parameters, key, `n` and baseline.
 * @returns The mean gradient with its per-sample variance and standard error.
 * @throws OmaiError with code `value` for `n < 2` or an unknown baseline.
 * @concept gradient-estimators
 * @reference R. J. Williams, Simple statistical gradient-following algorithms for connectionist reinforcement learning
 *   (Machine Learning, 1992); S. Mohamed et al., Monte Carlo gradient estimation in machine learning (JMLR, 2020).
 * @example
 * ```ts
 * import { scoreFunctionGradient } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { square } from 'omai-engine/tensor'
 * const e = scoreFunctionGradient({
 *   dist: (p) => normal(p.loc, p.scale),
 *   fn: (x) => square(x),
 *   params: { loc: 1, scale: 1 },
 *   key: key(0),
 *   n: 4000,
 *   baseline: 'leave-one-out',
 * })
 * // d/dloc E[x^2] = 2 loc = 2, within a few standard errors
 * Math.abs((e.gradient['loc']?.[0] ?? 0) - 2) < 4 * (e.standardError['loc']?.[0] ?? 0) // => true
 * ```
 */
export function scoreFunctionGradient<P extends string>(options: ScoreFunctionOptions<P>): GradientEstimate {
  checkEstimator(options, 'scoreFunctionGradient')
  const { n, fn } = options
  const baseline = options.baseline ?? 'none'
  if (typeof baseline === 'string' && baseline !== 'none' && baseline !== 'mean' && baseline !== 'leave-one-out') {
    throw new OmaiError('value', `scoreFunctionGradient: unknown baseline ${baseline}`)
  }
  const params = tensors(options.params)
  const names = Object.keys(params) as P[]
  const draws = Array.from({ length: n }, (_, i) => stopGradient(options.dist(params).sample(foldIn(options.key, i))))
  const f = draws.map((x) => toFlat(fn(x))[0] ?? Number.NaN)
  const total = f.reduce((a, b) => a + b, 0)
  const constant = typeof baseline === 'number' ? baseline : baseline === 'mean' ? total / n : 0
  const rows = draws.map((x, i) => {
    const b = baseline === 'leave-one-out' ? (total - (f[i] ?? 0)) / (n - 1) : constant
    const weight = (f[i] ?? 0) - b
    const { grad } = valueAndGrad((p: Record<P, Value>) => mul(sum(options.dist(p).logProb(x)), weight))(params)
    return flatGradient(grad, names)
  })
  return summarise(
    `score function (baseline ${typeof baseline === 'number' ? baseline : baseline})`,
    names,
    rows,
    baseline === 'mean' || typeof baseline === 'number' ? constant : undefined,
  )
}

/**
 * The pathwise (reparameterisation) gradient estimate `mean_i grad_theta f(g_theta(eps_i))`: each sample is drawn with
 * `rsample` as a differentiable function of the parameters and `f` is differentiated through it. Unbiased whenever `f`
 * is differentiable and the distribution has a reparameterisation, and typically far lower variance than the score
 * function.
 *
 * @param options - Distribution builder (its `rsample` must exist), function, parameters, key and `n`.
 * @returns The mean gradient with its per-sample variance and standard error.
 * @throws OmaiError with code `value` for `n < 2`, or from `rsample` of a distribution without one.
 * @concept gradient-estimators
 * @reference D. P. Kingma and M. Welling, Auto-encoding variational Bayes (ICLR 2014).
 * @example
 * ```ts
 * import { pathwiseGradient } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { square } from 'omai-engine/tensor'
 * const e = pathwiseGradient({
 *   dist: (p) => normal(p.loc, p.scale),
 *   fn: (x) => square(x),
 *   params: { loc: 1, scale: 1 },
 *   key: key(0),
 *   n: 4000,
 * })
 * Math.abs((e.gradient['scale']?.[0] ?? 0) - 2) < 4 * (e.standardError['scale']?.[0] ?? 0) // => true
 * ```
 */
export function pathwiseGradient<P extends string>(options: GradientEstimatorOptions<P>): GradientEstimate {
  checkEstimator(options, 'pathwiseGradient')
  const params = tensors(options.params)
  const names = Object.keys(params) as P[]
  const rows = Array.from({ length: options.n }, (_, i) => {
    const { grad } = valueAndGrad((p: Record<P, Value>) =>
      sum(options.fn(options.dist(p).rsample(foldIn(options.key, i)))),
    )(params)
    return flatGradient(grad, names)
  })
  return summarise('pathwise', names, rows)
}

/**
 * The straight-through gradient estimate for a discrete distribution (Bengio, Leonard and Courville 2013): the forward
 * pass uses the hard sample (a 0/1 value or a one-hot vector), the backward pass the gradient of the differentiable
 * surrogate (the probabilities, or the relaxed sample). It is biased (it returns the slope of `f` at the hard sample, not
 * the slope of the expectation); compare it with {@link scoreFunctionGradient}, which is
 * unbiased, on the same distribution.
 *
 * @param options - Builder of a `bernoulli`, `oneHotCategorical`, `relaxedBernoulli` or `relaxedOneHotCategorical`, the
 *   function, parameters, key and `n`.
 * @returns The mean gradient with its per-sample variance and standard error.
 * @throws OmaiError with code `value` for `n < 2` or a distribution `straightThrough` does not support.
 * @concept gradient-estimators
 * @reference Y. Bengio, N. Leonard and A. Courville, Estimating or propagating gradients through stochastic neurons
 *   for conditional computation (arXiv 1308.3432, 2013).
 * @example
 * ```ts
 * import { straightThroughGradient } from 'omai-engine/sampling'
 * import { bernoulli } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { mul } from 'omai-engine/tensor'
 * // E[3 x] = 3 p for x ~ Bernoulli(p): the gradient with respect to p is exactly 3.
 * const e = straightThroughGradient({ dist: (p) => bernoulli({ probs: p.probs }), fn: (x) => mul(x, 3), params: { probs: 0.3 }, key: key(0), n: 50 })
 * e.gradient['probs']?.[0]?.toFixed(6) // => '3.000000'
 * ```
 */
export function straightThroughGradient<P extends string>(options: GradientEstimatorOptions<P>): GradientEstimate {
  const through = pathwiseGradient({ ...options, dist: (p) => straightThrough(options.dist(p)) })
  return { ...through, method: 'straight-through' }
}

/** Options for {@link gumbelSoftmaxGradient}. */
export interface GumbelSoftmaxOptions {
  /** The function of one relaxed (or hard) one-hot sample, returning a scalar. */
  readonly fn: (x: Value) => Value
  /** The categorical logits. */
  readonly logits: TensorLike
  /** The softmax temperature `tau > 0`: small values are nearly discrete with large gradient variance. */
  readonly temperature: number
  /** Round the forward pass to a one-hot vector and keep the relaxed gradient (straight-through Gumbel-softmax). */
  readonly hard?: boolean
  /** Random key. */
  readonly key: Key
  /** Number of samples (at least 2). */
  readonly n: number
}

/**
 * The Gumbel-softmax (concrete) gradient estimate (Jang, Gu and Poole 2017; Maddison, Mnih and Teh 2017): a categorical
 * sample is replaced by `softmax((logits + Gumbel noise) / tau)`, which is differentiable in the logits, and `f` is
 * differentiated through it. As `tau -> 0` the sample approaches one-hot and the estimate approaches the true gradient
 * but with exploding variance; with `hard` the forward value is the one-hot argmax.
 *
 * @param options - Function, logits, temperature, `hard`, key and `n`.
 * @returns The mean gradient of `E f(x)` with respect to `logits` (under the name `logits`), with variance and error.
 * @throws OmaiError with code `value` for `n < 2` or a non-positive temperature.
 * @concept gradient-estimators
 * @reference E. Jang, S. Gu and B. Poole, Categorical reparameterization with Gumbel-softmax (ICLR 2017).
 * @example
 * ```ts
 * import { gumbelSoftmaxGradient } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * import { sum, mul } from 'omai-engine/tensor'
 * // f(x) = x . [1, 2, 3]: the gradient of E[f] over softmax(logits) is positive along the larger payoffs.
 * const e = gumbelSoftmaxGradient({ fn: (x) => sum(mul(x, [1, 2, 3])), logits: [0, 0, 0], temperature: 0.5, key: key(0), n: 500 })
 * (e.gradient['logits']?.[2] ?? 0) > (e.gradient['logits']?.[0] ?? 0) // => true
 * ```
 */
export function gumbelSoftmaxGradient(options: GumbelSoftmaxOptions): GradientEstimate {
  if (!(options.temperature > 0)) {
    throw new OmaiError('value', `gumbelSoftmaxGradient: temperature must be positive, got ${options.temperature}`)
  }
  const build = (p: Record<'logits', Value>): Distribution => {
    const relaxed = relaxedOneHotCategorical(options.temperature, { logits: p.logits })
    return options.hard === true ? straightThrough(relaxed) : relaxed
  }
  const estimate = pathwiseGradient({
    dist: build,
    fn: options.fn,
    params: { logits: options.logits },
    key: options.key,
    n: options.n,
  })
  return { ...estimate, method: options.hard === true ? 'gumbel-softmax (hard)' : 'gumbel-softmax' }
}

/**
 * The Euclidean distance between an estimated gradient and an exact one, summed over the parameters: the bias plus noise
 * of a single estimate, to be averaged over batches to separate the two.
 *
 * @param estimate - A gradient estimate.
 * @param exact - The exact gradient by parameter name.
 * @returns `sqrt(sum (estimate - exact)^2)`.
 * @throws OmaiError with code `shape` for a missing parameter or a length mismatch.
 * @concept gradient-estimators
 * @example
 * ```ts
 * import { gradientError } from 'omai-engine/sampling'
 * const est = { method: 'x', n: 2, gradient: { a: Float64Array.of(1, 2) }, variance: {}, standardError: {} }
 * gradientError(est, { a: [1, 4] }) // => 2
 * ```
 */
export function gradientError(estimate: GradientEstimate, exact: Readonly<Record<string, ArrayLike<number>>>): number {
  let total = 0
  for (const [name, g] of Object.entries(estimate.gradient)) {
    const e = exact[name]
    if (e === undefined || e.length !== g.length)
      throw new OmaiError('shape', `gradientError: no exact gradient of the right length for ${name}`)
    for (let i = 0; i < g.length; i++) total += ((g[i] ?? 0) - (e[i] ?? 0)) ** 2
  }
  return Math.sqrt(total)
}
