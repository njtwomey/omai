import { WhiteboxError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import {
  asTensor,
  broadcastShapes,
  exp,
  log,
  log1p,
  neg,
  sqrt,
  stopGradient,
  sub,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'

/** The set a distribution's values lie in. */
export type Support =
  | { readonly kind: 'real' }
  | { readonly kind: 'positive' }
  | { readonly kind: 'nonNegative' }
  | { readonly kind: 'interval'; readonly low: Value; readonly high: Value }
  | { readonly kind: 'unitInterval' }
  | { readonly kind: 'integerInterval'; readonly low: number; readonly high: number }
  | { readonly kind: 'boolean' }
  | { readonly kind: 'simplex' }

/**
 * A probability distribution: a plain object with its parameters (values, possibly traced, so everything differentiates
 * with respect to them) and pure methods. Shapes follow torch.distributions: a sample of shape `s` has shape
 * `[...s, ...batchShape, ...eventShape]`.
 *
 * `rsample` draws samples that are differentiable functions of the parameters (pathwise gradients); `sample` is the
 * same draw with gradients stopped. Distributions without a reparameterisation (`hasRsample` false) throw from
 * `rsample`.
 *
 * @concept distributions
 */
export interface Distribution {
  readonly name: string
  readonly params: Readonly<Record<string, Value>>
  readonly support: Support
  readonly batchShape: readonly number[]
  readonly eventShape: readonly number[]
  readonly hasRsample: boolean
  logProb(x: TensorLike): Value
  prob(x: TensorLike): Value
  cdf(x: TensorLike): Value
  logCdf(x: TensorLike): Value
  sf(x: TensorLike): Value
  logSf(x: TensorLike): Value
  ppf(q: TensorLike): Value
  isf(q: TensorLike): Value
  sample(key: Key, shape?: readonly number[]): Value
  rsample(key: Key, shape?: readonly number[]): Value
  mean(): Value
  variance(): Value
  stddev(): Value
  mode(): Value
  entropy(): Value
}

/** What a distribution must define; everything else in {@link Distribution} has a default. */
export interface DistributionSpec {
  readonly name: string
  readonly params: Readonly<Record<string, Value>>
  readonly support: Support
  readonly batchShape: readonly number[]
  readonly eventShape?: readonly number[]
  logProb(x: Value): Value
  cdf?(x: Value): Value
  logCdf?(x: Value): Value
  sf?(x: Value): Value
  logSf?(x: Value): Value
  ppf?(q: Value): Value
  isf?(q: Value): Value
  /** Reparameterised sampling; when given, `sample` is its stop-gradient version. */
  rsample?(key: Key, shape: readonly number[]): Value
  /** Non-reparameterised sampling (discrete distributions). */
  sample?(key: Key, shape: readonly number[]): Value
  mean(): Value
  variance(): Value
  mode(): Value
  entropy(): Value
}

function missing(name: string, what: string): never {
  throw new WhiteboxError('not-implemented', `${name}: ${what} is not available for this distribution`)
}

/**
 * Build a distribution from its defining parts. Defaults: `prob = exp(logProb)`, `sf = 1 - cdf`, `logCdf = log(cdf)`,
 * `logSf = log1p(-cdf)`, `isf(q) = ppf(1 - q)`, `stddev = sqrt(variance)`, `sample = stopGradient(rsample)`.
 *
 * @param spec - The defining parts.
 * @returns The distribution.
 * @concept distributions
 * @example
 * ```ts
 * import { defineDistribution } from 'whitebox-engine/dists'
 * import { tensor, neg, item } from 'whitebox-engine/tensor'
 * const unitExponential = defineDistribution({
 *   name: 'unitExponential', params: {}, support: { kind: 'positive' }, batchShape: [],
 *   logProb: (x) => neg(x), mean: () => tensor(1), variance: () => tensor(1), mode: () => tensor(0),
 *   entropy: () => tensor(1),
 * })
 * item(unitExponential.prob(0)) // => 1
 * ```
 */
export function defineDistribution(spec: DistributionSpec): Distribution {
  const value = (x: TensorLike): Value => asTensor(x)
  const cdf = (x: Value): Value => (spec.cdf ? spec.cdf(x) : missing(spec.name, 'cdf'))
  const ppf = (q: Value): Value => (spec.ppf ? spec.ppf(q) : missing(spec.name, 'ppf'))
  const rsample = (key: Key, shape: readonly number[] = []): Value =>
    spec.rsample
      ? spec.rsample(key, shape)
      : missing(spec.name, 'rsample (no reparameterisation; use a relaxed distribution or a score-function estimator)')
  return {
    name: spec.name,
    params: spec.params,
    support: spec.support,
    batchShape: spec.batchShape,
    eventShape: spec.eventShape ?? [],
    hasRsample: spec.rsample !== undefined,
    logProb: (x) => spec.logProb(value(x)),
    prob: (x) => exp(spec.logProb(value(x))),
    cdf: (x) => cdf(value(x)),
    logCdf: (x) => (spec.logCdf ? spec.logCdf(value(x)) : log(cdf(value(x)))),
    sf: (x) => (spec.sf ? spec.sf(value(x)) : sub(1, cdf(value(x)))),
    logSf: (x) => (spec.logSf ? spec.logSf(value(x)) : log1p(neg(cdf(value(x))))),
    ppf: (q) => ppf(value(q)),
    isf: (q) => (spec.isf ? spec.isf(value(q)) : ppf(sub(1, value(q)))),
    sample: (key, shape = []) => (spec.sample ? spec.sample(key, shape) : stopGradient(rsample(key, shape))),
    rsample,
    mean: spec.mean,
    variance: spec.variance,
    stddev: () => sqrt(spec.variance()),
    mode: spec.mode,
    entropy: spec.entropy,
  }
}

/** The common batch shape of a distribution's parameters. */
export function batchShapeOf(...params: readonly Value[]): number[] {
  return params.reduce<number[]>((s, p) => broadcastShapes(s, p.shape), [])
}
