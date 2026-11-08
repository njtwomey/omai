import { OmaiError, type Reduction } from '../base/index.ts'
import {
  asTensor,
  broadcastShapes,
  broadcastTo,
  div,
  mean,
  mul,
  primalOf,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'

/** Options every loss accepts. */
export interface LossOptions {
  readonly reduction?: Reduction
  /** Per-element (or per-sample, broadcast) weights. A weighted mean divides by the sum of the weights. */
  readonly weight?: Value
}

/** Apply a reduction (and optional weights) to elementwise loss terms. */
export function reduce(terms: Value, options: LossOptions = {}): Value {
  const weighted = options.weight === undefined ? terms : mul(terms, options.weight)
  switch (options.reduction ?? 'mean') {
    case 'none':
      return weighted
    case 'sum':
      return sum(weighted)
    case 'mean':
      if (options.weight === undefined) return mean(weighted)
      return div(sum(weighted), sum(broadcastTo(options.weight, broadcastShapes(options.weight.shape, terms.shape))))
    default:
      throw new OmaiError('value', `unknown reduction ${String(options.reduction)}`)
  }
}

/**
 * Reject a hyperparameter outside its domain. An absent value is the default and passes; NaN and infinities are outside
 * every domain. `positive` is (0, inf), `nonNegative` [0, inf), `unit` [0, 1].
 */
export function checkScalar(
  loss: string,
  name: string,
  value: number | undefined,
  domain: 'positive' | 'nonNegative' | 'unit',
): void {
  if (value === undefined) return
  const ok =
    Number.isFinite(value) &&
    (domain === 'positive' ? value > 0 : domain === 'nonNegative' ? value >= 0 : value <= 1 && value >= 0)
  if (ok) return
  const want = { positive: 'positive', nonNegative: 'non-negative', unit: 'in [0, 1]' }[domain]
  throw new OmaiError('value', `${loss}: ${name} must be ${want}, got ${value}`)
}

/** Reject negative or non-finite entries of a weight vector (read from its primal when it is traced). */
export function checkWeights(loss: string, name: string, weights: TensorLike | undefined): void {
  if (weights === undefined) return
  if (toFlat(primalOf(asTensor(weights))).some((w) => !(w >= 0) || !Number.isFinite(w)))
    throw new OmaiError('value', `${loss}: ${name} must be non-negative and finite`)
}
