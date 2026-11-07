import { MdmlError, type Reduction } from '../base/index.ts'
import { broadcastShapes, broadcastTo, div, mean, mul, sum, type Value } from '../tensor/index.ts'

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
      throw new MdmlError('value', `unknown reduction ${String(options.reduction)}`)
  }
}
