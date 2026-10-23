import { WhiteboxError } from '../../base/index.ts'
import { type Key, uniform as uniformNoise } from '../../random/index.ts'
import {
  arange,
  argmax,
  broadcastShapes,
  broadcastTo,
  cast,
  clip,
  cumsum,
  exp,
  expandDims,
  floor,
  greaterEqual,
  less,
  log,
  logicalOr,
  logSoftmax,
  mul,
  neg,
  promoteFloat,
  square,
  squeeze,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  takeAlongAxis,
  type Value,
  where,
} from '../../tensor/index.ts'
import { discreteSample } from '../_sampling.ts'
import { type Distribution, defineDistribution } from '../distribution.ts'

/** Parameters of a categorical distribution: probabilities (normalised along the last axis) or logits. */
export type CategoricalParams = { readonly probs: TensorLike } | { readonly logits: TensorLike }

/**
 * The categorical distribution over {0, ..., K-1}, with categories along the last axis (torch.distributions.Categorical).
 *
 * @param params - `{ probs }` (normalised for you) or `{ logits }`.
 * @returns The distribution (no reparameterisation: `rsample` throws; use `relaxedOneHotCategorical`).
 * @concept distributions
 * @example
 * ```ts
 * import { categorical } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(categorical({ probs: [0.2, 0.8] }).logProb(1)) - Math.log(0.8)) < 1e-15 // => true
 * ```
 */
export function categorical(params: CategoricalParams): Distribution {
  const raw =
    'logits' in params ? (promoteFloat(params.logits)[0] as Value) : log(promoteFloat(params.probs)[0] as Value)
  if (raw.shape.length === 0) throw new WhiteboxError('shape', 'categorical: parameters need a category axis')
  const logp = logSoftmax(raw, -1)
  const probs = exp(logp)
  const k = raw.shape[raw.shape.length - 1] ?? 0
  const batchShape = raw.shape.slice(0, -1)
  const index = arange(k)
  // Broadcast the per-category values to the shape of x and gather at x.
  const gather = (values: Value, x: Value): Value => {
    const shape = broadcastShapes(x.shape, batchShape)
    const table = broadcastTo(values, [...shape, k])
    const at = expandDims(broadcastTo(clip(x, 0, k - 1), shape), -1)
    return squeeze(takeAlongAxis(table, cast(at, 'int32'), -1), -1)
  }
  const cumulative = cumsum(probs, -1)
  return defineDistribution({
    name: 'categorical',
    params: { logits: logp, probs },
    support: { kind: 'integerInterval', low: 0, high: k - 1 },
    batchShape,
    logProb: (x) => {
      const v = promoteFloat(x)[0] as Value
      const invalid = logicalOr(logicalOr(less(v, 0), greaterEqual(v, k)), less(floor(v), v))
      return where(invalid, Number.NEGATIVE_INFINITY, gather(logp, v))
    },
    cdf: (x) => {
      const v = floor(promoteFloat(x)[0] as Value)
      return where(less(v, 0), 0, where(greaterEqual(v, k - 1), 1, gather(cumulative, v)))
    },
    ppf: (q) => {
      // The smallest k with cdf(k) >= q: the number of cumulative probabilities below q.
      const qv = promoteFloat(q)[0] as Value
      return sum(cast(less(cumulative, expandDims(qv, -1)), 'float64'), -1)
    },
    sample: (key: Key, shape) =>
      discreteSample([cumulative], [...shape, ...batchShape], 'float64', ([cdf]) => {
        const u = expandDims(uniformNoise(key, [...shape, ...batchShape]), -1)
        return sum(cast(less(cdf as Tensor, u), 'float64'), -1) as Tensor
      }),
    mean: () => sum(mul(probs, index), -1),
    variance: () => {
      const m = sum(mul(probs, index), -1)
      return sub(sum(mul(probs, square(index)), -1), square(m))
    },
    mode: () => cast(argmax(logp, -1), 'float64'),
    entropy: () => neg(sum(mul(probs, logp), -1)),
  })
}
