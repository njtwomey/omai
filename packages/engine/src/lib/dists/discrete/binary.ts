import { logSigmoid } from '../../numerics/special/index.ts'
import { type Key, uniform as uniformNoise } from '../../random/index.ts'
import {
  cast,
  equal,
  greater,
  log,
  log1p,
  mul,
  neg,
  promoteFloat,
  sigmoid,
  softplus,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { discreteSample } from '../_sampling.ts'
import { batchShapeOf, type Distribution, defineDistribution } from '../distribution.ts'

/** Parameters of a Bernoulli distribution: probabilities or logits. */
export type BernoulliParams = { readonly probs: TensorLike } | { readonly logits: TensorLike }

/**
 * The Bernoulli distribution on {0, 1} (scipy.stats.bernoulli, torch.distributions.Bernoulli).
 *
 * @param params - `{ probs }` or `{ logits }` (logits keep log-probabilities accurate near 0 and 1).
 * @returns The distribution (no reparameterisation: `rsample` throws; use `relaxedBernoulli`).
 * @concept distributions
 * @example
 * ```ts
 * import { bernoulli } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bernoulli({ probs: 0.25 }).mean()) // => 0.25
 * ```
 */
export function bernoulli(params: BernoulliParams): Distribution {
  const logits =
    'logits' in params
      ? (promoteFloat(params.logits)[0] as Value)
      : ((): Value => {
          const [p] = promoteFloat(params.probs) as [Value]
          return sub(log(p), log1p(neg(p)))
        })()
  const p = sigmoid(logits)
  const batchShape = batchShapeOf(logits)
  return defineDistribution({
    name: 'bernoulli',
    params: { logits, probs: p },
    support: { kind: 'boolean' },
    batchShape,
    logProb: (x) => {
      const [v] = promoteFloat(x) as [Value]
      return where(
        equal(v, 1),
        logSigmoid(logits),
        where(equal(v, 0), logSigmoid(neg(logits)), Number.NEGATIVE_INFINITY),
      )
    },
    cdf: (x) => {
      const [v] = promoteFloat(x) as [Value]
      return where(greater(0, v), 0, where(greater(1, v), sub(1, p), 1))
    },
    sample: (key: Key, shape) =>
      discreteSample([p], [...shape, ...batchShape], 'float64', ([probs]) => {
        const u = uniformNoise(key, [...shape, ...batchShape])
        return cast(greater(probs as Tensor, u), 'float64') as Tensor
      }),
    mean: () => p,
    variance: () => mul(p, sub(1, p)),
    mode: () => cast(greater(logits, 0), 'float64'),
    entropy: () => sub(softplus(logits), mul(p, logits)),
  })
}
