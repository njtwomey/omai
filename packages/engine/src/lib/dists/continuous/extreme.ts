import { uniform as uniformNoise } from '../../random/index.ts'
import {
  add,
  div,
  exp,
  log,
  mul,
  neg,
  promoteFloat,
  square,
  sub,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { batchShapeOf, type Distribution, defineDistribution } from '../distribution.ts'

const EULER = 0.5772156649015329

/**
 * The Gumbel distribution of maxima (scipy.stats.gumbel_r): F(x) = exp(-exp(-(x - loc) / scale)).
 *
 * @param loc - Location.
 * @param scale - Scale (positive).
 * @returns The distribution; `rsample` is loc - scale log(-log U).
 * @concept distributions
 * @example
 * ```ts
 * import { gumbel } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(gumbel(0, 1).cdf(0)) === Math.exp(-1) // => true
 * ```
 */
export function gumbel(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, beta] = promoteFloat(loc, scale) as [Value, Value]
  const batchShape = batchShapeOf(mu, beta)
  const z = (x: Value): Value => div(sub(x, mu), beta)
  const ppf = (q: Value): Value => sub(mu, mul(beta, log(neg(log(q)))))
  return defineDistribution({
    name: 'gumbel',
    params: { loc: mu, scale: beta },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => {
      const v = z(x)
      return sub(neg(add(v, exp(neg(v)))), log(beta))
    },
    cdf: (x) => exp(neg(exp(neg(z(x))))),
    logCdf: (x) => neg(exp(neg(z(x)))),
    ppf,
    rsample: (key, shape) => ppf(uniformNoise(key, [...shape, ...batchShape])),
    mean: () => add(mu, mul(beta, EULER)),
    variance: () => add(mul(square(beta), (Math.PI * Math.PI) / 6), mul(mu, 0)),
    mode: () => add(mu, mul(beta, 0)),
    entropy: () => add(add(log(beta), EULER + 1), mul(mu, 0)),
  })
}
