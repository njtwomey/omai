/**
 * Moving between a distribution's constrained support and unconstrained reals, as NumPyro's `biject_to`:
 * `bijectorForSupport` picks the bijector onto a support, and `transformLogDensity` and `unconstrained` apply the
 * change of variables so that samplers and optimisers can work on the whole real line.
 */

import { OmaiError } from '../../base/index.ts'
import { add, sub, type TensorLike, type Value } from '../../tensor/index.ts'
import type { Distribution, Support } from '../distribution.ts'
import { type Bijector, chain } from './bijector.ts'
import { affine, exp, identity, sigmoid, tanh } from './elementwise.ts'
import { choleskyOuterProduct, correlationCholesky, lowerCholesky, ordered, stickBreaking } from './structured.ts'

/**
 * The bijector from unconstrained reals onto a support, as samplers and variational families need (NumPyro's
 * `biject_to`, TFP's default event-space bijectors): identity on the reals; exp onto (0, inf) and [0, inf); sigmoid
 * then affine onto intervals; exp then affine above or below a bound; sigmoid onto (0, 1); pi tanh onto angles;
 * stick-breaking onto the simplex; ordered onto increasing vectors; lowerCholesky, correlationCholesky and the outer
 * products onto Cholesky factors, covariances and correlations.
 *
 * @param support - The support.
 * @returns The bijector from the unconstrained space onto the support.
 * @throws OmaiError with code `not-implemented` for discrete supports and the sphere (no bijection from R^n).
 * @example
 * ```ts
 * import { bijectors } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(bijectors.bijectorForSupport({ kind: 'positive' }).forward(0)) // => 1
 * ```
 */
export function bijectorForSupport(support: Support): Bijector {
  switch (support.kind) {
    case 'real':
      return identity()
    case 'positive':
    case 'nonNegative':
      return exp()
    case 'unitInterval':
      return sigmoid()
    case 'interval':
      return chain([sigmoid(), affine(support.low, sub(support.high, support.low))])
    case 'greaterThan':
      return chain([exp(), affine(support.low, 1)])
    case 'lessThan':
      return chain([exp(), affine(support.high, -1)])
    case 'circular':
      return chain([tanh(), affine(0, Math.PI)])
    case 'simplex':
      return stickBreaking()
    case 'ordered':
      return ordered()
    case 'lowerCholesky':
      return lowerCholesky()
    case 'correlationCholesky':
      return correlationCholesky()
    case 'positiveDefinite':
      return chain([lowerCholesky(), choleskyOuterProduct()])
    case 'correlation':
      return chain([correlationCholesky(), choleskyOuterProduct('correlation')])
    default:
      throw new OmaiError(
        'not-implemented',
        `bijectorForSupport: the ${support.kind} support has no bijection from unconstrained reals`,
      )
  }
}

/**
 * The log density of u = f^{-1}(x) for x with log density `logDensity`, where f is the bijector: log p(f(u)) +
 * log |det df/du|, the change of variables that lets a sampler or optimiser work on unconstrained reals.
 *
 * @param logDensity - log p(x) on the constrained space.
 * @param bijector - f, from the unconstrained space onto the support (see {@link bijectorForSupport}).
 * @returns The log density on the unconstrained space.
 * @example
 * ```ts
 * import { bijectors, exponential } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const d = exponential(1)
 * const onLine = bijectors.transformLogDensity((x) => d.logProb(x), bijectors.exp())
 * item(onLine(0)) // => -1
 * ```
 */
export function transformLogDensity(logDensity: (x: Value) => Value, bijector: Bijector): (u: TensorLike) => Value {
  return (u) => add(logDensity(bijector.forward(u)), bijector.forwardLogDetJacobian(u))
}

/**
 * A distribution's log density on unconstrained reals: `transformLogDensity` with the bijector for its support, plus
 * the bijector (to map draws back with `forward`).
 *
 * @param dist - The distribution.
 * @returns The bijector and the unconstrained log density.
 * @example
 * ```ts
 * import { bijectors, beta } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const { logDensity } = bijectors.unconstrained(beta(2, 2))
 * // At u = 0: log p(1/2) + log(1/4) = log 1.5 - log 4.
 * Math.abs(item(logDensity(0)) - Math.log(1.5 / 4)) < 1e-14 // => true
 * ```
 */
export function unconstrained(dist: Distribution): {
  readonly bijector: Bijector
  readonly logDensity: (u: TensorLike) => Value
} {
  const bijector = bijectorForSupport(dist.support)
  return { bijector, logDensity: transformLogDensity((x) => dist.logProb(x), bijector) }
}
