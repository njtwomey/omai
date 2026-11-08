/**
 * The `Bijector` contract: an invertible, differentiable map with a tractable log-Jacobian determinant, as TensorFlow
 * Probability's Bijector and torch.distributions.Transform, which turns one distribution into another by a change of
 * variables. `defineBijector` builds one from its forward and inverse maps, and `chain` and `invert` compose and
 * reverse them.
 *
 * @index structure The Bijector contract: invertible maps with log-Jacobian, for flows.
 */

import { OmaiError } from '../../base/index.ts'
import { add, asTensor, neg, type TensorLike, type Value } from '../../tensor/index.ts'
import { type Support, sumRightmost } from '../distribution.ts'

/**
 * An invertible, differentiable map with a tractable log-Jacobian determinant (TensorFlow Probability's Bijector,
 * torch.distributions.Transform). `inputRank` trailing axes of an input are transformed jointly (0 for elementwise
 * maps, 1 for vectors) into `outputRank` trailing axes; the log-determinants reduce those axes. `sign` is the sign of
 * the derivative of an elementwise map (1 increasing, -1 decreasing, 0 when unknown or not elementwise), which lets a
 * transformed distribution keep its CDF and quantiles.
 *
 * @concept normalizing-flows
 */
export interface Bijector {
  readonly name: string
  readonly inputRank: number
  readonly outputRank: number
  readonly domain: Support
  readonly codomain: Support
  readonly sign: 1 | -1 | 0
  forward(x: TensorLike): Value
  inverse(y: TensorLike): Value
  /** log |det dy/dx| at x, with the input's transformed axes reduced. */
  forwardLogDetJacobian(x: TensorLike): Value
  /** log |det dx/dy| at y (minus the forward log-determinant at the preimage). */
  inverseLogDetJacobian(y: TensorLike): Value
  /** The shape of forward(x) for x of `shape`. */
  forwardShape(shape: readonly number[]): number[]
  /** The shape of inverse(y) for y of `shape`. */
  inverseShape(shape: readonly number[]): number[]
}

/** The defining parts of a bijector; {@link defineBijector} fills in the rest. */
export interface BijectorSpec {
  readonly name: string
  readonly inputRank?: number
  readonly outputRank?: number
  readonly domain?: Support
  readonly codomain?: Support
  readonly sign?: 1 | -1 | 0
  forward(x: Value): Value
  inverse(y: Value): Value
  forwardLogDetJacobian(x: Value): Value
  inverseLogDetJacobian?(y: Value): Value
  forwardShape?(shape: readonly number[]): number[]
  inverseShape?(shape: readonly number[]): number[]
}

/**
 * Build a bijector from its defining parts. Defaults: ranks 0, real domain and codomain, `sign` 0, shapes unchanged,
 * and the inverse log-determinant as minus the forward one at the preimage.
 *
 * @param spec - The defining parts.
 * @returns The bijector.
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'omai-engine/dists'
 * import { add, item, mul } from 'omai-engine/tensor'
 * const double = bijectors.defineBijector({
 *   name: 'double', sign: 1, forward: (x) => mul(x, 2), inverse: (y) => mul(y, 0.5),
 *   forwardLogDetJacobian: (x) => add(mul(x, 0), Math.LN2),
 * })
 * item(double.inverse(3)) // => 1.5
 * ```
 */
export function defineBijector(spec: BijectorSpec): Bijector {
  const v = (x: TensorLike): Value => asTensor(x)
  return {
    name: spec.name,
    inputRank: spec.inputRank ?? 0,
    outputRank: spec.outputRank ?? spec.inputRank ?? 0,
    domain: spec.domain ?? { kind: 'real' },
    codomain: spec.codomain ?? { kind: 'real' },
    sign: spec.sign ?? 0,
    forward: (x) => spec.forward(v(x)),
    inverse: (y) => spec.inverse(v(y)),
    forwardLogDetJacobian: (x) => spec.forwardLogDetJacobian(v(x)),
    inverseLogDetJacobian: (y) =>
      spec.inverseLogDetJacobian
        ? spec.inverseLogDetJacobian(v(y))
        : neg(spec.forwardLogDetJacobian(spec.inverse(v(y)))),
    forwardShape: (shape) => (spec.forwardShape ? spec.forwardShape(shape) : [...shape]),
    inverseShape: (shape) => (spec.inverseShape ? spec.inverseShape(shape) : [...shape]),
  }
}

/**
 * The composition of bijectors, applied in order: `chain([a, b])` maps x to b(a(x)) (torch's ComposeTransform). Ranks
 * are tracked through the chain, and each part's log-determinant is summed over the axes the chain treats as one event
 * beyond that part's own.
 *
 * @param parts - The bijectors, first applied first.
 * @returns The composite bijector.
 * @throws OmaiError with code `value` for an empty chain.
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const b = bijectors.chain([bijectors.exp(), bijectors.affine(1, 2)])
 * item(b.forward(0)) // => 3
 * item(b.inverse(3)) // => 0
 * ```
 */
export function chain(parts: readonly Bijector[]): Bijector {
  const first = parts[0]
  const last = parts[parts.length - 1]
  if (first === undefined || last === undefined) throw new OmaiError('value', 'chain: needs at least one bijector')
  // The smallest input rank that every part's input rank fits inside, given the rank changes before it.
  let shift = 0
  let inputRank = 0
  for (const b of parts) {
    inputRank = Math.max(inputRank, b.inputRank - shift)
    shift += b.outputRank - b.inputRank
  }
  const outputRank = inputRank + shift
  const product = parts.reduce<number>((s, b) => s * b.sign, 1)
  const sign: 1 | -1 | 0 = product > 0 ? 1 : product < 0 ? -1 : 0
  return {
    name: `chain(${parts.map((b) => b.name).join(', ')})`,
    inputRank,
    outputRank,
    domain: first.domain,
    codomain: last.codomain,
    sign,
    forward: (x) => parts.reduce<Value>((v, b) => b.forward(v), asTensor(x)),
    inverse: (y) => parts.reduceRight<Value>((v, b) => b.inverse(v), asTensor(y)),
    forwardLogDetJacobian: (x) => {
      let v = asTensor(x)
      let rank = inputRank
      let total: Value | null = null
      for (const b of parts) {
        const term = sumRightmost(b.forwardLogDetJacobian(v), rank - b.inputRank)
        total = total === null ? term : add(total, term)
        v = b.forward(v)
        rank += b.outputRank - b.inputRank
      }
      return total ?? asTensor(0)
    },
    inverseLogDetJacobian: (y) => {
      let v = asTensor(y)
      let rank = outputRank
      let total: Value | null = null
      for (let i = parts.length - 1; i >= 0; i--) {
        const b = parts[i] as Bijector
        const term = sumRightmost(b.inverseLogDetJacobian(v), rank - b.outputRank)
        total = total === null ? term : add(total, term)
        v = b.inverse(v)
        rank += b.inputRank - b.outputRank
      }
      return total ?? asTensor(0)
    },
    forwardShape: (shape) => parts.reduce<number[]>((s, b) => b.forwardShape(s), [...shape]),
    inverseShape: (shape) => parts.reduceRight<number[]>((s, b) => b.inverseShape(s), [...shape]),
  }
}

/**
 * The inverse of a bijector: forward and inverse (and their log-determinants, ranks, supports and shapes) swapped.
 *
 * @param b - The bijector.
 * @returns Its inverse.
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(bijectors.invert(bijectors.exp()).forward(Math.E)) // => 1
 * ```
 */
export function invert(b: Bijector): Bijector {
  return {
    name: `invert(${b.name})`,
    inputRank: b.outputRank,
    outputRank: b.inputRank,
    domain: b.codomain,
    codomain: b.domain,
    sign: b.sign,
    forward: b.inverse,
    inverse: b.forward,
    forwardLogDetJacobian: b.inverseLogDetJacobian,
    inverseLogDetJacobian: b.forwardLogDetJacobian,
    forwardShape: b.inverseShape,
    inverseShape: b.forwardShape,
  }
}
