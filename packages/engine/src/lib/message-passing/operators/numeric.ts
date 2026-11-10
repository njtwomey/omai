/**
 * The numeric operator: a factor given only by its log density over Gaussian variables, whose EP messages are found by
 * Gauss-Hermite quadrature of the tilted distribution (the product rule over the arguments, each with its own
 * incoming message as the weight). The fallback for the factors with no closed form: a robust likelihood, a
 * non-conjugate link, the clutter problem.
 */

import { OmaiError } from '../../base/index.ts'
import { gaussianNodes } from '../_quadrature.ts'
import type { Arg, FactorOperator, OperatorContext, Port, TiltedMoments } from '../types.ts'
import { allProper, gaussianOf, massOf, scalarOf } from './args.ts'

/** What {@link numericOperator} builds the operator from. */
export interface NumericOperatorOptions {
  readonly name: string
  readonly description?: string
  /** The number of Gaussian arguments (1 to 3: the grid has `points^arity` nodes). */
  readonly arity: number
  /** log f at a point, one coordinate per argument. */
  readonly logDensity: (x: readonly number[]) => number
  /** Quadrature nodes per argument (default 48, 24 or 12 for 1, 2 or 3 arguments). */
  readonly points?: number
}

/**
 * An operator for a factor given by its log density over Gaussian variables. EP projects the quadrature moments of the
 * tilted distribution onto a Gaussian (the engine does the projection); the factor's mass under the incoming messages
 * is the same quadrature. Arguments may be observed.
 *
 * @param options - The name, arity and log density.
 * @returns The operator, to be put in the registry the graph is compiled against.
 * @throws OmaiError with code `value` for an arity outside 1 to 3.
 * @concept factor-operators
 * @index function An EP operator for any density over Gaussian variables, by quadrature.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, numericOperator } from 'omai-engine/message-passing'
 * // exp(-x^2 / 2) as a factor: the tilted distribution of N(0, 1) is N(0, 1/2).
 * const op = numericOperator({ name: 'bell', arity: 1, logDensity: ([x]) => -0.5 * (x as number) ** 2 })
 * const t = op.tilted?.({ mode: 'ep', to: 0, args: [messageArg(gaussian, gaussianEta(0, 1))], params: {}, power: 1 })
 * t?.moments.map((v) => v.toFixed(6)) // => ['0.000000', '0.500000']
 * ```
 */
export function numericOperator(options: NumericOperatorOptions): FactorOperator {
  const n = options.arity
  if (!(Number.isInteger(n) && n >= 1 && n <= 3))
    throw new OmaiError('value', 'numericOperator: the arity must be 1, 2 or 3')
  const points = options.points ?? (n === 1 ? 48 : n === 2 ? 24 : 12)
  const ports: Port[] = Array.from({ length: n }, (_, i) => ({ name: `x${i}`, accepts: ['gaussian'], constant: true }))
  const tilted = (c: OperatorContext): TiltedMoments => {
    const grids = c.args.map((a: Arg) => {
      if (a.value !== undefined) return { x: Float64Array.of(scalarOf(a)), w: Float64Array.of(1) }
      const g = gaussianOf(a)
      return gaussianNodes(g.mean, g.variance, points)
    })
    const point = new Array<number>(n).fill(0)
    const xs: number[] = []
    const ws: number[] = []
    const walk = (d: number, w: number): void => {
      if (d === n) {
        const lf = options.logDensity(point)
        const v = w * Math.exp(lf)
        if (v > 0) {
          xs.push(point[c.to] ?? 0)
          ws.push(v)
        }
        return
      }
      const g = grids[d]
      if (g === undefined) return
      for (let i = 0; i < g.x.length; i++) {
        point[d] = g.x[i] ?? 0
        walk(d + 1, w * (g.w[i] ?? 0))
      }
    }
    walk(0, 1)
    const mass = ws.reduce((a, b) => a + b, 0)
    if (!(mass > 0)) throw new OmaiError('value', `${options.name}: the factor has no mass under the incoming messages`)
    const mean = ws.reduce((s, w, i) => s + w * (xs[i] ?? 0), 0) / mass
    const variance = ws.reduce((s, w, i) => s + w * ((xs[i] ?? 0) - mean) ** 2, 0) / mass
    let logZ = Math.log(mass)
    for (const a of c.args) logZ += massOf(a)
    return { moments: [mean, variance + mean * mean], logZ }
  }
  return {
    name: options.name,
    description: options.description ?? `a factor given by its log density over ${n} Gaussian variable(s)`,
    ports,
    support: { ep: 'numeric' },
    readsTarget: true,
    tilted,
    logEvidence: (c) => (allProper(c.args) ? tilted({ ...c, to: 0 }).logZ : Number.NaN),
  }
}
