/**
 * The numeric operator: a factor given only by its log density over Gaussian variables, whose EP messages are found by
 * a panelled Gauss-Legendre product rule over the arguments (each with its own incoming message as the weight, then
 * zoomed onto where the tilted mass is). The fallback for the factors with no closed form: a robust likelihood, a
 * non-conjugate link, the clutter problem.
 */

import { OmaiError } from '../../base/index.ts'
import { gaussianGrid, gaussianGridOn } from '../_quadrature.ts'
import type { Arg, FactorOperator, OperatorContext, Params, Port, TiltedMoments } from '../types.ts'
import { allProper, gaussianOf, massOf, scalarOf } from './args.ts'

/** What {@link numericOperator} builds the operator from. */
export interface NumericOperatorOptions {
  readonly name: string
  readonly description?: string
  /** The number of Gaussian arguments (1 to 3: the grid is a product over them). */
  readonly arity: number
  /** log f at a point, one coordinate per argument, and the factor's parameters (its data), so one operator serves every factor of a kind. */
  readonly logDensity: (x: readonly number[], params: Params) => number
}

/**
 * An operator for a factor given by its log density over Gaussian variables. EP projects the quadrature moments of the
 * tilted distribution onto a Gaussian (the engine does the projection); the factor's mass under the incoming messages
 * is the same quadrature. Arguments may be observed. The operator supports a power (fractional and power EP, the
 * alpha-divergence projection): the factor raised to `OperatorContext.power` is what is integrated.
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
 * t?.moments[1]?.toFixed(6) // => '0.500000'
 * ```
 */
export function numericOperator(options: NumericOperatorOptions): FactorOperator {
  const n = options.arity
  if (!(Number.isInteger(n) && n >= 1 && n <= 3))
    throw new OmaiError('value', 'numericOperator: the arity must be 1, 2 or 3')
  const ports: Port[] = Array.from({ length: n }, (_, i) => ({ name: `x${i}`, accepts: ['gaussian'], constant: true }))
  const tilted = (c: OperatorContext): TiltedMoments => {
    const gs = c.args.map((a: Arg) => (a.value === undefined ? gaussianOf(a) : undefined))
    // Integrate over a product grid; returns the mass and, per free port, the grid's raw first and second moments.
    const integrateOver = (
      grids: readonly { x: Float64Array; w: Float64Array }[],
    ): { mass: number; m1: number[]; m2: number[]; xs: number[]; ws: number[] } => {
      const point = new Array<number>(n).fill(0)
      const m1 = new Array<number>(n).fill(0)
      const m2 = new Array<number>(n).fill(0)
      const xs: number[] = []
      const ws: number[] = []
      let mass = 0
      const walk = (d: number, w: number): void => {
        if (d === n) {
          const v = w * Math.exp(c.power * options.logDensity(point, c.params))
          if (!(v > 0)) return
          mass += v
          for (let k = 0; k < n; k++) {
            m1[k] = (m1[k] ?? 0) + v * (point[k] ?? 0)
            m2[k] = (m2[k] ?? 0) + v * (point[k] ?? 0) ** 2
          }
          xs.push(point[c.to] ?? 0)
          ws.push(v)
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
      return { mass, m1, m2, xs, ws }
    }
    const grids = c.args.map((a: Arg, d: number) => {
      if (a.value !== undefined) return { x: Float64Array.of(scalarOf(a)), w: Float64Array.of(1) }
      const g = gs[d] as { mean: number; variance: number }
      return gaussianGrid(g.mean, g.variance, n === 1 ? 64 : 24, 8)
    })
    let pass = integrateOver(grids)
    // The factor is usually much narrower than the messages: zoom each argument onto where the tilted mass is, and integrate again.
    if (pass.mass > 0) {
      const zoomed = c.args.map((a: Arg, d: number) => {
        const g = gs[d]
        if (a.value !== undefined || g === undefined) return grids[d] as { x: Float64Array; w: Float64Array }
        const mu = (pass.m1[d] ?? 0) / pass.mass
        const sd = Math.sqrt(Math.max((pass.m2[d] ?? 0) / pass.mass - mu * mu, 0))
        const spread = Math.sqrt(g.variance)
        const lo = Math.max(g.mean - 9 * spread, mu - 8 * Math.max(sd, 1e-3 * spread))
        const hi = Math.min(g.mean + 9 * spread, mu + 8 * Math.max(sd, 1e-3 * spread))
        return hi > lo
          ? gaussianGridOn(g.mean, g.variance, lo, hi, n === 1 ? 48 : n === 2 ? 14 : 6, n === 1 ? 10 : 8)
          : (grids[d] as { x: Float64Array; w: Float64Array })
      })
      const second = integrateOver(zoomed)
      if (second.mass > 0) pass = second
    }
    const mass = pass.mass
    if (!(mass > 0)) throw new OmaiError('value', `${options.name}: the factor has no mass under the incoming messages`)
    const mean = pass.ws.reduce((s, w, i) => s + w * (pass.xs[i] ?? 0), 0) / mass
    const variance = pass.ws.reduce((s, w, i) => s + w * ((pass.xs[i] ?? 0) - mean) ** 2, 0) / mass
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
    power: true,
    tilted,
    logEvidence: (c) => (allProper(c.args) ? tilted({ ...c, to: 0 }).logZ : Number.NaN),
  }
}
