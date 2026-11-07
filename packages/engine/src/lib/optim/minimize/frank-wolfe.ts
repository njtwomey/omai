/**
 * The Frank-Wolfe (conditional gradient) method, which minimises over a compact convex set without projections by
 * stepping towards the vertex a linear minimisation oracle returns, with away and pairwise steps and an active set.
 * Oracles for the simplex, L1, L2 and nuclear-norm balls and the box are included.
 */

import { argmaxOf, argminOf, MdmlError } from '../../base/index.ts'
import { daxpy, ddot, dnrm2, svd } from '../../numerics/linalg/index.ts'
import { fromStorage, primalOf, toFlat } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import {
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

/** An extreme point returned by a linear minimisation oracle, with an identity for the away-step bookkeeping. */
export interface FrankWolfeAtom {
  readonly vertex: Float64Array
  /** A stable key for polytope vertices; absent for smooth sets (no away steps there). */
  readonly id?: string
}

/** A linear minimisation oracle: argmin over the feasible set of g . s. */
export type Lmo = (grad: Float64Array) => FrankWolfeAtom

/**
 * The linear minimisation oracle of the simplex {s >= 0, sum s = radius}: the vertex radius e_i at the smallest
 * gradient entry.
 *
 * @param radius - The simplex's total (default 1).
 * @returns The oracle.
 * @concept frank-wolfe
 * @example
 * ```ts
 * import { simplexLmo } from 'mdml-engine/optim'
 * Array.from(simplexLmo()(new Float64Array([0.3, -1, 2])).vertex) // => [0, 1, 0]
 * ```
 */
export function simplexLmo(radius = 1): Lmo {
  return (g) => {
    const i = argminOf(g)
    const vertex = new Float64Array(g.length)
    vertex[i] = radius
    return { vertex, id: `${i}` }
  }
}

/**
 * The linear minimisation oracle of the L1 ball {|s|_1 <= r}: -r sign(g_i) e_i at the largest |g_i|.
 *
 * @param r - The radius (default 1).
 * @returns The oracle.
 * @example
 * ```ts
 * import { l1BallLmo } from 'mdml-engine/optim'
 * Array.from(l1BallLmo(2)(new Float64Array([0.3, -1, 0.5])).vertex) // => [0, 2, 0]
 * ```
 */
export function l1BallLmo(r = 1): Lmo {
  return (g) => {
    const i = argmaxOf(Float64Array.from(g, Math.abs))
    const sgn = (g[i] ?? 0) > 0 ? -1 : 1
    const vertex = new Float64Array(g.length)
    vertex[i] = sgn * r
    return { vertex, id: `${i}${sgn > 0 ? '+' : '-'}` }
  }
}

/**
 * The linear minimisation oracle of the box lo <= s <= hi: each coordinate at the bound its gradient points away from.
 *
 * @param lo - Lower bounds (a number or one per coordinate).
 * @param hi - Upper bounds.
 * @returns The oracle.
 * @example
 * ```ts
 * import { boxLmo } from 'mdml-engine/optim'
 * Array.from(boxLmo(0, 1)(new Float64Array([0.3, -1])).vertex) // => [0, 1]
 * ```
 */
export function boxLmo(lo: number | ArrayLike<number>, hi: number | ArrayLike<number>): Lmo {
  const at = (b: number | ArrayLike<number>, i: number): number => (typeof b === 'number' ? b : (b[i] ?? 0))
  return (g) => {
    let id = ''
    const vertex = Float64Array.from(g, (v, i) => {
      id += v > 0 ? '0' : '1'
      return v > 0 ? at(lo, i) : at(hi, i)
    })
    return { vertex, id }
  }
}

/**
 * The linear minimisation oracle of the L2 ball {|s - c|_2 <= r}: c - r g / |g| (a smooth set: no away steps).
 *
 * @param r - The radius (default 1).
 * @param center - The centre (default 0).
 * @returns The oracle.
 * @example
 * ```ts
 * import { l2BallLmo } from 'mdml-engine/optim'
 * Array.from(l2BallLmo(5)(new Float64Array([3, 4])).vertex) // => [-3, -4]
 * ```
 */
export function l2BallLmo(r = 1, center?: ArrayLike<number>): Lmo {
  return (g) => {
    const norm = dnrm2(g)
    return { vertex: Float64Array.from(g, (v, i) => (center?.[i] ?? 0) - (norm === 0 ? 0 : (r * v) / norm)) }
  }
}

/**
 * The linear minimisation oracle of the nuclear-norm ball {|S|_* <= r} of `rows` x `cols` matrices (flattened row
 * major): -r u_1 v_1^T from the top singular pair of the gradient, the rank-one step behind matrix completion.
 *
 * @param r - The radius.
 * @param rows - Matrix rows.
 * @param cols - Matrix columns.
 * @returns The oracle.
 * @example
 * ```ts
 * import { nuclearBallLmo } from 'mdml-engine/optim'
 * Array.from(nuclearBallLmo(1, 2, 2)(new Float64Array([2, 0, 0, 1])).vertex) // => [-1, 0, 0, 0]
 * ```
 */
export function nuclearBallLmo(r: number, rows: number, cols: number): Lmo {
  return (g) => {
    const { u, vh } = svd(fromStorage(Float64Array.from(g), [rows, cols], 'float64'))
    const uu = toFlat(primalOf(u))
    const vv = toFlat(primalOf(vh))
    const k = Math.min(rows, cols)
    const vertex = new Float64Array(rows * cols)
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) vertex[i * cols + j] = -r * (uu[i * k] ?? 0) * (vv[j] ?? 0)
    return { vertex }
  }
}

/** Options of {@link frankWolfe}. */
export interface FrankWolfeOptions extends VectorMethodOptions {
  /**
   * `standard` (step 2 / (k + 2)), `lineSearch` (the adaptive short step of Pedregosa et al. 2020), `away` (away steps,
   * Guelat and Marcotte; linear convergence on polytopes) or `pairwise` (mass moved from the away vertex to the
   * Frank-Wolfe vertex). Default `lineSearch`.
   */
  readonly variant?: 'standard' | 'lineSearch' | 'away' | 'pairwise'
  /** Initial estimate of the gradient's Lipschitz constant for the short step (estimated by default). */
  readonly lipschitz?: number
  /** Stop once the Frank-Wolfe gap falls to this (default 1e-10). */
  readonly gapTol?: number
}

/** One atom of the active set with its convex weight. */
export interface ActiveAtom {
  readonly id: string
  readonly vertex: Float64Array
  readonly weight: number
}

/** State of {@link frankWolfeAlgorithm}. */
export interface FrankWolfeState extends VectorState {
  /** The Frank-Wolfe duality gap g . (x - s), an upper bound on f(x) - f*. */
  readonly gap: number
  readonly lipschitz: number
  readonly active: readonly ActiveAtom[]
  readonly kind: 'start' | 'frankWolfe' | 'away' | 'drop' | 'pairwise'
}

/**
 * The Frank-Wolfe (conditional gradient) method over a compact convex set given by its linear minimisation oracle, as
 * a traced algorithm: projection-free, each step moves towards the oracle's vertex (or away from an active vertex).
 * Records the Frank-Wolfe gap, the step and the kind of step.
 *
 * @param f - The smooth objective.
 * @param x0 - A feasible starting point (for away and pairwise steps it is the first atom).
 * @param lmo - The linear minimisation oracle.
 * @param options - Variant, Lipschitz estimate and gap tolerance.
 * @returns The algorithm.
 * @concept frank-wolfe
 * @example
 * ```ts
 * import { frankWolfeAlgorithm, simplexLmo } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * frankWolfeAlgorithm((x) => sum(square(x as Value)), [1, 0], simplexLmo()).name // => 'frankWolfe'
 * ```
 */
export function frankWolfeAlgorithm(
  f: Objective,
  x0: Parameters,
  lmo: Lmo,
  options: FrankWolfeOptions = {},
): Algorithm<FrankWolfeState> {
  const obj = flatObjective(f, x0)
  const variant = options.variant ?? 'lineSearch'
  const gapTol = options.gapTol ?? 1e-10
  const activeSet = variant === 'away' || variant === 'pairwise'
  const estimateL = (x: Float64Array, g: Float64Array, d: Float64Array): number => {
    const dn = dnrm2(d)
    if (dn === 0) return 1
    const eps = 1e-3 / dn
    const g2 = obj.valueAndGrad(daxpy(eps, d, x)).grad
    return Math.max(dnrm2(daxpy(-1, g, g2)) / (eps * dn), 1e-12)
  }
  // When the start is a vertex, the oracle at -x0 returns it: use its id so the atom is not duplicated later.
  const startId = (x: Float64Array): string => {
    const s = lmo(Float64Array.from(x, (v) => -v))
    return s.id !== undefined && dnrm2(daxpy(-1, x, s.vertex)) <= 1e-12 * Math.max(1, dnrm2(x)) ? s.id : 'x0'
  }
  return {
    name: 'frankWolfe',
    config: configOf({ ...options, variant, gapTol }),
    schema: vectorSchema({
      metrics: {
        gap: { description: 'Frank-Wolfe duality gap', direction: 'lower' },
        lipschitz: { description: 'current Lipschitz estimate of the short step' },
        activeAtoms: { description: 'size of the active set' },
      },
      messages: { step: { description: 'the kind of step taken: frankWolfe, away, drop or pairwise' } },
    }),
    init: () => {
      const x = Float64Array.from(obj.x0)
      const { value, grad } = obj.valueAndGrad(x)
      return {
        x,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        gap: Infinity,
        lipschitz: options.lipschitz ?? Number.NaN,
        active: activeSet ? [{ id: startId(x), vertex: x, weight: 1 }] : [],
        kind: 'start',
      }
    },
    step: (s, ctx) => {
      const g = s.grad
      const fw = lmo(g)
      const dFw = daxpy(-1, s.x, fw.vertex)
      const gap = -ddot(g, dFw)
      if (gap <= gapTol) return { ...s, gap, stop: { status: 'converged', message: 'Frank-Wolfe gap below tolerance' } }
      let d = dFw
      let gammaMax = 1
      let kind: FrankWolfeState['kind'] = 'frankWolfe'
      let away: ActiveAtom | undefined
      if (activeSet) {
        if (fw.id === undefined)
          throw new MdmlError('value', 'away and pairwise steps need a polytope oracle with vertex ids')
        away = s.active.reduce((best, a) => (ddot(g, a.vertex) > ddot(g, best.vertex) ? a : best))
        const dAway = daxpy(-1, away.vertex, s.x)
        if (variant === 'pairwise') {
          d = daxpy(-1, away.vertex, fw.vertex)
          gammaMax = away.weight
          kind = 'pairwise'
        } else if (-ddot(g, dAway) > gap) {
          d = dAway
          gammaMax = away.weight / (1 - away.weight)
          kind = 'away'
        }
      }
      const slope = -ddot(g, d)
      const dd = ddot(d, d)
      let nfev = s.nfev
      let ngev = s.ngev
      let gamma: number
      let L = s.lipschitz
      let x: Float64Array
      let value: number
      if (variant === 'standard') {
        gamma = 2 / (ctx.t + 2)
        x = daxpy(gamma, d, s.x)
        value = Number.NaN
      } else {
        if (!Number.isFinite(L)) {
          L = estimateL(s.x, g, d)
          ngev += 1
        } else L *= 0.9
        for (;;) {
          gamma = Math.min(slope / (L * dd), gammaMax)
          x = daxpy(gamma, d, s.x)
          value = obj.value(x)
          nfev += 1
          if (value <= s.value - gamma * slope + 0.5 * gamma * gamma * L * dd + 1e-14 * Math.abs(s.value) || L > 1e20)
            break
          L *= 2
        }
      }
      let active = s.active
      if (activeSet && away !== undefined) {
        const awayId = away.id
        const fwId = fw.id as string
        if (kind === 'frankWolfe') {
          active = active.map((a) => ({ ...a, weight: (1 - gamma) * a.weight }))
          const found = active.find((a) => a.id === fwId)
          active = found
            ? active.map((a) => (a.id === fwId ? { ...a, weight: a.weight + gamma } : a))
            : [...active, { id: fwId, vertex: fw.vertex, weight: gamma }]
        } else if (kind === 'away') {
          active = active.map((a) => ({ ...a, weight: (1 + gamma) * a.weight - (a.id === awayId ? gamma : 0) }))
          if (gamma >= gammaMax - 1e-15) kind = 'drop'
        } else {
          active = active.map((a) => (a.id === awayId ? { ...a, weight: a.weight - gamma } : a))
          const found = active.find((a) => a.id === fwId)
          active = found
            ? active.map((a) => (a.id === fwId ? { ...a, weight: a.weight + gamma } : a))
            : [...active, { id: fwId, vertex: fw.vertex, weight: gamma }]
        }
        active = active.filter((a) => a.weight > 1e-14)
      }
      const r = obj.valueAndGrad(x)
      return {
        x,
        value: r.value,
        grad: r.grad,
        nfev: nfev + 1,
        ngev: ngev + 1,
        nhev: 0,
        step: gamma,
        gap,
        lipschitz: L,
        active,
        kind,
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { gap: s.gap, lipschitz: s.lipschitz, activeAtoms: s.active.length },
        messages: [{ kind: 'step', payload: s.kind }],
      }),
  }
}

/**
 * Minimise a smooth function over a compact convex set by Frank-Wolfe.
 *
 * @param f - The objective.
 * @param x0 - A feasible starting point.
 * @param lmo - The linear minimisation oracle (`simplexLmo`, `l1BallLmo`, `boxLmo`, `l2BallLmo`, `nuclearBallLmo`, or
 *   your own).
 * @param options - Variant, Lipschitz estimate, gap tolerance, iteration cap (default 1000) and run options.
 * @returns The result.
 * @concept frank-wolfe
 * @example
 * ```ts
 * import { frankWolfe, simplexLmo } from 'mdml-engine/optim'
 * import { square, sub, sum, toArray, type Value } from 'mdml-engine/tensor'
 * // The projection of (1, 0.4, -1) onto the simplex is (0.8, 0.2, 0).
 * const r = frankWolfe((x) => sum(square(sub(x as Value, [1, 0.4, -1]))), [1, 0, 0], simplexLmo(), { variant: 'away' })
 * toArray(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8) // => [0.8, 0.2, 0]
 * ```
 */
export function frankWolfe(f: Objective, x0: Parameters, lmo: Lmo, options: FrankWolfeOptions = {}): MinimizeResult {
  return runVector(frankWolfeAlgorithm(f, x0, lmo, options), flatObjective(f, x0), options, 1000)
}
