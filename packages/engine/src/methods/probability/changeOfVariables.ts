/**
 * The change-of-variables formula watched: standard normal samples and a square grid are pushed through a bijection
 * that deforms from the identity to its full form, one notch per step, while the pushed density over the plane follows
 * from the base density and the Jacobian, p_X(x) = p_Z(g⁻¹(x)) |det J_{g⁻¹}(x)|. Four families with closed-form
 * inverses and determinants: a radial power (a hole opens at the origin), a tanh squash (mass piles up at the edges), a
 * banana shear (areas preserved, the Jacobian is one) and a linear map (one constant Jacobian everywhere).
 */

import { type CurveSet, curveSet, type DecisionSurface, decisionSurface } from '../../lib/exposition/index.ts'
import { type Key, normal } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { type Config, defineMethod, derived, Log, type Method, scalar } from '../../lib/trace/index.ts'

/** The families of bijection on offer. */
export type MapFamily = 'radial' | 'squash' | 'banana' | 'linear'

/** The state: the notch reached; everything else follows from it and the configuration. */
export interface ChangeOfVariablesState {
  readonly k: number
}

type Pair = readonly [number, number]

/** A bijection of the plane with a deformation parameter s in [0, 1] (0 the identity), in closed form. */
interface Bijection {
  readonly forward: (z: Pair, s: number) => Pair
  /** The inverse, or `undefined` where x is outside the image. */
  readonly inverse: (x: Pair, s: number) => Pair | undefined
  /** log |det J_g(z)|. */
  readonly logDet: (z: Pair, s: number) => number
}

const LOG_2PI = Math.log(2 * Math.PI)
/** The differential entropy of the standard normal on the plane. */
const BASE_ENTROPY = 1 + LOG_2PI

function bijectionOf(family: MapFamily, a: number): Bijection {
  switch (family) {
    case 'radial': {
      // r -> r^gamma with gamma below one: the centre spreads out and the density vanishes at the origin.
      const gamma = (s: number): number => 1 - 0.25 * s * a
      return {
        forward: ([z1, z2], s) => {
          const r = Math.hypot(z1, z2)
          const f = r === 0 ? 0 : r ** (gamma(s) - 1)
          return [z1 * f, z2 * f]
        },
        inverse: ([x1, x2], s) => {
          const r = Math.hypot(x1, x2)
          const f = r === 0 ? 0 : r ** (1 / gamma(s) - 1)
          return [x1 * f, x2 * f]
        },
        logDet: ([z1, z2], s) => {
          const g = gamma(s)
          const r = Math.max(Math.hypot(z1, z2), 1e-12)
          return Math.log(g) + 2 * (g - 1) * Math.log(r)
        },
      }
    }
    case 'squash': {
      const b = (s: number): number => s * a
      return {
        forward: ([z1, z2], s) => {
          const k = b(s)
          return k < 1e-9 ? [z1, z2] : [Math.tanh(k * z1) / k, Math.tanh(k * z2) / k]
        },
        inverse: ([x1, x2], s) => {
          const k = b(s)
          if (k < 1e-9) return [x1, x2]
          if (Math.abs(k * x1) >= 1 || Math.abs(k * x2) >= 1) return undefined
          return [Math.atanh(k * x1) / k, Math.atanh(k * x2) / k]
        },
        logDet: ([z1, z2], s) => {
          const k = b(s)
          return Math.log(1 - Math.tanh(k * z1) ** 2) + Math.log(1 - Math.tanh(k * z2) ** 2)
        },
      }
    }
    case 'banana':
      return {
        forward: ([z1, z2], s) => [z1, z2 + s * a * (z1 * z1 - 1)],
        inverse: ([x1, x2], s) => [x1, x2 - s * a * (x1 * x1 - 1)],
        logDet: () => 0,
      }
    case 'linear': {
      // (1 - s) I + s A, with A an upper-triangular shear and scale: the Jacobian is one constant.
      const m = (s: number): [number, number, number] => [1 - s + s * a, 0.8 * s, 1 - s + 0.6 * s * a]
      return {
        forward: ([z1, z2], s) => {
          const [m11, m12, m22] = m(s)
          return [m11 * z1 + m12 * z2, m22 * z2]
        },
        inverse: ([x1, x2], s) => {
          const [m11, m12, m22] = m(s)
          const z2 = x2 / m22
          return [(x1 - m12 * z2) / m11, z2]
        },
        logDet: (_z, s) => {
          const [m11, , m22] = m(s)
          return Math.log(m11 * m22)
        },
      }
    }
  }
}

const GRID_LINES = Array.from({ length: 13 }, (_, i) => -3 + 0.5 * i)
const GRID_POINTS = Array.from({ length: 61 }, (_, i) => -3 + 0.1 * i)

/** What a configuration and key build: the base samples, the bijection and the window the plane is drawn in. */
interface Problem {
  readonly samples: readonly Pair[]
  readonly map: Bijection
  readonly notches: number
  readonly window: readonly [Pair, Pair]
}

const PROBLEMS = new Map<string, Problem>()
function problemOf(c: Config, key: Key): Problem {
  const id = `${JSON.stringify(c)}|${key.words.join(',')}`
  const cached = PROBLEMS.get(id)
  if (cached !== undefined) return cached
  const n = c['samples'] as number
  const data = normal(key, [n, 2]).data
  const samples: Pair[] = Array.from({ length: n }, (_, i) => [data[2 * i] ?? 0, data[2 * i + 1] ?? 0])
  const map = bijectionOf(c['map'] as MapFamily, c['strength'] as number)
  // The window holds the grid's image at full deformation, with a margin, within ±12.
  let lo = [0, 0]
  let hi = [0, 0]
  for (const line of GRID_LINES)
    for (const p of GRID_POINTS)
      for (const z of [
        [line, p],
        [p, line],
      ] as Pair[]) {
        const [x1, x2] = map.forward(z, 1)
        lo = [Math.min(lo[0] ?? 0, x1), Math.min(lo[1] ?? 0, x2)]
        hi = [Math.max(hi[0] ?? 0, x1), Math.max(hi[1] ?? 0, x2)]
      }
  const pad = (a: number, b: number): Pair => {
    const margin = 0.1 * Math.max(b - a, 1)
    return [Math.max(-12, a - margin), Math.min(12, b + margin)]
  }
  const problem: Problem = {
    samples,
    map,
    notches: c['notches'] as number,
    window: [pad(lo[0] ?? -3, hi[0] ?? 3), pad(lo[1] ?? -3, hi[1] ?? 3)],
  }
  PROBLEMS.set(id, problem)
  return problem
}

const deformation = (p: Problem, k: number): number => Math.min(1, k / p.notches)

function gridRecord(p: Problem, s: number): CurveSet {
  const series = GRID_LINES.flatMap((line) => {
    const vertical = GRID_POINTS.map((t) => p.map.forward([line, t], s))
    const horizontal = GRID_POINTS.map((t) => p.map.forward([t, line], s))
    // One name per direction: the lines of a direction share a legend entry and a colour.
    return [
      { name: 'z₀ constant', x: vertical.map((q) => q[0]), y: vertical.map((q) => q[1]) },
      { name: 'z₁ constant', x: horizontal.map((q) => q[0]), y: horizontal.map((q) => q[1]) },
    ]
  })
  return curveSet({ series, xLabel: 'x₀', yLabel: 'x₁' })
}

function densityRecord(p: Problem, s: number): DecisionSurface {
  const res = 60
  const axis = ([lo, hi]: Pair): number[] => Array.from({ length: res }, (_, i) => lo + ((hi - lo) * i) / (res - 1))
  const xs = axis(p.window[0])
  const ys = axis(p.window[1])
  const values = ys.map((y) =>
    xs.map((x) => {
      const z = p.map.inverse([x, y], s)
      if (z === undefined) return 0
      const logp = -0.5 * (z[0] * z[0] + z[1] * z[1]) - LOG_2PI - p.map.logDet(z, s)
      // Capped for the picture: a vanishing Jacobian makes the density spike.
      return Math.min(3, Math.exp(logp))
    }),
  )
  const points = p.samples.slice(0, 300).map((z) => [...p.map.forward(z, s)])
  return decisionSurface({
    features: ['x₀', 'x₁'],
    x: xs,
    y: ys,
    values,
    field: { name: 'pushed density', smooth: true },
    rows: { points, features: ['x₀', 'x₁'] },
  })
}

/**
 * Standard normal samples and a grid pushed through a bijection deforming from the identity, one notch per step.
 *
 * Configuration: `map` (radial, squash, banana, linear), `strength` (how far the full deformation goes), `samples`
 * and `notches` (steps from the identity to the full map). Paths: `map/s` (the deformation reached),
 * `map/meanLogDet` (the mean log Jacobian over the samples), `map/entropy` (the pushed distribution's entropy, the
 * base's plus that mean), and the probes `map/grid` (the grid's image as curves) and `map/density` (the pushed density
 * over the plane with the pushed samples as rows).
 */
export const changeOfVariables: Method<ChangeOfVariablesState, Config> = defineMethod<ChangeOfVariablesState, Config>({
  name: 'probability/change-of-variables',
  config: dict({
    map: choice(['radial', 'squash', 'banana', 'linear'] as const, { default: 'radial' }),
    strength: real(0.1, 3, { default: 1 }),
    samples: int(50, 2000, { default: 400 }),
    notches: int(10, 300, { default: 60 }),
  }),
  make: (c, key) => {
    const p = problemOf(c, key)
    return {
      name: 'change-of-variables',
      config: { map: c['map'] as string, strength: c['strength'] as number, notches: p.notches },
      init: () => ({ k: 0 }),
      step: (state, ctx) => {
        const k = state.k + 1
        const s = deformation(p, k)
        const meanLogDet = p.samples.reduce((acc, z) => acc + p.map.logDet(z, s), 0) / p.samples.length
        ctx.log('map/s', s)
        ctx.log('map/meanLogDet', meanLogDet)
        ctx.log('map/entropy', BASE_ENTROPY + meanLogDet)
        return { k }
      },
      done: (state) => (state.k >= p.notches ? 'converged' : undefined),
      serialise: (state) => ({ k: state.k }),
      deserialise: (j) => ({ k: (j as { readonly k: number }).k }),
    }
  },
  contract: (c, key) => {
    const p = problemOf(c, key)
    return {
      'map/s': scalar({ description: 'the deformation reached, from 0 (the identity) to 1 (the full map)' }),
      'map/meanLogDet': scalar({ description: 'the mean log |det J| of the map over the base samples' }),
      'map/entropy': scalar({ description: 'the differential entropy of the pushed distribution' }),
      'map/grid': derived<ChangeOfVariablesState>({
        description: 'the image of a square grid under the map at this step',
        compute: (s) => gridRecord(p, deformation(p, s.k)),
      }),
      'map/density': derived<ChangeOfVariablesState>({
        description: 'the pushed density over the plane, with the pushed samples',
        cost: 'dear',
        compute: (s) => densityRecord(p, deformation(p, s.k)),
      }),
    }
  },
  defaults: { 'map/s': Log(), 'map/meanLogDet': Log(), 'map/entropy': Log() },
})
