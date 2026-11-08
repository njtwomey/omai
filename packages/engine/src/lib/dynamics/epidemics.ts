/**
 * Epidemics: the SEIR compartmental model (susceptible, exposed, infectious, recovered) as equations, its basic
 * reproduction number, and the same disease spreading between people who move about a room (some staying put, as
 * distancing) as a traced agent-based algorithm, the simulations that made the rounds in 2020.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { foldIn, uniform } from '../random/index.ts'
import { asTensor, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels } from '../trace/index.ts'
import { solveIvp } from './ode/index.ts'

/** The rates of the SEIR model, per day. */
export interface SeirRates {
  /** Transmission: new infections per infectious person per day in a fully susceptible population. */
  readonly beta: number
  /** One over the mean incubation (exposed) period. */
  readonly sigma: number
  /** One over the mean infectious period. */
  readonly gamma: number
}

/**
 * The SEIR equations' solution: dS/dt = -beta S I / N, dE/dt = beta S I / N - sigma E, dI/dt = sigma E - gamma I,
 * dR/dt = gamma I, as shares of the population over the days.
 *
 * @param rates - beta, sigma and gamma per day.
 * @param options - The share initially infectious (default 0.01), the days (default 120) and output points (default 241).
 * @returns The days and the four shares, and the basic reproduction number beta / gamma.
 * @throws OmaiError with code `value` for non-positive rates or days.
 * @concept compartmental-models
 * @example
 * ```ts
 * import { seirCurves } from 'omai-engine/dynamics'
 * const c = seirCurves({ beta: 0.5, sigma: 0.2, gamma: 0.1 })
 * [c.r0, Math.round((c.s.at(-1) ?? 0) * 100) / 100] // => [5, 0.01]
 * ```
 */
export function seirCurves(
  rates: SeirRates,
  options: { readonly infected?: number; readonly days?: number; readonly points?: number } = {},
): {
  readonly t: number[]
  readonly s: number[]
  readonly e: number[]
  readonly i: number[]
  readonly r: number[]
  readonly r0: number
} {
  const { beta, sigma, gamma } = rates
  const days = options.days ?? 120
  const points = options.points ?? 241
  const i0 = options.infected ?? 0.01
  if (!(beta > 0 && sigma > 0 && gamma > 0 && days > 0 && i0 > 0 && i0 < 1))
    throw new OmaiError('value', 'seirCurves: rates and days must be positive, the initial share in (0, 1)')
  const t = Array.from({ length: points }, (_, k) => (days * k) / (points - 1))
  const sol = solveIvp(
    (_t, y) => {
      const [s, e, i] = toFlat(y)
      const force = beta * (s ?? 0) * (i ?? 0)
      return asTensor([-force, force - sigma * (e ?? 0), sigma * (e ?? 0) - gamma * (i ?? 0), gamma * (i ?? 0)])
    },
    [0, days],
    [1 - i0, 0, i0, 0],
    { rtol: 1e-8, atol: 1e-10, tEval: t },
  )
  const col = (c: number): number[] => sol.ys.map((y) => toFlat(y)[c] ?? 0)
  return { t, s: col(0), e: col(1), i: col(2), r: col(3), r0: beta / gamma }
}

/** Options of {@link epidemicAlgorithm}. */
export interface EpidemicOptions {
  /** People in the room (default 300). */
  readonly people?: number
  /** People infectious at the start (default 3). */
  readonly infected?: number
  /** The room: `width` by `height` (default 100 by 100); people bounce off its walls. */
  readonly width?: number
  readonly height?: number
  /** The share of people who stay put (default 0: nobody distances). */
  readonly distancing?: number
  /** How far an infectious person reaches (default 2.5). */
  readonly radius?: number
  /** The chance a susceptible person within reach is infected, per step (default 0.08). */
  readonly transmission?: number
  /** Mean days exposed before becoming infectious, and mean days infectious (defaults 3 and 7). */
  readonly incubation?: number
  readonly infectious?: number
  /** Steps per day (default 4) and how far a moving person goes per step (default 1). */
  readonly stepsPerDay?: number
  readonly speed?: number
}

/** The people: positions, velocities, who stays put, and each one's state (0 S, 1 E, 2 I, 3 R). */
export interface EpidemicState {
  readonly x: Float64Array
  readonly y: Float64Array
  readonly vx: Float64Array
  readonly vy: Float64Array
  readonly still: Uint8Array
  readonly state: Uint8Array
}

/** The names of the SEIR states, in index order. */
export const SEIR_STATES: readonly ['susceptible', 'exposed', 'infectious', 'recovered'] = [
  'susceptible',
  'exposed',
  'infectious',
  'recovered',
]

/**
 * How many people are in each SEIR state.
 *
 * @param s - The people.
 * @returns The counts: susceptible, exposed, infectious, recovered.
 * @concept compartmental-models
 * @example
 * ```ts
 * import { seirCounts } from 'omai-engine/dynamics'
 * const z = new Float64Array(3)
 * seirCounts({ x: z, y: z, vx: z, vy: z, still: new Uint8Array(3), state: Uint8Array.of(0, 2, 2) }) // => [1, 0, 2, 0]
 * ```
 */
export function seirCounts(s: EpidemicState): [number, number, number, number] {
  const c: [number, number, number, number] = [0, 0, 0, 0]
  for (const v of s.state) c[v as 0 | 1 | 2 | 3]++
  return c
}

/**
 * A disease spreading between people moving about a room, as a traced algorithm with one step per `1 / stepsPerDay`
 * days. Moving people go in straight lines and bounce off the walls; a share stays put (distancing). Each step every
 * infectious person may infect each susceptible person within `radius` with chance `transmission`; the exposed become
 * infectious, and the infectious recover, at the rates of means `incubation` and `infectious` days. Every random
 * number comes from the keys.
 *
 * @param options - The people, room, distancing, reach, transmission, periods and time step.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for bad counts, shares or periods.
 * @concept compartmental-models
 * @example
 * ```ts
 * import { epidemicAlgorithm, seirCounts } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const s = runAlgorithm(epidemicAlgorithm({ people: 50 }), { seed: 0, maxSteps: 10 }).state
 * seirCounts(s).reduce((a, b) => a + b, 0) // => 50
 * ```
 */
export function epidemicAlgorithm(options: EpidemicOptions = {}): Algorithm<EpidemicState> {
  const n = options.people ?? 300
  const infected0 = options.infected ?? 3
  const w = options.width ?? 100
  const h = options.height ?? 100
  const distancing = options.distancing ?? 0
  const radius = options.radius ?? 2.5
  const p = options.transmission ?? 0.08
  const incubation = options.incubation ?? 3
  const infectious = options.infectious ?? 7
  const perDay = options.stepsPerDay ?? 4
  const speed = options.speed ?? 1
  if (
    !(Number.isInteger(n) && n >= 1 && Number.isInteger(infected0) && infected0 >= 1 && infected0 <= n) ||
    !(distancing >= 0 && distancing <= 1 && p >= 0 && p <= 1 && radius > 0 && incubation > 0 && infectious > 0) ||
    !(w > 0 && h > 0 && perDay > 0 && speed >= 0)
  )
    throw new OmaiError('value', 'epidemic: bad people, shares, reach or periods')
  const toInfectious = 1 - Math.exp(-1 / (incubation * perDay))
  const toRecovered = 1 - Math.exp(-1 / (infectious * perDay))
  return {
    name: 'dynamics/epidemic',
    config: { n, infected0, w, h, distancing, radius, p, incubation, infectious, perDay, speed } as Json,
    schema: channels({
      metrics: {
        susceptible: { description: 'people susceptible' },
        exposed: { description: 'people exposed (infected, not yet infectious)' },
        infectious: { description: 'people infectious' },
        recovered: { description: 'people recovered' },
      },
    }),
    init: (key) => {
      const u = uniform(key, [4 * n]).data
      const x = new Float64Array(n)
      const y = new Float64Array(n)
      const vx = new Float64Array(n)
      const vy = new Float64Array(n)
      const still = new Uint8Array(n)
      const state = new Uint8Array(n)
      for (let i = 0; i < n; i++) {
        x[i] = w * (u[4 * i] ?? 0)
        y[i] = h * (u[4 * i + 1] ?? 0)
        const a = 2 * Math.PI * (u[4 * i + 2] ?? 0)
        // The first `infected0` people start infectious, and they always move; of the rest, a share stays put.
        still[i] = i >= infected0 && (u[4 * i + 3] ?? 1) < distancing ? 1 : 0
        vx[i] = still[i] === 1 ? 0 : speed * Math.cos(a)
        vy[i] = still[i] === 1 ? 0 : speed * Math.sin(a)
        state[i] = i < infected0 ? 2 : 0
      }
      return { x, y, vx, vy, still, state }
    },
    step: (s, ctx) => {
      const x = Float64Array.from(s.x)
      const y = Float64Array.from(s.y)
      const vx = Float64Array.from(s.vx)
      const vy = Float64Array.from(s.vy)
      const state = Uint8Array.from(s.state)
      for (let i = 0; i < n; i++) {
        let xi = (x[i] ?? 0) + (vx[i] ?? 0)
        let yi = (y[i] ?? 0) + (vy[i] ?? 0)
        if (xi < 0 || xi > w) {
          vx[i] = -(vx[i] ?? 0)
          xi = Math.min(w, Math.max(0, xi))
        }
        if (yi < 0 || yi > h) {
          vy[i] = -(vy[i] ?? 0)
          yi = Math.min(h, Math.max(0, yi))
        }
        x[i] = xi
        y[i] = yi
      }
      const contact = uniform(foldIn(ctx.key, 0), [n]).data
      const progress = uniform(foldIn(ctx.key, 1), [n]).data
      const r2 = radius * radius
      // Infection: each susceptible person escapes every infectious person within reach with chance 1 - p.
      for (let i = 0; i < n; i++) {
        if (s.state[i] !== 0) continue
        let reached = 0
        for (let j = 0; j < n; j++) {
          if (s.state[j] !== 2) continue
          const dx = (x[j] ?? 0) - (x[i] ?? 0)
          const dy = (y[j] ?? 0) - (y[i] ?? 0)
          if (dx * dx + dy * dy <= r2) reached++
        }
        if (reached > 0 && (contact[i] ?? 1) < 1 - (1 - p) ** reached) state[i] = 1
      }
      for (let i = 0; i < n; i++) {
        if (s.state[i] === 1 && (progress[i] ?? 1) < toInfectious) state[i] = 2
        else if (s.state[i] === 2 && (progress[i] ?? 1) < toRecovered) state[i] = 3
      }
      return { x, y, vx, vy, still: s.still, state }
    },
    done: (s) => (s.state.some((v) => v === 1 || v === 2) ? undefined : 'converged'),
    observe: (s) => {
      const [sus, exp, inf, rec] = seirCounts(s)
      return { metrics: { susceptible: sus, exposed: exp, infectious: inf, recovered: rec } }
    },
  }
}
